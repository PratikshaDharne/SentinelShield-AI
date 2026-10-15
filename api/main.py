"""SentinelShield AI gateway (FastAPI).
Public:  POST /v1/guard  /v1/rewrite  /v1/chat  /v1/chat/completions  /v1/guard/tool   GET /v1/health /v1/metrics
Owner :  POST /owner/login   GET /owner/overview|audit|review|keys|policy_versions|evaluation   (JWT)"""
from __future__ import annotations
import json, re, time, uuid, datetime as dt
from pathlib import Path
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, text
from api import db, llm
from api.db import SessionLocal, AuditLog, ReviewItem, ApiClient, PolicyVersion, RLEvent
from api.pipeline import Guardrail, GuardResult, redact
from api.rl_sim import Simulator
from policy.hybrid import ACTIONS
from api.schemas import *
from api.security import client_required, owner_required, client_or_owner, login_owner, new_api_key
from configs.settings import settings

app = FastAPI(title=settings.app_name, version=settings.policy_version,
              description="Real-time guardrail firewall for LLM applications (Allow / Rewrite / Block).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

REFUSAL = "I can't help with that request."


@app.on_event("startup")
def _startup():
    db.init_db()
    Guardrail.get()


# ----------------------------------------------------------------------------------- helpers
def _eval(prompt: str, history: list[str] | None = None, session_key=None) -> GuardResult:
    """Guardrail.evaluate + remember the live-RL event so _audit can persist it (owner labels later become rewards)."""
    o: dict = {}
    res = Guardrail.get().evaluate(prompt, history, session_key=session_key, rl_out=o)
    res._rl = o.get("event")
    return res


def _audit(res: GuardResult, endpoint: str, client_id: int | None, llm_info: dict | None = None) -> int:
    rw = res.rewrite or {}
    with SessionLocal() as s:
        a = AuditLog(request_id=uuid.uuid4().hex[:16], client_id=client_id, endpoint=endpoint, prompt_hash=res.prompt_hash,
                     prompt_len=res.prompt_len, decision=res.decision, category=res.category, risk_score=res.risk_score,
                     confidence=res.confidence, reason_codes=res.reason_codes, flags=res.flags, policy_version=res.policy_version,
                     latency_ms=res.latency_ms, stage_latency=res.stage_latency_ms, rewrite_status=rw.get("status"),
                     rewrite_similarity=rw.get("similarity"), rewrite_attempts=rw.get("attempts"),
                     llm_provider=(llm_info or {}).get("provider", "none"), llm_latency_ms=(llm_info or {}).get("latency_ms"),
                     preview=None)
        s.add(a); s.flush()
        ev = getattr(res, "_rl", None)
        if ev is not None:
            s.add(RLEvent(audit_id=a.id, mode=ev["mode"], acting=ev["acting"], explored=ev["explored"], base=ev["base"], agent=ev["agent"],
                          exec_=ev["exec"], eff=ev["eff"], risk=ev["risk"], q=ev["q"], x=[float(v) for v in ev["x"]],
                          legal=[float(v) for v in ev["legal"]], rw=ev.get("rw")))
        s.commit(); return a.id


def _enqueue_review(audit_id: int, res: GuardResult, text: str):
    """Ambiguous / low-confidence / classifier-only blocks go to the false-positive review queue (redacted, short)."""
    if not res.needs_review:
        return
    with SessionLocal() as s:
        s.add(ReviewItem(audit_id=audit_id, decision=res.decision, category=res.category, risk_score=res.risk_score,
                         preview=redact(text)[:160]))
        s.commit()


def _history(h: list[Turn]) -> list[str]:
    return [t.content for t in h if t.role == "user"][-6:]


def validate_output(text: str, client_system: str | None) -> tuple[str, list[str]]:
    """Output validator: redact secrets and stop verbatim leakage of the client's system prompt."""
    flags = []
    clean = redact(text)
    if clean != text:
        flags.append("SECRETS_REDACTED")
    if client_system and len(client_system) >= 40:
        chunks = [client_system[i:i + 40] for i in range(0, len(client_system) - 39, 20)]
        if any(c.lower() in clean.lower() for c in chunks):
            return REFUSAL, flags + ["SYSTEM_PROMPT_LEAK_BLOCKED"]
    return clean, flags


def _blocked_payload(res: GuardResult, rid: str) -> dict:
    return {"decision": "BLOCK", "answer": REFUSAL, "reason_codes": res.reason_codes, "request_id": rid}


# ----------------------------------------------------------------------------------- public API
@app.post("/v1/guard")
def guard(req: GuardRequest, client: ApiClient = Depends(client_required)):
    res = _eval(req.prompt, _history(req.history), client.id)
    aid = _audit(res, "guard", client.id); _enqueue_review(aid, res, req.prompt)
    out = res.public(); out["audit_id"] = aid
    return out


@app.post("/v1/rewrite")
def rewrite(req: GuardRequest, client: ApiClient = Depends(client_required)):
    """Controlled rewriting: returns a validated rewrite, or BLOCK if the legitimate intent cannot be safely preserved."""
    g = Guardrail.get(); res = _eval(req.prompt, _history(req.history), client.id)
    if res.decision == "ALLOW":      # nothing to neutralise; caller may use the prompt unchanged
        res.reason_codes = ["NO_REWRITE_NEEDED"] + res.reason_codes
    aid = _audit(res, "rewrite", client.id); _enqueue_review(aid, res, req.prompt)
    return {"decision": res.decision, "rewritten_prompt": res.rewritten_prompt, "rewrite": res.rewrite,
            "reason_codes": res.reason_codes, "risk_score": res.risk_score, "audit_id": aid}


@app.post("/v1/chat")
def chat(req: ChatRequest, client: ApiClient = Depends(client_required)):
    """Gateway mode: guard -> (our rewrite) -> answer from Gemini (Groq fallback) -> output validation."""
    g = Guardrail.get(); res = _eval(req.prompt, _history(req.history), client.id)
    rid = uuid.uuid4().hex[:12]
    if res.decision == "BLOCK" or not res.final_prompt:
        aid = _audit(res, "chat", client.id); _enqueue_review(aid, res, req.prompt)
        return {**_blocked_payload(res, rid), "risk_score": res.risk_score, "category": res.category, "audit_id": aid}
    msgs = [t.model_dump() for t in req.history if t.role in ("user", "assistant")] + [{"role": "user", "content": res.final_prompt}]
    try:
        a = llm.answer(msgs, req.system)
    except llm.LLMError as e:
        aid = _audit(res, "chat", client.id)
        raise HTTPException(502, {"error": "LLM_UNAVAILABLE", "detail": str(e), "decision": res.decision, "audit_id": aid})
    text, oflags = validate_output(a["text"], req.system)
    aid = _audit(res, "chat", client.id, a); _enqueue_review(aid, res, req.prompt)
    return {"decision": res.decision, "answer": text, "rewritten_prompt": res.rewritten_prompt, "reason_codes": res.reason_codes,
            "risk_score": res.risk_score, "category": res.category, "llm_provider": a["provider"],
            "output_flags": oflags, "guard_latency_ms": res.latency_ms, "llm_latency_ms": a["latency_ms"], "audit_id": aid}


@app.post("/v1/chat/completions")
def openai_compat(req: OpenAIChatRequest, response: Response, client: ApiClient = Depends(client_required)):
    """OpenAI-compatible route: point an existing app's base_url here (non-streaming)."""
    users = [m for m in req.messages if m.role == "user"]
    if not users:
        raise HTTPException(400, "no user message")
    last = users[-1].content
    sys_prompt = "\n".join(m.content for m in req.messages if m.role == "system") or None
    prior = [m for m in req.messages[:-1] if m.role in ("user", "assistant")] if req.messages[-1].role == "user" else \
            [m for m in req.messages if m.role in ("user", "assistant")][:-1]
    g = Guardrail.get(); res = _eval(last, [m.content for m in prior if m.role == "user"][-6:], client.id)
    response.headers["X-SentinelShield-Decision"] = res.decision
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:20], int(time.time())
    def wrap(text, finish="stop", provider="none"):
        return {"id": cid, "object": "chat.completion", "created": created, "model": req.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                "sentinelshield": {"decision": res.decision, "risk_score": res.risk_score, "reason_codes": res.reason_codes,
                                   "rewritten": res.decision == "REWRITE", "llm_provider": provider}}
    if res.decision == "BLOCK" or not res.final_prompt:
        aid = _audit(res, "chat_completions", client.id); _enqueue_review(aid, res, last)
        return wrap(REFUSAL, "content_filter")
    msgs = [m.model_dump() for m in prior] + [{"role": "user", "content": res.final_prompt}]
    try:
        a = llm.answer(msgs, sys_prompt)
    except llm.LLMError as e:
        _audit(res, "chat_completions", client.id)
        raise HTTPException(502, {"error": {"message": "LLM unavailable: " + str(e), "type": "upstream_error"}})
    text, _ = validate_output(a["text"], sys_prompt)
    aid = _audit(res, "chat_completions", client.id, a); _enqueue_review(aid, res, last)
    return wrap(text, "stop", a["provider"])


@app.post("/v1/guard/tool")
def guard_tool(req: ToolRequest, client: ApiClient = Depends(client_required)):
    return Guardrail.get().guard_tool(req.tool_name, req.instruction, req.arguments, req.sensitivity)


@app.get("/v1/health")
def health():
    ok_db = True
    try:
        with SessionLocal() as s:
            s.execute(text("SELECT 1"))
    except Exception:
        ok_db = False
    return {"status": "ok" if ok_db else "degraded", "database": ok_db, **Guardrail.get().health(),
            "answer_llm": {"gemini_configured": bool(settings.gemini_api_key), "groq_fallback_configured": bool(settings.groq_api_key)}}


def aggregate(minutes: int = 60, client_id: int | None = None) -> dict:
    since = db.now() - dt.timedelta(minutes=minutes)
    with SessionLocal() as s:
        q = s.query(AuditLog).filter(AuditLog.ts >= since)
        if client_id is not None:
            q = q.filter(AuditLog.client_id == client_id)
        rows = q.order_by(AuditLog.ts).all()
    lat = np.array([r.latency_ms for r in rows if r.latency_ms is not None]) if rows else np.array([])
    pct = {f"p{p}": round(float(np.percentile(lat, p)), 2) for p in (50, 95, 99)} if lat.size else {"p50": 0, "p95": 0, "p99": 0}
    per_min: dict[str, int] = {}
    for r in rows:
        k = r.ts.strftime("%Y-%m-%d %H:%M"); per_min[k] = per_min.get(k, 0) + 1
    def count(attr):
        d: dict[str, int] = {}
        for r in rows:
            d[getattr(r, attr) or "none"] = d.get(getattr(r, attr) or "none", 0) + 1
        return d
    rw = [r for r in rows if r.rewrite_status]
    return {"window_minutes": minutes, "total": len(rows), "requests_per_minute": round(len(rows) / max(minutes, 1), 3),
            "decisions": count("decision"), "categories": count("category"), "latency_ms": pct,
            "rewrite": {"attempted": len(rw), "validated": sum(r.rewrite_status == "REWRITE" for r in rw),
                        "failed_to_block": sum(r.rewrite_status == "BLOCK" for r in rw),
                        "avg_similarity": round(float(np.mean([r.rewrite_similarity for r in rw if r.rewrite_similarity])), 3)
                        if any(r.rewrite_similarity for r in rw) else None},
            "llm_providers": count("llm_provider"), "series_per_minute": per_min}


@app.get("/v1/metrics")
def metrics(minutes: int = 60, who=Depends(client_or_owner)):
    kind, ident = who
    return aggregate(minutes, ident.id if kind == "client" else None)


# ----------------------------------------------------------------------------------- owner API
@app.post("/owner/login")
def owner_login(req: LoginRequest):
    tok = login_owner(req.username, req.password)
    if not tok:
        time.sleep(0.6)
        raise HTTPException(401, "invalid credentials")
    return {"access_token": tok, "token_type": "bearer"}


@app.get("/owner/overview")
def owner_overview(minutes: int = 60, _=Depends(owner_required)):
    return {**aggregate(minutes), "health": health()}


@app.get("/owner/audit")
def owner_audit(limit: int = 200, _=Depends(owner_required)):
    with SessionLocal() as s:
        rows = s.query(AuditLog).order_by(AuditLog.id.desc()).limit(min(limit, 1000)).all()
    return [{c.name: (getattr(r, c.name).isoformat() if isinstance(getattr(r, c.name), dt.datetime) else getattr(r, c.name))
             for c in AuditLog.__table__.columns} for r in rows]


@app.get("/owner/review")
def owner_review(status: str = "pending", _=Depends(owner_required)):
    with SessionLocal() as s:
        rows = s.query(ReviewItem).filter_by(status=status).order_by(ReviewItem.id.desc()).limit(200).all()
    return [{"id": r.id, "ts": r.ts.isoformat(), "decision": r.decision, "category": r.category, "risk_score": r.risk_score,
             "preview": r.preview, "status": r.status, "note": r.note} for r in rows]


@app.post("/owner/review/{item_id}")
def owner_review_update(item_id: int, upd: ReviewUpdate, _=Depends(owner_required)):
    with SessionLocal() as s:
        r = s.get(ReviewItem, item_id)
        if not r:
            raise HTTPException(404, "not found")
        r.status, r.note, r.resolved_at = upd.status, upd.note, db.now()
        aid = r.audit_id
        s.commit()
    lab = {"confirmed": "CONFIRM", "false_positive": "ALLOW", "should_block": "BLOCK", "should_rewrite": "REWRITE"}.get(upd.status)
    learned = _rl_label(aid, lab) if (lab and aid) else None
    return {"ok": True, "rl_learned": learned is not None}


# ----------------------------------------------------------------------------------- live RL (owner)
def _rl_label(audit_id: int, label: str) -> dict | None:
    """Owner label -> full-information reward -> one online DQN update. Returns the learning record or None."""
    mgr = Guardrail.get().rl
    if mgr is None or mgr.agent is None or label not in ("CONFIRM", "ALLOW", "REWRITE", "BLOCK"):
        return None
    with SessionLocal() as s:
        r = s.query(RLEvent).filter_by(audit_id=audit_id).first()
        if r is None or r.label:
            return None
        ev = {"x": np.array(r.x, np.float32), "legal": np.array(r.legal, np.float32), "q": r.q, "base": r.base, "agent": r.agent, "exec": r.exec_,
              "eff": r.eff, "acting": bool(r.acting), "explored": bool(r.explored), "mode": r.mode, "risk": r.risk, "rw": r.rw}
        gt = ACTIONS[r.eff] if label == "CONFIRM" else label
        rec = mgr.learn(ev, gt, None, source="owner", full_info=True)
        r.label, r.label_source, r.labeled_at = gt, "owner", db.now(); s.commit()
    return rec


_SIM: Simulator | None = None


def _sim() -> Simulator:
    global _SIM
    if _SIM is None:
        _SIM = Simulator(Guardrail.get())
    return _SIM


@app.get("/owner/rl/status")
def rl_status(_=Depends(owner_required)):
    g = Guardrail.get()
    if g.rl is None:
        return {"available": False, "hint": "python scripts/train_dqn.py"}
    return {"available": True, **g.rl.status(), "simulator": _sim().status()}


@app.get("/owner/rl/curve")
def rl_curve(last: int = 600, smooth: int = 50, _=Depends(owner_required)):
    g = Guardrail.get()
    return g.rl.curve(min(last, 5000), max(5, min(smooth, 500))) if g.rl else {"points": []}


@app.post("/owner/rl/mode")
def rl_mode(req: RLMode, _=Depends(owner_required)):
    g = Guardrail.get()
    if g.rl is None:
        raise HTTPException(400, "RL module unavailable")
    try:
        g.rl.set_mode(req.mode)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "mode": g.rl.mode}


@app.post("/owner/rl/reset")
def rl_reset(req: RLReset, _=Depends(owner_required)):
    g = Guardrail.get()
    if _sim().s["running"]:
        raise HTTPException(409, "stop the simulation first")
    try:
        g.rl.reset(warm=req.warm)
    except (ValueError, AttributeError) as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/owner/rl/simulate")
def rl_simulate(req: RLSim, _=Depends(owner_required)):
    try:
        _sim().start(steps=max(10, min(req.steps, 20000)), rate=max(0.5, min(req.rate, 60.0)), source=req.source, seed=req.seed)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return _sim().status()


@app.post("/owner/rl/simulate/stop")
def rl_simulate_stop(_=Depends(owner_required)):
    _sim().stop(); return {"ok": True}


@app.post("/owner/rl/feedback")
def rl_feedback(req: RLFeedback, _=Depends(owner_required)):
    rec = _rl_label(req.audit_id, req.label.upper())
    if rec is None:
        raise HTTPException(404, "no unlabeled RL event for that request (RL off, hard-rule decision, or already labeled)")
    return {"ok": True, "reward_agent": rec["r_agent"], "reward_baseline": rec["r_base"], "reward_executed": rec["r_exec"]}


@app.get("/owner/keys")
def owner_keys(_=Depends(owner_required)):
    with SessionLocal() as s:
        rows = s.query(ApiClient).order_by(ApiClient.id.desc()).all()
        counts = dict(s.query(AuditLog.client_id, func.count(AuditLog.id)).group_by(AuditLog.client_id).all())
    return [{"id": c.id, "name": c.name, "key_prefix": c.key_prefix, "active": c.active, "rate_limit_per_min": c.rate_limit_per_min,
             "requests": counts.get(c.id, 0), "created_at": c.created_at.isoformat(),
             "last_used_at": c.last_used_at.isoformat() if c.last_used_at else None} for c in rows]


@app.post("/owner/keys")
def owner_key_create(req: KeyCreate, _=Depends(owner_required)):
    key = new_api_key()
    with SessionLocal() as s:
        s.add(ApiClient(name=req.name, key_prefix=key[:12], key_hash=db.hash_key(key), rate_limit_per_min=req.rate_limit_per_min))
        s.commit()
    return {"api_key": key, "note": "Store this now - it is shown only once and kept only as a hash."}


@app.delete("/owner/keys/{key_id}")
def owner_key_revoke(key_id: int, _=Depends(owner_required)):
    with SessionLocal() as s:
        c = s.get(ApiClient, key_id)
        if not c:
            raise HTTPException(404, "not found")
        c.active = False; s.commit()
    return {"ok": True}


@app.get("/owner/policy_versions")
def owner_policy_versions(_=Depends(owner_required)):
    with SessionLocal() as s:
        return [{"version": p.version, "created_at": p.created_at.isoformat(), "notes": p.notes, "metrics": p.metrics}
                for p in s.query(PolicyVersion).order_by(PolicyVersion.id.desc()).all()]


def _clean(o):
    """JSON-safe: NaN/Inf -> None (result files may contain NaN)."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


@app.get("/owner/evaluation")
def owner_evaluation(_=Depends(owner_required)):
    out = {}
    for f in sorted(Path(settings.results_dir).glob("*.json")):
        try:
            out[f.stem] = _clean(json.loads(f.read_text()))
        except Exception:
            pass
    return out


@app.get("/owner/report")
def owner_report(_=Depends(owner_required)):
    f = Path(settings.results_dir) / "final_report.md"
    return {"markdown": f.read_text(encoding="utf-8") if f.exists() else ""}


@app.post("/owner/playground")
def owner_playground(req: GuardRequest, _=Depends(owner_required)):
    res = _eval(req.prompt, _history(req.history))
    aid = _audit(res, "playground", None)
    out = res.public(); out["audit_id"] = aid
    return out
