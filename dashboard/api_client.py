"""Thin HTTP client for the SentinelShield AI owner API."""
from __future__ import annotations
import os
import requests

API_URL = os.getenv("API_URL", f"http://localhost:{os.getenv('API_PORT', '8130')}").rstrip("/")


class ApiError(Exception):
    pass


def _h(token): return {"Authorization": f"Bearer {token}"} if token else {}


def _req(method, path, token=None, timeout=20, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", headers=_h(token), timeout=timeout, **kw)
    except requests.RequestException as e:
        raise ApiError(f"API unreachable at {API_URL}: {e.__class__.__name__}")
    if r.status_code == 401:
        raise ApiError("401")
    if r.status_code >= 400:
        raise ApiError(f"{r.status_code}: {r.text[:200]}")
    return r.json()


def login(user, pw): return _req("POST", "/owner/login", json={"username": user, "password": pw})["access_token"]
def health(): return _req("GET", "/v1/health", timeout=8)
def overview(t, minutes=60): return _req("GET", "/owner/overview", t, params={"minutes": minutes})
def audit(t, limit=200): return _req("GET", "/owner/audit", t, params={"limit": limit})
def review(t, status="pending"): return _req("GET", "/owner/review", t, params={"status": status})
def review_update(t, i, status, note=""): return _req("POST", f"/owner/review/{i}", t, json={"status": status, "note": note})
def keys(t): return _req("GET", "/owner/keys", t)
def key_create(t, name, rate): return _req("POST", "/owner/keys", t, json={"name": name, "rate_limit_per_min": rate})
def key_revoke(t, i): return _req("DELETE", f"/owner/keys/{i}", t)
def policy_versions(t): return _req("GET", "/owner/policy_versions", t)
def evaluation(t): return _req("GET", "/owner/evaluation", t, timeout=60)
def report(t): return _req("GET", "/owner/report", t)["markdown"]
def playground(t, prompt, history=None): return _req("POST", "/owner/playground", t, json={"prompt": prompt, "history": history or []}, timeout=60)


# ---- live deep-RL ----
def rl_status(t): return _req("GET", "/owner/rl/status", t)
def rl_curve(t, last=600, smooth=50): return _req("GET", "/owner/rl/curve", t, params={"last": last, "smooth": smooth})
def rl_mode(t, mode): return _req("POST", "/owner/rl/mode", t, json={"mode": mode})
def rl_reset(t, warm=True): return _req("POST", "/owner/rl/reset", t, json={"warm": warm})
def rl_simulate(t, steps, rate, source, seed=0): return _req("POST", "/owner/rl/simulate", t, json={"steps": steps, "rate": rate, "source": source, "seed": seed})
def rl_simulate_stop(t): return _req("POST", "/owner/rl/simulate/stop", t)
def rl_feedback(t, audit_id, label): return _req("POST", "/owner/rl/feedback", t, json={"audit_id": audit_id, "label": label})
