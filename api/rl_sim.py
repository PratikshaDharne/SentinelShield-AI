"""Labelled-traffic simulator for the live RL loop.

Streams prompts from a labelled split (real public datasets + synthetic) through the REAL pipeline (normalizer -> classifier ->
retrieval -> policy -> RL layer -> rewrite agent). Every decision is then scored with the PDF reward function using the ground-truth
label, which gives the online DQN a genuine reward signal in real time. It is a simulation: it writes nothing to the audit log and
never touches the isolated final_test / real_test sets. For real users, the same learning path is fed by owner feedback instead."""
from __future__ import annotations
import random, threading, time
from evaluation.features import Rec, load_rows, rewrite_outcome

SOURCES = {"val": "validation split (real + synthetic)", "val_unseen": "validation, unseen phrasing", "unseen_test_drift": "val, then DRIFT to unseen attack families"}


class Simulator:
    def __init__(self, guard):
        self.g = guard; self.thread: threading.Thread | None = None; self.stop_flag = threading.Event()
        self.s = {"running": False, "done": 0, "total": 0, "rate": 0.0, "source": None, "started": None, "finished": None, "error": None, "drifted": False}

    def status(self) -> dict:
        d = dict(self.s)
        if d["started"] and d["running"]: d["actual_rate"] = round(d["done"] / max(time.time() - d["started"], 1e-6), 2)
        return d

    def start(self, steps: int = 600, rate: float = 8.0, source: str = "val", seed: int = 0, p_risky: float = 0.45):
        if self.s["running"]: raise RuntimeError("simulation already running")
        if self.g.rl is None or self.g.rl.agent is None or self.g.rl.mode == "off":
            raise RuntimeError("RL is off or no agent is trained (python scripts/train_dqn.py; set mode to shadow/active/auto)")
        if source == "unseen_test_drift":
            first, second = load_rows("val"), load_rows("unseen_test")
        else:
            first, second = load_rows(source if source in ("val", "val_unseen") else "val"), []
        if not first: raise RuntimeError(f"no rows for {source}; run scripts/build_dataset.py")
        self.stop_flag.clear()
        self.s.update({"running": True, "done": 0, "total": int(steps), "rate": float(rate), "source": source, "started": time.time(), "finished": None, "error": None, "drifted": False})
        self.thread = threading.Thread(target=self._run, args=(first, second, int(steps), float(rate), seed, p_risky), daemon=True); self.thread.start()

    def stop(self): self.stop_flag.set()

    def _draw(self, rng, pool_risky, pool_benign, p_risky):
        pool = pool_risky if (rng.random() < p_risky and pool_risky) else pool_benign
        return rng.choice(pool if pool else (pool_risky or pool_benign))

    def _run(self, first, second, steps, rate, seed, p_risky):
        rng = random.Random(seed); mgr = self.g.rl; pools = None
        try:
            def mk(rows): return [r for r in rows if r["decision"] != "ALLOW"], [r for r in rows if r["decision"] == "ALLOW"]
            pools = mk(first); done = 0; sid = 0
            while done < steps and not self.stop_flag.is_set():
                if second and done >= steps // 2 and not self.s["drifted"]:
                    pools = mk(second); self.s["drifted"] = True
                sid += 1; key = f"sim-{seed}-{sid}"; hist: list[str] = []; prev = None
                for _ in range(rng.randint(1, 4)):
                    if done >= steps or self.stop_flag.is_set(): break
                    t0 = time.perf_counter(); row = self._draw(rng, *pools, p_risky)
                    o: dict = {}
                    self.g.evaluate(row["text"], hist[-6:], session_key=key, rl_out=o)
                    ev = o.get("event"); hist.append(row["text"])
                    if ev is not None:
                        rw = rewrite_outcome(self.g, Rec(row, o["an"]))
                        if prev is not None:
                            mgr.learn(prev[0], prev[1], prev[2], source="sim", x2=ev["x"], legal2=ev["legal"])
                        prev = (ev, row["decision"], rw)
                    done += 1; self.s["done"] = done
                    time.sleep(max(0.0, 1.0 / max(rate, 0.1) - (time.perf_counter() - t0)))
                if prev is not None:
                    mgr.learn(prev[0], prev[1], prev[2], source="sim")
            mgr.save()
        except Exception as e:                       # surfaced in the dashboard, never crashes the API
            self.s["error"] = f"{type(e).__name__}: {e}"
        finally:
            self.s.update({"running": False, "finished": time.time()})
