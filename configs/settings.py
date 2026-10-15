"""Central settings. Everything comes from environment / .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _b(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    app_name = os.getenv("APP_NAME", "SentinelShield AI")
    root = ROOT
    models_dir = ROOT / "models"
    datasets_dir = ROOT / "datasets"
    results_dir = ROOT / "evaluation" / "results"

    database_url = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'sentinelshield_dev.db'}")
    owner_username = os.getenv("OWNER_USERNAME", "owner")
    owner_password = os.getenv("OWNER_PASSWORD", "change_me_owner_password")
    jwt_secret = os.getenv("JWT_SECRET", "dev-secret-change-me")
    jwt_expire_minutes = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))

    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

    embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    intent_sim_threshold = float(os.getenv("INTENT_SIM_THRESHOLD", "0.50"))
    rewrite_max_tries = int(os.getenv("REWRITE_MAX_TRIES", "3"))
    rewriter_backend = os.getenv("REWRITER_BACKEND", "auto")
    log_raw_prompts = _b("LOG_RAW_PROMPTS", "false")
    log_retention_days = int(os.getenv("LOG_RETENTION_DAYS", "30"))
    policy_version = os.getenv("POLICY_VERSION", "1.0.0")
    force_q_policy = _b("FORCE_Q_POLICY", "false")    # use the Q-policy even if validation did not recommend it

    # live deep-RL controller (policy/q_learning/live.py): off | shadow | active | auto
    rl_mode = os.getenv("RL_MODE", "shadow")
    rl_explore_eps = float(os.getenv("RL_EXPLORE_EPS", "0.0"))     # live exploration (safe side of the baseline only); 0 = none

    eval_judge = os.getenv("EVAL_JUDGE", "ollama")
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_judge_model = os.getenv("OLLAMA_JUDGE_MODEL", "llama3.2:3b")
    ollama_guard_model = os.getenv("OLLAMA_GUARD_MODEL", "llama-guard3:1b")


settings = Settings()
