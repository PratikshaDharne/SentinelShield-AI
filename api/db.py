"""MySQL (SQLAlchemy) models: owner user, client API keys, audit log, false-positive review queue, policy versions."""
from __future__ import annotations
import datetime as dt, hashlib, hmac, os, secrets
from sqlalchemy import (create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey, Index)
from sqlalchemy.orm import declarative_base, sessionmaker
from configs.settings import settings

Base = declarative_base()
_kw = {"pool_pre_ping": True, "pool_recycle": 1800}
if settings.database_url.startswith("sqlite"):
    _kw = {"connect_args": {"check_same_thread": False}}
engine = create_engine(settings.database_url, **_kw)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def now() -> dt.datetime:
    return dt.datetime.utcnow()


class OwnerUser(Base):
    __tablename__ = "owner_users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    pw_salt = Column(String(64), nullable=False)
    pw_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=now)


class ApiClient(Base):
    __tablename__ = "api_clients"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    key_prefix = Column(String(16), nullable=False)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    rate_limit_per_min = Column(Integer, default=120)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=now)
    last_used_at = Column(DateTime)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=now, index=True)
    request_id = Column(String(40), index=True)
    client_id = Column(Integer, ForeignKey("api_clients.id"), nullable=True, index=True)
    endpoint = Column(String(40))
    prompt_hash = Column(String(64))
    prompt_len = Column(Integer)
    decision = Column(String(12), index=True)
    category = Column(String(40), index=True)
    risk_score = Column(Float)
    confidence = Column(Float)
    reason_codes = Column(JSON)
    flags = Column(JSON)
    policy_version = Column(String(20))
    latency_ms = Column(Float)
    stage_latency = Column(JSON)
    rewrite_status = Column(String(12))
    rewrite_similarity = Column(Float)
    rewrite_attempts = Column(Integer)
    llm_provider = Column(String(12))      # gemini | groq | none
    llm_latency_ms = Column(Float)
    preview = Column(Text)                 # NULL unless LOG_RAW_PROMPTS=true (redacted, truncated)
    __table_args__ = (Index("ix_audit_ts_decision", "ts", "decision"),)


class ReviewItem(Base):
    __tablename__ = "review_queue"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=now, index=True)
    audit_id = Column(Integer, ForeignKey("audit_logs.id"))
    decision = Column(String(12))
    category = Column(String(40))
    risk_score = Column(Float)
    preview = Column(Text)                 # redacted, <=160 chars, purged by retention job
    status = Column(String(20), default="pending", index=True)    # pending | confirmed | false_positive
    note = Column(Text)
    resolved_at = Column(DateTime)


class RLEvent(Base):
    """One live-RL decision (features + Q-values, never the prompt text) so an owner label can become a reward later."""
    __tablename__ = "rl_events"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=now, index=True)
    audit_id = Column(Integer, index=True)
    mode = Column(String(8)); acting = Column(Boolean, default=False); explored = Column(Boolean, default=False)
    base = Column(Integer); agent = Column(Integer); exec_ = Column("exec", Integer); eff = Column(Integer)
    risk = Column(Float); q = Column(JSON); x = Column(JSON); legal = Column(JSON); rw = Column(JSON)
    label = Column(String(12)); label_source = Column(String(12)); labeled_at = Column(DateTime)


class PolicyVersion(Base):
    __tablename__ = "policy_versions"
    id = Column(Integer, primary_key=True)
    version = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=now)
    notes = Column(Text)
    metrics = Column(JSON)


def hash_pw(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    return salt, hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def verify_pw(password: str, salt: str, h: str) -> bool:
    return hmac.compare_digest(hash_pw(password, salt)[1], h)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def init_db(retries: int = 30, delay: float = 2.0):
    import time
    for i in range(retries):                      # wait for MySQL to accept connections (docker start-up order)
        try:
            Base.metadata.create_all(engine); break
        except Exception:
            if i == retries - 1: raise
            time.sleep(delay)
    with SessionLocal() as s:
        if not s.query(OwnerUser).filter_by(username=settings.owner_username).first():
            salt, h = hash_pw(settings.owner_password)
            s.add(OwnerUser(username=settings.owner_username, pw_salt=salt, pw_hash=h))
        if not s.query(PolicyVersion).filter_by(version=settings.policy_version).first():
            s.add(PolicyVersion(version=settings.policy_version, notes="Initial hybrid policy (rules + classifier + retrieval)", metrics={}))
        s.commit()


def purge_old(days: int | None = None):
    cutoff = now() - dt.timedelta(days=days or settings.log_retention_days)
    with SessionLocal() as s:
        s.query(ReviewItem).filter(ReviewItem.ts < cutoff).update({ReviewItem.preview: None})
        s.query(AuditLog).filter(AuditLog.ts < cutoff).delete()
        s.commit()
