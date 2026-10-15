"""Auth: client API keys (hashed in MySQL) and owner JWT login for the dashboard."""
from __future__ import annotations
import collections, secrets, time
import jwt
from fastapi import Header, HTTPException
from api.db import SessionLocal, ApiClient, OwnerUser, hash_key, verify_pw, now
from configs.settings import settings

_WINDOWS: dict[int, collections.deque] = collections.defaultdict(collections.deque)


def new_api_key() -> str:
    return "sg_live_" + secrets.token_urlsafe(32)


def issue_owner_token(username: str) -> str:
    return jwt.encode({"sub": username, "role": "owner", "exp": int(time.time()) + settings.jwt_expire_minutes * 60},
                      settings.jwt_secret, algorithm="HS256")


def login_owner(username: str, password: str) -> str | None:
    with SessionLocal() as s:
        u = s.query(OwnerUser).filter_by(username=username).first()
        if u and verify_pw(password, u.pw_salt, u.pw_hash):
            return issue_owner_token(username)
    return None


def _bearer(authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def owner_required(authorization: str | None = Header(default=None)):
    tok = _bearer(authorization, None)
    try:
        data = jwt.decode(tok or "", settings.jwt_secret, algorithms=["HS256"])
        if data.get("role") != "owner":
            raise ValueError
        return data["sub"]
    except Exception:
        raise HTTPException(401, "owner authentication required")


def client_required(authorization: str | None = Header(default=None), x_api_key: str | None = Header(default=None)) -> ApiClient:
    tok = _bearer(authorization, x_api_key)
    if not tok or not tok.startswith("sg_"):
        raise HTTPException(401, "missing or invalid API key")
    with SessionLocal() as s:
        c = s.query(ApiClient).filter_by(key_hash=hash_key(tok), active=True).first()
        if not c:
            raise HTTPException(401, "invalid or revoked API key")
        c.last_used_at = now(); s.commit()
        limit, cid = c.rate_limit_per_min, c.id
        s.expunge(c)
    w = _WINDOWS[cid]; t = time.time()
    while w and w[0] < t - 60:
        w.popleft()
    if len(w) >= limit:
        raise HTTPException(429, "rate limit exceeded")
    w.append(t)
    return c


def client_or_owner(authorization: str | None = Header(default=None), x_api_key: str | None = Header(default=None)):
    tok = _bearer(authorization, x_api_key)
    if tok and tok.startswith("sg_"):
        return ("client", client_required(authorization, x_api_key))
    return ("owner", owner_required(authorization))
