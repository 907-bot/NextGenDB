"""
RBAC / ABAC Security Module — authentication, authorization, and schema enforcement.

Features:
  • JWT-based authentication (HS256)
  • Role-Based Access Control (RBAC) + Attribute-Based extensions
  • Schema enforcement + safe migration registry
  • Multi-tenant namespace isolation
"""
import hashlib
import hmac
import json
import logging
import re
import time
import uuid
from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("nextgendb.security")

# ── JWT (no external dependency) ──────────────────────────────────────────────

def _b64_encode(data: bytes) -> str:
    return urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return urlsafe_b64decode(s + "=" * padding)


class JWTManager:
    ALGORITHM = "HS256"

    def __init__(self, secret: str):
        self._secret = secret.encode()

    def _sign(self, msg: str) -> str:
        sig = hmac.new(self._secret, msg.encode(), hashlib.sha256).digest()
        return _b64_encode(sig)

    def encode(self, payload: Dict, expires_in: int = 3600) -> str:
        header  = _b64_encode(json.dumps({"alg": self.ALGORITHM, "typ": "JWT"}).encode())
        payload = dict(payload, exp=int(time.time()) + expires_in, iat=int(time.time()), jti=str(uuid.uuid4()))
        body    = _b64_encode(json.dumps(payload).encode())
        sig     = self._sign(f"{header}.{body}")
        return f"{header}.{body}.{sig}"

    def decode(self, token: str) -> Dict:
        try:
            header, body, sig = token.split(".")
        except ValueError:
            raise ValueError("Malformed JWT")
        expected = self._sign(f"{header}.{body}")
        if not hmac.compare_digest(expected, sig):
            raise ValueError("Invalid JWT signature")
        payload = json.loads(_b64_decode(body))
        if payload.get("exp", 0) < time.time():
            raise ValueError("JWT expired")
        return payload


# ── RBAC ──────────────────────────────────────────────────────────────────────

class Permission(str, Enum):
    READ        = "read"
    WRITE       = "write"
    DELETE      = "delete"
    ADMIN       = "admin"
    SCHEMA_EDIT = "schema_edit"
    METRICS     = "metrics"


ROLE_PERMISSIONS: Dict[str, Set[Permission]] = {
    "admin":     set(Permission),
    "developer": {Permission.READ, Permission.WRITE, Permission.METRICS},
    "analyst":   {Permission.READ, Permission.METRICS},
    "readonly":  {Permission.READ},
    "ingestor":  {Permission.READ, Permission.WRITE},
}


@dataclass
class User:
    user_id:    str
    username:   str
    role:       str
    tenant_id:  str = "default"
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def permissions(self) -> Set[Permission]:
        return ROLE_PERMISSIONS.get(self.role, set())

    def can(self, perm: Permission) -> bool:
        return perm in self.permissions


class AuthManager:
    """Simple in-memory user store + JWT issuance."""

    def __init__(self, jwt_secret: str = "nextgendb-dev-secret-change-in-prod"):
        self._jwt    = JWTManager(jwt_secret)
        self._users: Dict[str, Dict] = {}   # username → {password_hash, role, tenant_id}
        # Seed default admin
        self.create_user("admin", "admin", "admin", tenant_id="default")

    def _hash_pw(self, pw: str) -> str:
        return hashlib.sha256(pw.encode()).hexdigest()

    def create_user(self, username: str, password: str, role: str, tenant_id: str = "default") -> str:
        if username in self._users:
            raise ValueError(f"User '{username}' already exists")
        user_id = str(uuid.uuid4())
        self._users[username] = {
            "user_id": user_id, "password_hash": self._hash_pw(password),
            "role": role, "tenant_id": tenant_id,
        }
        logger.info("User created: %s (role=%s, tenant=%s)", username, role, tenant_id)
        return user_id

    def login(self, username: str, password: str) -> str:
        """Return a JWT token on successful authentication."""
        user = self._users.get(username)
        if not user or user["password_hash"] != self._hash_pw(password):
            raise ValueError("Invalid credentials")
        token = self._jwt.encode({
            "sub": username,
            "uid": user["user_id"],
            "role": user["role"],
            "tenant": user["tenant_id"],
        })
        logger.info("User '%s' authenticated", username)
        return token

    def validate(self, token: str) -> User:
        payload = self._jwt.decode(token)
        return User(
            user_id=payload["uid"],
            username=payload["sub"],
            role=payload["role"],
            tenant_id=payload.get("tenant", "default"),
        )

    def require(self, token: str, perm: Permission) -> User:
        user = self.validate(token)
        if not user.can(perm):
            raise PermissionError(f"User '{user.username}' lacks permission: {perm}")
        return user


# ── Schema Enforcement ────────────────────────────────────────────────────────

@dataclass
class NodeSchema:
    label:    str
    required: List[str] = field(default_factory=list)
    optional: Dict[str, type] = field(default_factory=dict)

    def validate(self, properties: Dict[str, Any]) -> List[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        for req in self.required:
            if req not in properties:
                errors.append(f"Missing required property: '{req}'")
        return errors


@dataclass
class Migration:
    version:     int
    description: str
    up:          str   # Cypher/SQL to run
    down:        str   # Rollback


class SchemaRegistry:
    """Registry of node/edge schemas and migration history."""

    def __init__(self):
        self._schemas: Dict[str, NodeSchema] = {}
        self._migrations: List[Migration] = []
        self._applied: List[int] = []

    def register_schema(self, schema: NodeSchema):
        self._schemas[schema.label] = schema
        logger.info("Schema registered: %s", schema.label)

    def validate_node(self, label: str, properties: Dict[str, Any]) -> List[str]:
        schema = self._schemas.get(label)
        if schema is None:
            return []    # permissive if no schema defined
        return schema.validate(properties)

    def register_migration(self, migration: Migration):
        self._migrations.append(migration)

    def pending_migrations(self) -> List[Migration]:
        return [m for m in self._migrations if m.version not in self._applied]

    def mark_applied(self, version: int):
        self._applied.append(version)
        logger.info("Migration v%d marked applied", version)
