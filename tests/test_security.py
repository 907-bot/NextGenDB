"""Tests for Security Module (Auth + RBAC)"""
import pytest

from backend.security.auth import (
    JWTManager, AuthManager, Permission, User, SchemaRegistry, NodeSchema
)


class TestJWTManager:
    """JWT encoding and decoding"""

    def test_jwt_encode_decode(self):
        """Should encode and decode JWTs"""
        jwt = JWTManager("secret-key")
        
        payload = {"sub": "user123", "role": "admin"}
        token = jwt.encode(payload, expires_in=3600)
        
        decoded = jwt.decode(token)
        assert decoded["sub"] == "user123"
        assert decoded["role"] == "admin"

    def test_jwt_signature_validation(self):
        """Should reject tampered tokens"""
        jwt1 = JWTManager("secret-1")
        jwt2 = JWTManager("secret-2")
        
        token = jwt1.encode({"sub": "user1"}, expires_in=3600)
        
        with pytest.raises(ValueError, match="Invalid JWT signature"):
            jwt2.decode(token)

    def test_jwt_expiration(self):
        """Should reject expired tokens"""
        jwt = JWTManager("secret")
        
        token = jwt.encode({"sub": "user1"}, expires_in=0)  # Expires immediately
        
        import time
        time.sleep(0.1)
        
        with pytest.raises(ValueError, match="JWT expired"):
            jwt.decode(token)


class TestAuthManager:
    """User authentication and session management"""

    def test_user_creation_and_login(self):
        """Should create users and authenticate"""
        auth = AuthManager(jwt_secret="test-secret")
        
        user_id = auth.create_user("alice", "password123", "developer")
        assert user_id is not None
        
        token = auth.login("alice", "password123")
        assert token is not None

    def test_invalid_credentials(self):
        """Should reject invalid credentials"""
        auth = AuthManager(jwt_secret="test-secret")
        auth.create_user("alice", "correct-password", "user")
        
        with pytest.raises(ValueError, match="Invalid credentials"):
            auth.login("alice", "wrong-password")

    def test_token_validation(self):
        """Should validate user from token"""
        auth = AuthManager(jwt_secret="test-secret")
        auth.create_user("alice", "password", "developer")
        
        token = auth.login("alice", "password")
        user = auth.validate(token)
        
        assert user.username == "alice"
        assert user.role == "developer"

    def test_permission_checking(self):
        """Should enforce role-based permissions"""
        auth = AuthManager(jwt_secret="test-secret")
        auth.create_user("alice", "password", "analyst")
        auth.create_user("bob", "password", "admin")
        
        alice_token = auth.login("alice", "password")
        bob_token = auth.login("bob", "password")
        
        # Analyst can't delete
        with pytest.raises(PermissionError):
            auth.require(alice_token, Permission.DELETE)
        
        # Admin can do everything
        user = auth.require(bob_token, Permission.DELETE)
        assert user.role == "admin"

    def test_multi_tenant_isolation(self):
        """Should isolate users by tenant"""
        auth = AuthManager(jwt_secret="test-secret")
        
        user1 = auth.create_user("user1", "pass", "user", tenant_id="tenant_a")
        user2 = auth.create_user("user2", "pass", "user", tenant_id="tenant_b")
        
        token1 = auth.login("user1", "pass")
        u1 = auth.validate(token1)
        
        assert u1.tenant_id == "tenant_a"


class TestSchemaRegistry:
    """Schema validation and migrations"""

    def test_schema_registration(self):
        """Should register and validate schemas"""
        registry = SchemaRegistry()
        
        schema = NodeSchema(
            label="User",
            required=["username", "email"],
            optional={"age": int, "bio": str}
        )
        registry.register_schema(schema)
        
        errors = registry.validate_node("User", {"username": "alice", "email": "alice@example.com"})
        assert len(errors) == 0

    def test_schema_validation_errors(self):
        """Should detect missing required fields"""
        registry = SchemaRegistry()
        
        schema = NodeSchema(
            label="User",
            required=["username", "email"]
        )
        registry.register_schema(schema)
        
        errors = registry.validate_node("User", {"username": "alice"})
        assert len(errors) == 1
        assert "email" in errors[0]

    def test_migration_tracking(self):
        """Should track applied migrations"""
        from backend.security.auth import Migration
        
        registry = SchemaRegistry()
        
        m1 = Migration(1, "Add username index", "up_v1", "down_v1")
        m2 = Migration(2, "Add email field", "up_v2", "down_v2")
        
        registry.register_migration(m1)
        registry.register_migration(m2)
        
        # Initially both are pending
        assert len(registry.pending_migrations()) == 2
        
        # Mark v1 as applied
        registry.mark_applied(1)
        assert len(registry.pending_migrations()) == 1
