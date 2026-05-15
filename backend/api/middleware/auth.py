"""API key authentication middleware."""

import os
import hashlib
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In production: load from database. Here: load from environment.
VALID_API_KEYS: set[str] = set(
    filter(None, os.environ.get("VALID_API_KEYS", "test_key_dev_only").split(","))
)


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key. Visit scandiumlabs.ai/access to get one.",
        )
    return api_key