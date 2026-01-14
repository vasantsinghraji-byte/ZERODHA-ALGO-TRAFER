# -*- coding: utf-8 -*-
"""
API Security - Authentication and Authorization
"""
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Define the security scheme
# The API key should be passed in the "X-API-Key" header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header), required_key: str = None):
    """
    Verify API key from request header

    Args:
        api_key: API key from request header
        required_key: The expected API key

    Raises:
        HTTPException: If API key is missing or invalid

    Returns:
        The validated API key
    """
    # SECURITY: Fail-closed - reject all requests if API key not configured
    if required_key is None:
        logger.error("Security misconfiguration: No API_SECRET_KEY set. Rejecting request.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server security misconfiguration",
        )

    # Check if API key is provided
    if not api_key:
        logger.warning("API request rejected - missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Please provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Verify API key
    if api_key != required_key:
        logger.warning(f"API request rejected - invalid API key: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )

    return api_key


def create_api_key_dependency(settings):
    """
    Create an API key dependency with the configured secret key

    Args:
        settings: Application settings

    Returns:
        Dependency function for FastAPI
    """
    def dependency(api_key: Optional[str] = Security(api_key_header)):
        return verify_api_key(api_key, required_key=settings.api_secret_key)

    return dependency
