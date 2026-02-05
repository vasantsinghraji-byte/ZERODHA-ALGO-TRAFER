import os
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional
import yaml

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class Settings(BaseSettings):
    """
    Application settings with security-conscious defaults.

    SECURITY FIX: Database credentials are NOT hardcoded.
    All sensitive values must be provided via environment variables or .env file.

    Required environment variables for production:
    - DATABASE_URL: PostgreSQL connection string
    - REDIS_URL: Redis connection string
    - ZERODHA_API_KEY, ZERODHA_API_SECRET, ZERODHA_ACCESS_TOKEN
    """
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    # Environment
    ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")

    # Database - SECURITY FIX: No hardcoded credentials
    # Defaults to None; required in production via validator
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection string. Required in production."
    )
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis connection string. Required in production."
    )

    # Zerodha API
    ZERODHA_API_KEY: Optional[str] = Field(default=None)
    ZERODHA_API_SECRET: Optional[str] = Field(default=None)
    ZERODHA_ACCESS_TOKEN: Optional[str] = Field(default=None)

    # Feature flags
    ENABLE_PAPER_TRADING: bool = True
    ENABLE_LIVE_TRADING: bool = False

    # Trading parameters
    # PRECISION FIX (Bug #12): Use Decimal for financial values to avoid
    # floating-point precision errors (e.g., 0.1 + 0.2 != 0.3)
    # Decimal provides exact decimal arithmetic required for money calculations
    MAX_POSITION_SIZE: Decimal = Decimal("100000.00")
    RISK_PER_TRADE: Decimal = Decimal("0.01")

    @field_validator('DATABASE_URL', 'REDIS_URL')
    @classmethod
    def validate_database_credentials(cls, v: Optional[str], info) -> Optional[str]:
        """
        SECURITY FIX: Validate database credentials are provided in production.

        In development, None is allowed (uses SQLite or mock).
        In production, credentials MUST be provided via environment variables.
        """
        field_name = info.field_name
        env_value = info.data.get('ENV', 'development')

        if v is None and env_value == 'production':
            raise ValueError(
                f"{field_name} must be set in production environment. "
                f"Add {field_name}=... to your .env file or environment variables."
            )

        # Warn if credentials look like they contain defaults (shouldn't happen now)
        if v and 'trader123' in v:
            import warnings
            warnings.warn(
                f"SECURITY WARNING: {field_name} appears to contain default credentials. "
                "Please use unique, secure credentials in production.",
                SecurityWarning,
                stacklevel=2
            )

        return v

    @field_validator('ZERODHA_API_KEY', 'ZERODHA_API_SECRET', 'ZERODHA_ACCESS_TOKEN')
    @classmethod
    def validate_zerodha_credentials(cls, v: Optional[str], info) -> Optional[str]:
        field_name = info.field_name
        # Use Pydantic's validation context, not os.getenv which may not reflect .env file values
        env_value = info.data.get('ENV', 'development')
        if v is None and env_value == 'production':
            raise ValueError(f'{field_name} must be set in production environment')
        return v

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "Settings":
        """Load settings from YAML file"""
        with open(yaml_path) as f:
            config_data: Dict = yaml.safe_load(f)
        return cls(**config_data)

settings = Settings()
