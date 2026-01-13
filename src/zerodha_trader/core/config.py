        logger.info("="*70)


# ============================================================================
# SINGLETON PATTERN
# ============================================================================

_settings: Optional[Settings] = None


def get_settings(reload: bool = False, _env_file=None) -> Settings:
    """
    Get settings singleton

    Args:
        reload: Force reload settings from environment
        _env_file: Override env file path (for testing)

    Returns:
        Settings instance
    """
    global _settings

    if _settings is None or reload:
        # Pass _env_file if provided (for testing)
        if _env_file is not None or os.getenv('TESTING'):
            _settings = Settings(_env_file=_env_file)
        else:
            _settings = Settings()

        # Log configuration on first load
        if reload or _settings is not None:
            _settings.log_config()

    return _settings


def get_kite_client():
    """
    Get configured Kite client

    Returns:
        KiteConnect instance configured with API credentials
    """
    from kiteconnect import KiteConnect
    settings = get_settings()

    # Safety check for live trading
    if settings.live_trading and not settings.is_production:
        raise RuntimeError(