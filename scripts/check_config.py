"""
Script to validate configuration before running the application
"""
import sys
from config.config import settings

def check_config():
    """Validate all required configuration settings"""
    errors = []
    
    # Check Zerodha credentials
    if not settings.ZERODHA_API_KEY:
        errors.append("ZERODHA_API_KEY is not set")
    if not settings.ZERODHA_API_SECRET:
        errors.append("ZERODHA_API_SECRET is not set")
    if not settings.ZERODHA_ACCESS_TOKEN:
        errors.append("ZERODHA_ACCESS_TOKEN is not set")
    
    # Check database configuration
    if not settings.DATABASE_URL:
        errors.append("DATABASE_URL is not set")
    if not settings.REDIS_URL:
        errors.append("REDIS_URL is not set")
    
    # Print results
    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease check your .env file or environment variables")
        return False
    else:
        print("✅ Configuration is valid")
        print(f"  - Environment: {settings.ENV}")
        print(f"  - Log Level: {settings.LOG_LEVEL}")
        print(f"  - Paper Trading: {settings.ENABLE_PAPER_TRADING}")
        print(f"  - Live Trading: {settings.ENABLE_LIVE_TRADING}")
        return True

if __name__ == "__main__":
    if not check_config():
        sys.exit(1)
