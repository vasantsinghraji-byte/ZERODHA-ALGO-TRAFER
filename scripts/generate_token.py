import argparse
from kiteconnect import KiteConnect
from config.config import settings

def generate_access_token(request_token: str):
    """Generate access token from request token"""
    try:
        kite = KiteConnect(api_key=settings.ZERODHA_API_KEY)
        data = kite.generate_session(
            request_token=request_token,
            api_secret=settings.ZERODHA_API_SECRET
        )
        
        print("\nAccess Token generated successfully!")
        print(f"Access Token: {data['access_token']}")
        print(f"\nAdd this to your .env file:")
        print(f"ZERODHA_ACCESS_TOKEN={data['access_token']}")
        
    except Exception as e:
        print(f"Error generating token: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Zerodha access token")
    parser.add_argument("--request_token", required=True, help="Request token from Zerodha login")
    args = parser.parse_args()
    
    generate_access_token(args.request_token)
