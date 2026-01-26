# -*- coding: utf-8 -*-
"""
FastAPI Remote Control Example
Demonstrates how to use the REST API to monitor and control the trading bot
"""
import requests
import json
import time
from typing import Dict, Any


class TradingBotAPIClient:
    """Client for interacting with the Trading Bot API"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", api_key: str = None):
        """
        Initialize API client

        Args:
            base_url: Base URL of the API server
            api_key: API key for authentication (optional)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key

    def _get(self, endpoint: str) -> Dict[str, Any]:
        """Make GET request"""
        response = requests.get(f"{self.base_url}{endpoint}", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make POST request"""
        response = requests.post(f"{self.base_url}{endpoint}", json=data or {}, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        return self._get("/health")

    def get_status(self) -> Dict[str, Any]:
        """Get bot status"""
        return self._get("/status")

    def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        return self._get("/positions")

    def get_orders(self, active_only: bool = False) -> Dict[str, Any]:
        """Get orders"""
        endpoint = "/orders?active_only=true" if active_only else "/orders"
        return self._get(endpoint)

    def get_pnl(self) -> Dict[str, Any]:
        """Get P&L metrics"""
        return self._get("/pnl")

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics"""
        return self._get("/risk")

    def get_signals(self, limit: int = 50) -> Dict[str, Any]:
        """Get recent signals"""
        return self._get(f"/signals?limit={limit}")

    def pause_trading(self) -> Dict[str, Any]:
        """Pause trading"""
        return self._post("/pause")

    def resume_trading(self) -> Dict[str, Any]:
        """Resume trading"""
        return self._post("/resume")

    def stop_bot(self) -> Dict[str, Any]:
        """Stop the bot"""
        return self._post("/stop")

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        return self._post(f"/cancel_order/{order_id}")

    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close a position"""
        return self._post(f"/close_position/{symbol}")


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_json(data: Dict[str, Any]):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))


def main():
    """Main demonstration"""
    import os

    print_section("Trading Bot API Usage Example")

    # Get API key from environment (if configured)
    api_key = os.getenv("API_SECRET_KEY")

    if api_key:
        print(f"\nUsing API key authentication")
    else:
        print(f"\nNo API key configured - authentication disabled")
        print(f"To enable authentication, set API_SECRET_KEY environment variable")

    # Create API client
    client = TradingBotAPIClient(api_key=api_key)

    try:
        # 1. Health Check
        print_section("1. Health Check")
        health = client.health_check()
        print_json(health)
        time.sleep(1)

        # 2. Get Status
        print_section("2. Bot Status")
        status = client.get_status()
        print_json(status)
        time.sleep(1)

        # 3. Get Risk Metrics
        print_section("3. Risk Metrics")
        risk = client.get_risk_metrics()
        print_json(risk)
        time.sleep(1)

        # 4. Get P&L
        print_section("4. P&L Metrics")
        pnl = client.get_pnl()
        print_json(pnl)
        print(f"\nTotal P&L: Rs.{pnl.get('total_pnl', 0):,.2f}")
        print(f"Win Rate: {pnl.get('win_rate', 0):.1f}%")
        time.sleep(1)

        # 5. Get Positions
        print_section("5. Current Positions")
        positions = client.get_positions()
        if positions:
            print_json(positions)
        else:
            print("No open positions")
        time.sleep(1)

        # 6. Get Active Orders
        print_section("6. Active Orders")
        orders = client.get_orders(active_only=True)
        if orders:
            print_json(orders)
        else:
            print("No active orders")
        time.sleep(1)

        # 7. Get Recent Signals
        print_section("7. Recent Signals (Last 10)")
        signals = client.get_signals(limit=10)
        if signals:
            for signal in signals[:5]:  # Show first 5
                print(f"\nSignal #{signal.get('id')}:")
                print(f"  Type: {signal.get('signal_type')}")
                print(f"  Price: Rs.{signal.get('price', 0):,.2f}")
                print(f"  Stop Loss: Rs.{signal.get('stop_loss', 0):,.2f}")
                print(f"  Confidence: {signal.get('confidence', 0):.1f}%")
                print(f"  Time: {signal.get('timestamp')}")
        else:
            print("No signals generated yet")
        time.sleep(1)

        # 8. Control Operations (Optional - commented out for safety)
        print_section("8. Control Operations (Examples)")
        print("# Pause trading:")
        print("# client.pause_trading()")
        print("\n# Resume trading:")
        print("# client.resume_trading()")
        print("\n# Stop bot:")
        print("# client.stop_bot()")
        print("\n# Cancel order:")
        print("# client.cancel_order('ORDER123')")
        print("\n# Close position:")
        print("# client.close_position('RELIANCE')")

        # 9. Interactive API Documentation
        print_section("9. Interactive API Documentation")
        print(f"\nSwagger UI (Interactive Docs):")
        print(f"  http://127.0.0.1:8000/docs")
        print(f"\nReDoc (Alternative Docs):")
        print(f"  http://127.0.0.1:8000/redoc")

        print_section("Example Complete")
        print("\nYou can now:")
        print("  1. Visit http://127.0.0.1:8000/docs for interactive API docs")
        print("  2. Use this client library in your own scripts")
        print("  3. Monitor the bot remotely via HTTP requests")
        print("  4. Build custom dashboards or alerting systems")

    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API server.")
        print("Make sure the trading bot is running with the API enabled.")
        print("\nTo start the bot with API:")
        print("  python -m zerodha_trader.main")
        print("\nOr to disable API:")
        print("  python -m zerodha_trader.main --no-api")

    except Exception as e:
        print(f"\nERROR: {str(e)}")


if __name__ == "__main__":
    main()
