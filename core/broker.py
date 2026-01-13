"""
Zerodha Broker Integration
Makes talking to Zerodha super easy!

Think of this as a translator between you and Zerodha.
You say "buy RELIANCE" and this handles all the complex stuff.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import kiteconnect
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    logger.warning("kiteconnect not installed. Run: pip install kiteconnect")


class OrderType(Enum):
    """Types of orders - like different ways to shop"""
    MARKET = "MARKET"      # Buy/Sell at current price (instant)
    LIMIT = "LIMIT"        # Buy/Sell only at your price (may wait)
    SL = "SL"              # Stop-Loss order (safety net)
    SL_M = "SL-M"          # Stop-Loss Market order


class TransactionType(Enum):
    """Buy or Sell"""
    BUY = "BUY"
    SELL = "SELL"


class ProductType(Enum):
    """How long you want to hold"""
    INTRADAY = "MIS"       # Same day (must sell before 3:20 PM)
    DELIVERY = "CNC"       # Keep forever (like buying a book)
    MARGIN = "NRML"        # Futures & Options


@dataclass
class Quote:
    """Current price information for a stock"""
    symbol: str
    last_price: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime


@dataclass
class Order:
    """An order you placed"""
    order_id: str
    symbol: str
    transaction_type: str
    quantity: int
    price: float
    status: str
    filled_quantity: int = 0
    average_price: float = 0.0
    message: str = ""


@dataclass
class Position:
    """A stock you currently own"""
    symbol: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    pnl_percent: float


class ZerodhaBroker:
    """
    Your connection to Zerodha

    This class handles all communication with Zerodha's servers.
    It's like having a personal assistant at the stock exchange!

    Example:
        broker = ZerodhaBroker(api_key="xxx", api_secret="yyy")
        broker.login()
        broker.buy("RELIANCE", quantity=10)
    """

    def __init__(self, api_key: str = "", api_secret: str = ""):
        """
        Create a new broker connection.

        Args:
            api_key: Your Zerodha API key (from Kite Connect)
            api_secret: Your Zerodha API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token: Optional[str] = None
        self.kite: Optional[Any] = None
        self.ticker: Optional[Any] = None
        self.connected = False

        # Callbacks for real-time updates
        self._tick_callbacks: List[callable] = []
        self._order_callbacks: List[callable] = []

        # Initialize KiteConnect if available
        if KITE_AVAILABLE and api_key:
            self.kite = KiteConnect(api_key=api_key)

    @property
    def is_connected(self) -> bool:
        """Check if we're connected to Zerodha"""
        return self.connected and self.kite is not None

    def get_login_url(self) -> str:
        """
        Get the URL to login to Zerodha.

        Returns:
            URL to open in browser for login
        """
        if not self.kite:
            return ""
        return self.kite.login_url()

    def set_access_token(self, request_token: str) -> bool:
        """
        Complete the login process with the token from Zerodha.

        Args:
            request_token: The token you get after logging in

        Returns:
            True if successful, False otherwise
        """
        if not self.kite:
            logger.error("KiteConnect not initialized")
            return False

        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            self.connected = True
            logger.info("Successfully connected to Zerodha!")
            return True
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    def login_with_token(self, access_token: str) -> bool:
        """
        Login using a saved access token.

        Args:
            access_token: Previously saved access token

        Returns:
            True if successful
        """
        if not self.kite:
            return False

        try:
            self.kite.set_access_token(access_token)
            self.access_token = access_token
            # Verify by getting profile
            self.kite.profile()
            self.connected = True
            logger.info("Connected with saved token!")
            return True
        except Exception as e:
            logger.error(f"Token login failed: {e}")
            self.connected = False
            return False

    def get_profile(self) -> Optional[Dict[str, Any]]:
        """Get your Zerodha account details"""
        if not self.is_connected:
            return None
        try:
            return self.kite.profile()
        except Exception as e:
            logger.error(f"Failed to get profile: {e}")
            return None

    def get_balance(self) -> float:
        """
        Get your available balance.

        Returns:
            Available cash balance
        """
        if not self.is_connected:
            return 0.0
        try:
            margins = self.kite.margins()
            return margins.get('equity', {}).get('available', {}).get('cash', 0.0)
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Optional[Quote]:
        """
        Get current price of a stock.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            exchange: Exchange (NSE or BSE)

        Returns:
            Quote object with price details
        """
        if not self.is_connected:
            return None

        try:
            instrument = f"{exchange}:{symbol}"
            data = self.kite.quote([instrument])

            if instrument in data:
                q = data[instrument]
                return Quote(
                    symbol=symbol,
                    last_price=q.get('last_price', 0),
                    open=q.get('ohlc', {}).get('open', 0),
                    high=q.get('ohlc', {}).get('high', 0),
                    low=q.get('ohlc', {}).get('low', 0),
                    close=q.get('ohlc', {}).get('close', 0),
                    volume=q.get('volume', 0),
                    timestamp=datetime.now()
                )
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")

        return None

    def buy(self, symbol: str, quantity: int, price: float = 0,
            order_type: OrderType = OrderType.MARKET,
            product: ProductType = ProductType.INTRADAY,
            exchange: str = "NSE") -> Optional[str]:
        """
        Buy a stock! 🛒

        Args:
            symbol: Stock to buy (e.g., "RELIANCE")
            quantity: How many shares
            price: Price (only for LIMIT orders)
            order_type: MARKET or LIMIT
            product: INTRADAY or DELIVERY
            exchange: NSE or BSE

        Returns:
            Order ID if successful, None otherwise
        """
        return self._place_order(
            symbol=symbol,
            quantity=quantity,
            price=price,
            transaction_type=TransactionType.BUY,
            order_type=order_type,
            product=product,
            exchange=exchange
        )

    def sell(self, symbol: str, quantity: int, price: float = 0,
             order_type: OrderType = OrderType.MARKET,
             product: ProductType = ProductType.INTRADAY,
             exchange: str = "NSE") -> Optional[str]:
        """
        Sell a stock! 📤

        Args:
            symbol: Stock to sell
            quantity: How many shares
            price: Price (only for LIMIT orders)
            order_type: MARKET or LIMIT
            product: INTRADAY or DELIVERY
            exchange: NSE or BSE

        Returns:
            Order ID if successful, None otherwise
        """
        return self._place_order(
            symbol=symbol,
            quantity=quantity,
            price=price,
            transaction_type=TransactionType.SELL,
            order_type=order_type,
            product=product,
            exchange=exchange
        )

    def _place_order(self, symbol: str, quantity: int, price: float,
                     transaction_type: TransactionType,
                     order_type: OrderType,
                     product: ProductType,
                     exchange: str) -> Optional[str]:
        """Internal method to place orders"""
        if not self.is_connected:
            logger.error("Not connected to Zerodha")
            return None

        try:
            order_params = {
                'tradingsymbol': symbol,
                'exchange': exchange,
                'transaction_type': transaction_type.value,
                'quantity': quantity,
                'order_type': order_type.value,
                'product': product.value,
                'variety': 'regular'
            }

            # Add price for limit orders
            if order_type == OrderType.LIMIT and price > 0:
                order_params['price'] = price

            order_id = self.kite.place_order(**order_params)
            logger.info(f"Order placed: {transaction_type.value} {quantity} {symbol} - ID: {order_id}")
            return str(order_id)

        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: The order to cancel

        Returns:
            True if cancelled successfully
        """
        if not self.is_connected:
            return False

        try:
            self.kite.cancel_order(variety='regular', order_id=order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def get_orders(self) -> List[Order]:
        """Get all orders for today"""
        if not self.is_connected:
            return []

        try:
            orders = self.kite.orders()
            return [
                Order(
                    order_id=str(o['order_id']),
                    symbol=o['tradingsymbol'],
                    transaction_type=o['transaction_type'],
                    quantity=o['quantity'],
                    price=o.get('price', 0),
                    status=o['status'],
                    filled_quantity=o.get('filled_quantity', 0),
                    average_price=o.get('average_price', 0),
                    message=o.get('status_message', '')
                )
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    def get_positions(self) -> List[Position]:
        """Get all your current positions"""
        if not self.is_connected:
            return []

        try:
            positions = self.kite.positions()
            result = []

            for p in positions.get('net', []):
                if p['quantity'] != 0:
                    pnl = p.get('pnl', 0)
                    avg_price = p.get('average_price', 0)
                    pnl_percent = (pnl / (avg_price * abs(p['quantity'])) * 100) if avg_price > 0 else 0

                    result.append(Position(
                        symbol=p['tradingsymbol'],
                        quantity=p['quantity'],
                        average_price=avg_price,
                        last_price=p.get('last_price', 0),
                        pnl=pnl,
                        pnl_percent=round(pnl_percent, 2)
                    ))

            return result
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_historical_data(self, symbol: str, interval: str = "day",
                           days: int = 365, exchange: str = "NSE") -> List[Dict]:
        """
        Get historical price data.

        Args:
            symbol: Stock symbol
            interval: minute, 3minute, 5minute, 15minute, 30minute, 60minute, day
            days: How many days of data
            exchange: NSE or BSE

        Returns:
            List of OHLCV candles
        """
        if not self.is_connected:
            return []

        try:
            # Get instrument token
            instruments = self.kite.instruments(exchange)
            token = None
            for inst in instruments:
                if inst['tradingsymbol'] == symbol:
                    token = inst['instrument_token']
                    break

            if not token:
                logger.error(f"Instrument not found: {symbol}")
                return []

            # Get historical data
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)

            data = self.kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )

            return data

        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []

    def disconnect(self) -> None:
        """Disconnect from Zerodha"""
        if self.ticker:
            self.ticker.close()
        self.connected = False
        logger.info("Disconnected from Zerodha")
