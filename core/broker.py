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
    from kiteconnect import exceptions as kite_exceptions
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    kite_exceptions = None
    logger.warning("kiteconnect not installed. Run: pip install kiteconnect")

# Import requests exceptions for network error handling
try:
    from requests.exceptions import ConnectionError, Timeout, HTTPError
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    ConnectionError = Exception
    Timeout = Exception
    HTTPError = Exception


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
            logger.error("Not connected to broker")
            return None

        instrument = f"{exchange}:{symbol}"

        try:
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
            else:
                logger.warning(f"No quote data returned for {symbol}")
                return None

        except ConnectionError as e:
            logger.error(f"Network error fetching quote for {symbol}: {e}")
            return None
        except Timeout as e:
            logger.error(f"Timeout fetching quote for {symbol}: {e}")
            return None
        except HTTPError as e:
            status_code = getattr(getattr(e, 'response', None), 'status_code', None)
            if status_code == 429:
                logger.warning(f"Rate limit exceeded for {symbol}, please wait before retrying")
            elif status_code == 403:
                logger.error(f"Authentication failed for {symbol} - token may have expired")
                self.connected = False
            elif status_code == 400:
                logger.error(f"Invalid request for {symbol}: {e}")
            else:
                logger.error(f"HTTP error {status_code} fetching quote for {symbol}: {e}")
            return None
        except Exception as e:
            # Handle kiteconnect specific exceptions
            if kite_exceptions:
                if isinstance(e, kite_exceptions.TokenException):
                    logger.error(f"Token expired/invalid for {symbol}: {e}")
                    self.connected = False
                elif isinstance(e, kite_exceptions.NetworkException):
                    logger.error(f"Kite network error for {symbol}: {e}")
                elif isinstance(e, kite_exceptions.DataException):
                    logger.warning(f"No data available for {symbol}: {e}")
                elif isinstance(e, kite_exceptions.InputException):
                    logger.error(f"Invalid input for {symbol}: {e}")
                else:
                    logger.error(f"Unexpected error fetching quote for {symbol}: {e}", exc_info=True)
            else:
                logger.error(f"Unexpected error fetching quote for {symbol}: {e}", exc_info=True)
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
        """
        Internal method to place orders.

        Handles various error conditions:
        - Network errors (connection, timeout)
        - Authentication failures (token expiry)
        - Rate limiting
        - Insufficient funds/margin
        - Invalid order parameters
        """
        if not self.is_connected:
            logger.error("Not connected to Zerodha")
            return None

        order_desc = f"{transaction_type.value} {quantity} {symbol}"

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
            logger.info(f"Order placed: {order_desc} - ID: {order_id}")
            return str(order_id)

        except ConnectionError as e:
            logger.error(f"Network error placing order {order_desc}: {e}")
            return None
        except Timeout as e:
            logger.error(f"Timeout placing order {order_desc}: {e}")
            return None
        except HTTPError as e:
            status_code = getattr(getattr(e, 'response', None), 'status_code', None)
            if status_code == 429:
                logger.warning(f"Rate limit exceeded for order {order_desc}, please wait before retrying")
            elif status_code == 403:
                logger.error(f"Authentication failed for order {order_desc} - token may have expired")
                self.connected = False
            elif status_code == 400:
                logger.error(f"Invalid order parameters for {order_desc}: {e}")
            else:
                logger.error(f"HTTP error {status_code} placing order {order_desc}: {e}")
            return None
        except Exception as e:
            # Handle kiteconnect specific exceptions
            error_msg = str(e).lower()
            if kite_exceptions:
                if isinstance(e, kite_exceptions.TokenException):
                    logger.error(f"Token expired/invalid for order {order_desc}: {e}")
                    self.connected = False
                elif isinstance(e, kite_exceptions.NetworkException):
                    logger.error(f"Kite network error for order {order_desc}: {e}")
                elif isinstance(e, kite_exceptions.InputException):
                    logger.error(f"Invalid order input for {order_desc}: {e}")
                elif isinstance(e, kite_exceptions.OrderException):
                    # Check for specific order errors
                    if 'insufficient' in error_msg or 'margin' in error_msg:
                        logger.error(f"Insufficient funds/margin for {order_desc}: {e}")
                    elif 'quantity' in error_msg:
                        logger.error(f"Invalid quantity for {order_desc}: {e}")
                    else:
                        logger.error(f"Order rejected for {order_desc}: {e}")
                else:
                    logger.error(f"Unexpected error placing order {order_desc}: {e}", exc_info=True)
            else:
                # Fallback error classification without kite_exceptions
                if 'insufficient' in error_msg or 'margin' in error_msg:
                    logger.error(f"Insufficient funds/margin for {order_desc}: {e}")
                elif 'token' in error_msg or 'auth' in error_msg:
                    logger.error(f"Authentication error for {order_desc}: {e}")
                    self.connected = False
                else:
                    logger.error(f"Order failed for {order_desc}: {e}", exc_info=True)
            return None

    def place_stop_loss_order(
        self,
        symbol: str,
        quantity: int,
        trigger_price: float,
        transaction_type: TransactionType = TransactionType.SELL,
        product: ProductType = ProductType.INTRADAY,
        exchange: str = "NSE"
    ) -> Optional[str]:
        """
        Place a server-side Stop Loss Market (SL-M) order.

        CRITICAL FOR RISK MANAGEMENT:
        This order lives at the broker/exchange level, NOT in Python.
        If your script crashes, the stop loss is STILL ACTIVE.

        Args:
            symbol: Stock symbol
            quantity: Number of shares to exit
            trigger_price: Price at which stop loss triggers
            transaction_type: SELL for long positions, BUY for short positions
            product: INTRADAY or DELIVERY
            exchange: NSE or BSE

        Returns:
            Order ID if successful, None otherwise

        Example:
            # After buying RELIANCE at 2500, place server-side SL at 2450
            >>> sl_order_id = broker.place_stop_loss_order(
            ...     symbol="RELIANCE",
            ...     quantity=10,
            ...     trigger_price=2450,  # 2% below entry
            ...     transaction_type=TransactionType.SELL
            ... )
        """
        if not self.is_connected:
            logger.error("Not connected to Zerodha")
            return None

        order_desc = f"SL-M {transaction_type.value} {quantity} {symbol} @ trigger {trigger_price}"

        try:
            order_params = {
                'tradingsymbol': symbol,
                'exchange': exchange,
                'transaction_type': transaction_type.value,
                'quantity': quantity,
                'order_type': OrderType.SL_M.value,
                'product': product.value,
                'variety': 'regular',
                'trigger_price': trigger_price
            }

            order_id = self.kite.place_order(**order_params)
            logger.info(f"Server-side SL order placed: {order_desc} - ID: {order_id}")
            return str(order_id)

        except ConnectionError as e:
            logger.error(f"Network error placing SL order {order_desc}: {e}")
            return None
        except Timeout as e:
            logger.error(f"Timeout placing SL order {order_desc}: {e}")
            return None
        except HTTPError as e:
            status_code = getattr(getattr(e, 'response', None), 'status_code', None)
            logger.error(f"HTTP error {status_code} placing SL order {order_desc}: {e}")
            return None
        except Exception as e:
            if kite_exceptions and isinstance(e, kite_exceptions.OrderException):
                logger.error(f"SL order rejected for {order_desc}: {e}")
            else:
                logger.error(f"SL order failed for {order_desc}: {e}", exc_info=True)
            return None

    def modify_stop_loss_order(
        self,
        order_id: str,
        new_trigger_price: float,
        quantity: int = None
    ) -> bool:
        """
        Modify an existing stop loss order (for trailing stops).

        Args:
            order_id: The SL order to modify
            new_trigger_price: New trigger price
            quantity: New quantity (optional)

        Returns:
            True if successful
        """
        if not self.is_connected:
            logger.error("Not connected to broker")
            return False

        try:
            modify_params = {
                'variety': 'regular',
                'order_id': order_id,
                'trigger_price': new_trigger_price
            }
            if quantity:
                modify_params['quantity'] = quantity

            self.kite.modify_order(**modify_params)
            logger.info(f"SL order {order_id} modified to trigger @ {new_trigger_price}")
            return True
        except Exception as e:
            logger.error(f"Failed to modify SL order {order_id}: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: The order to cancel

        Returns:
            True if cancelled successfully
        """
        if not self.is_connected:
            logger.error("Not connected to broker")
            return False

        try:
            self.kite.cancel_order(variety='regular', order_id=order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except ConnectionError as e:
            logger.error(f"Network error cancelling order {order_id}: {e}")
            return False
        except Timeout as e:
            logger.error(f"Timeout cancelling order {order_id}: {e}")
            return False
        except HTTPError as e:
            status_code = getattr(getattr(e, 'response', None), 'status_code', None)
            if status_code == 429:
                logger.warning(f"Rate limit exceeded cancelling order {order_id}")
            elif status_code == 403:
                logger.error(f"Authentication failed cancelling order {order_id}")
                self.connected = False
            else:
                logger.error(f"HTTP error {status_code} cancelling order {order_id}: {e}")
            return False
        except Exception as e:
            error_msg = str(e).lower()
            if 'already' in error_msg or 'completed' in error_msg or 'rejected' in error_msg:
                logger.warning(f"Order {order_id} cannot be cancelled (already processed): {e}")
            elif kite_exceptions and isinstance(e, kite_exceptions.TokenException):
                logger.error(f"Token expired cancelling order {order_id}: {e}")
                self.connected = False
            else:
                logger.error(f"Failed to cancel order {order_id}: {e}")
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
