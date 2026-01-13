
        logger.info(f"PaperTradingBrokerService initialized with balance: ₹{initial_balance:,.2f}")

    async def place_order(self, **kwargs) -> str:
        """
        Simulate order placement with realistic async delay

        Args:
            **kwargs: Order parameters (symbol, quantity, price, transaction_type, etc.)

        Returns:
            Order ID (UUID)
        """
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        # Generate order ID
        order_id = str(uuid.uuid4())

        # Extract order details
        symbol = kwargs.get('tradingsymbol', 'UNKNOWN')
        quantity = kwargs.get('quantity', 0)
        price = kwargs.get('price', 0.0)
        transaction_type = kwargs.get('transaction_type', 'BUY')
        order_type = kwargs.get('order_type', 'LIMIT')

        # Create order record
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'transaction_type': transaction_type,
            'order_type': order_type,
            'status': 'OPEN',
            'filled_quantity': 0,
            'average_price': 0.0,
            'timestamp': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        self.orders[order_id] = order
        self.total_orders += 1

        logger.info(f"Paper order placed: {order_id} - {transaction_type} {quantity} {symbol} @ ₹{price}")

        # Simulate immediate execution for market orders
        if order_type == 'MARKET':
            await self._execute_order(order_id, price)

        return order_id

    async def _execute_order(self, order_id: str, execution_price: float) -> None:
        """
        Simulate order execution

        Args:
            order_id: Order ID to execute
            execution_price: Price at which to execute
        """
        if order_id not in self.orders:
            return

        order = self.orders[order_id]

        # Simulate execution delay
        await asyncio.sleep(0.05)  # 50ms execution time

        # Update order status
        order['status'] = 'COMPLETE'
        order['filled_quantity'] = order['quantity']
        order['average_price'] = execution_price
        order['updated_at'] = datetime.now().isoformat()

        # Update position
        symbol = order['symbol']
        quantity = order['quantity']
        transaction_type = order['transaction_type']

        if symbol not in self.positions:
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': 0,
                'average_price': 0.0,
                'pnl': 0.0
            }

        position = self.positions[symbol]

        if transaction_type == 'BUY':
            # Update average price
            total_quantity = position['quantity'] + quantity
            if total_quantity > 0:
                position['average_price'] = (
                    (position['average_price'] * position['quantity'] + execution_price * quantity)
                    / total_quantity
                )
            position['quantity'] += quantity
        else:  # SELL
            position['quantity'] -= quantity
            # Calculate realized P&L
            realized_pnl = (execution_price - position['average_price']) * quantity
            position['pnl'] += realized_pnl
            self.balance += realized_pnl

        self.completed_orders += 1
        logger.info(f"Paper order executed: {order_id} at ₹{execution_price}")

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions (async)

        Returns:
            List of position dictionaries
        """
        # Simulate API latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        return list(self.positions.values())

    async def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get all orders (async)

        Returns:
            List of order dictionaries
        """
        # Simulate API latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        return list(self.orders.values())

    async def cancel_order(self, order_id: str) -> None:
        """
        Cancel an order (async)

        Args:
            order_id: Order ID to cancel
        """
        # Simulate API latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        if order_id in self.orders:
            order = self.orders[order_id]
            if order['status'] == 'OPEN':
                order['status'] = 'CANCELLED'
                order['updated_at'] = datetime.now().isoformat()
                self.cancelled_orders += 1
                logger.info(f"Paper order cancelled: {order_id}")
            else:
                logger.warning(f"Cannot cancel order {order_id} - status: {order['status']}")
        else:
            logger.warning(f"Order not found: {order_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get paper trading statistics"""
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'pnl': self.balance - self.initial_balance,
            'total_orders': self.total_orders,
            'completed_orders': self.completed_orders,
            'cancelled_orders': self.cancelled_orders,
            'open_positions': len([p for p in self.positions.values() if p['quantity'] != 0])
        }


class HistoricalMarketDataService(IMarketDataService):
    """
    Historical market data service for backtesting
    Replays historical data as if it were live
    """

    def __init__(self, historical_data: List[Dict[str, Any]], speed_multiplier: float = 1.0):
        """
        Initialize historical market data service

        Args:
            historical_data: List of historical tick data (sorted by timestamp)
            speed_multiplier: Speed multiplier (1.0 = real-time, 0 = as fast as possible)
        """
        self.historical_data = historical_data
        self.speed_multiplier = speed_multiplier
        self.tick_callbacks = []
        self.is_connected = False
        self.current_index = 0

        logger.info(f"HistoricalMarketDataService initialized with {len(historical_data)} ticks")

    def connect(self) -> None:
        """Mark as connected"""
        self.is_connected = True
        self.current_index = 0
        logger.info("Historical market data service connected")

    def subscribe(self, instrument_tokens: List[int]) -> None:
        """Subscribe to instruments (no-op for historical data)"""
        logger.info(f"Subscribed to {len(instrument_tokens)} instruments (historical)")

    def disconnect(self) -> None: