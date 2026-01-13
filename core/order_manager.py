    def set_paper_trading(self, enabled: bool):
        """Enable/disable paper trading"""
        self.paper_trading = enabled

    def register_callback(self, event: str, callback):
        """Register callback for order events"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _trigger_callback(self, event: str, order: Order):
        """Trigger callbacks for event"""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(order)
                except Exception as e:
                    print(f"[ERROR] Callback error: {e}")

    def place_order(
        self,
        symbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "MARKET",
        product: str = "CNC",
        price: float = 0.0,
        trigger_price: float = 0.0,
        stop_loss: float = 0.0,
        target: float = 0.0