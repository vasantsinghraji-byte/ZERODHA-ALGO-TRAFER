            symbol=symbol,
            total_quantity=total_quantity,
            strategy=strategy,
            transaction_type=transaction_type
        )

        self.orders[adaptive_id] = adaptive_order

        print(f"Created adaptive order: {adaptive_id}")
        print(f"  Strategy: {strategy.value}")
        print(f"  Initial slice: {adaptive_order.current_slice_size}")

        return adaptive_order

    def adapt_to_market(self, adaptive_id: str, market_volatility: float, spread_bps: float):
        """Adapt execution to market conditions"""
        adaptive = self.orders.get(adaptive_id)
        if not adaptive:
            return

        # Adjust slice size based on volatility and spread
        volatility_threshold = getattr(adaptive, 'volatility_threshold', 0.02)
        spread_threshold_bps = getattr(adaptive, 'spread_threshold_bps', 20)

        if market_volatility > volatility_threshold:  # High volatility
            adaptive.current_slice_size = max(1, adaptive.current_slice_size // 2)
            print(f"  Reduced slice size due to high volatility")
        elif spread_bps > spread_threshold_bps:  # Wide spread
            adaptive.current_slice_size = max(1, adaptive.current_slice_size // 2)
            print(f"  Reduced slice size due to wide spread")

    def execute_slice(self, adaptive_id: str, fill_price: float, fill_qty: int):
        """Execute slice with adaptive logic"""
        adaptive = self.orders.get(adaptive_id)
        if not adaptive:
            return

        adaptive.total_filled += fill_qty
        adaptive.remaining_qty -= fill_qty

        from decimal import Decimal
        d_prev = Decimal(str(adaptive.avg_fill_price)) * (adaptive.total_filled - fill_qty)
        d_new = Decimal(str(fill_price)) * fill_qty
        adaptive.avg_fill_price = float((d_prev + d_new) / adaptive.total_filled)

        print(f"  Adaptive slice executed: {fill_qty} @ {fill_price}")
        print(f"  Remaining: {adaptive.remaining_qty}")


if __name__ == "__main__":
    print("Adaptive Execution - Market Impact Aware")
    print("=" * 60)
