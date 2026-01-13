    def update_trailing_sl(self, bracket_id: str, current_price: float):
        """Update trailing stop-loss"""
        if bracket_id not in self.orders:
            return

        bracket = self.orders[bracket_id]

        if not bracket.trailing_sl:
            return

        if bracket.status != BracketOrderStatus.SL_TARGET_PLACED:
            return

        if bracket.transaction_type == "BUY":
            # For BUY orders, trail upwards
            if bracket.highest_price is None or current_price > bracket.highest_price:
                bracket.highest_price = current_price
                new_sl = current_price - bracket.trailing_sl_points

                if new_sl > bracket.stop_loss:
                    old_sl = bracket.stop_loss
                    bracket.stop_loss = new_sl

                    # Modify SL order
                    self._modify_sl_order(bracket_id, new_sl)

                    print(f"Trailing SL updated: {bracket_id}")
                    print(f"  {old_sl:.2f} -> {new_sl:.2f}")

        else:  # SELL
            # For SELL orders, trail downwards
            if bracket.lowest_price is None or current_price < bracket.lowest_price:
                bracket.lowest_price = current_price
                new_sl = current_price + bracket.trailing_sl_points

                if new_sl < bracket.stop_loss:
                    old_sl = bracket.stop_loss
                    bracket.stop_loss = new_sl

                    # Modify SL order
                    self._modify_sl_order(bracket_id, new_sl)

                    print(f"Trailing SL updated: {bracket_id}")
                    print(f"  {old_sl:.2f} -> {new_sl:.2f}")

    def _modify_sl_order(self, bracket_id: str, new_price: float):
        """Modify stop-loss order"""
        bracket = self.orders[bracket_id]

        if self.broker:
            # Modify via broker API
            pass

        self._log_event(bracket_id, BracketLeg.STOP_LOSS, "modified",
                       {'new_price': new_price})

    def _cancel_order(self, order_id: Optional[str]):
        """Cancel an order"""
        if not order_id:
            return

        if self.broker:
            # Cancel via broker API
            pass

        print(f"  Order cancelled: {order_id}")

    def cancel_bracket_order(self, bracket_id: str):
        """Cancel entire bracket order"""
        if bracket_id not in self.orders:
            raise ValueError(f"Bracket order {bracket_id} not found")

        bracket = self.orders[bracket_id]

        # Cancel all active orders
        if bracket.entry_order_id:
            self._cancel_order(bracket.entry_order_id)
        if bracket.sl_order_id:
            self._cancel_order(bracket.sl_order_id)
        if bracket.target_order_id:
            self._cancel_order(bracket.target_order_id)

        bracket.status = BracketOrderStatus.CANCELLED

        self._log_event(bracket_id, None, "cancelled", {})  # type: ignore

        print(f"Bracket order cancelled: {bracket_id}")

    def get_bracket_order(self, bracket_id: str) -> Optional[BracketOrder]:
        """Get bracket order by ID"""
        return self.orders.get(bracket_id)

    def get_active_brackets(self) -> List[BracketOrder]:
        """Get all active bracket orders"""
        return [
            order for order in self.orders.values()
            if order.status not in [BracketOrderStatus.COMPLETED, BracketOrderStatus.CANCELLED]
        ]

    def get_pnl(self, bracket_id: str) -> Optional[float]:
        """Calculate P&L for bracket order"""
        bracket = self.orders.get(bracket_id)

        if not bracket or bracket.status != BracketOrderStatus.COMPLETED:
            return None

        if not bracket.entry_filled_price or not bracket.exit_price:
            return None

        if bracket.transaction_type == "BUY":
            pnl = (bracket.exit_price - bracket.entry_filled_price) * bracket.exit_qty
        else:
            pnl = (bracket.entry_filled_price - bracket.exit_price) * bracket.exit_qty

        return pnl

    def _log_event(self, bracket_id: str, leg: Optional[BracketLeg], event: str, data: Dict):
        """Log bracket order event"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'bracket_id': bracket_id,
            'leg': leg.value if leg else None,
            'event': event,
            'data': data
        }

        self.order_history.append(log_entry)

    def export_bracket_orders(self, filename: str = "bracket_orders.json"):
        """Export bracket orders to JSON"""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_brackets': len(self.orders),
            'active_brackets': len(self.get_active_brackets()),
            'orders': [
                {
                    'bracket_id': b.bracket_id,
                    'symbol': b.symbol,
                    'quantity': b.quantity,
                    'entry_price': b.entry_price,
                    'stop_loss': b.stop_loss,
                    'target': b.target,
                    'transaction_type': b.transaction_type,
                    'status': b.status.value,
                    'entry_filled_price': b.entry_filled_price,
                    'exit_price': b.exit_price,
                    'pnl': self.get_pnl(b.bracket_id),
                    'created_at': b.created_at.isoformat()
                }
                for b in self.orders.values()
            ],
            'history': self.order_history
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Exported {len(self.orders)} bracket orders to {filename}")


if __name__ == "__main__":
    print("Bracket Orders (OCO - One-Cancels-Other)")
    print("=" * 60)
    print("\nFeatures:")
    print("  Entry order with automatic SL and target")
    print("  OCO logic (one cancels the other)")
    print("  Trailing stop-loss")
    print("  Partial exit management")
    print("  Order modification support")
