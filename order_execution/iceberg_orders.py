            iceberg_slice.placed_at = datetime.now(tz=timezone.utc)

        iceberg.slices.append(iceberg_slice)
        iceberg.active_slice = iceberg_slice

        self._log_event(iceberg_id, "slice_placed", {
            'slice_id': slice_id,
            'quantity': slice_qty,
            'order_id': order_id
        })

        print(f"  Placed slice {len(iceberg.slices)}: {slice_qty} @ {iceberg.price}")

    def on_slice_fill(self,
                     iceberg_id: str,
                     filled_qty: int,
                     fill_price: float,
                     partial: bool = False):
        """
        Handle slice fill

        Args:
            iceberg_id: Iceberg order ID
            filled_qty: Filled quantity
            fill_price: Fill price
            partial: Is partial fill
        """
        if iceberg_id not in self.orders:
            return

        with self._lock:
            iceberg = self.orders[iceberg_id]

            if not iceberg.active_slice:
                return

            # Update slice
            slice_obj = iceberg.active_slice
            slice_obj.filled_qty += filled_qty
            slice_obj.avg_price = fill_price

            # Update iceberg totals
            iceberg.total_filled += filled_qty
            iceberg.remaining_qty -= filled_qty

            # Update average fill price using Decimal for precision
            from decimal import Decimal
            d_prev = Decimal(str(iceberg.avg_fill_price)) * (iceberg.total_filled - filled_qty)
            d_new = Decimal(str(fill_price)) * filled_qty
            iceberg.avg_fill_price = float((d_prev + d_new) / iceberg.total_filled)

            # Calculate price improvement
            d_price = Decimal(str(iceberg.price))
            d_fill = Decimal(str(fill_price))
            if iceberg.transaction_type == "BUY":
                improvement = d_price - d_fill
            else:
                improvement = d_fill - d_price

            iceberg.total_price_improvement += float(improvement * filled_qty)

            # Update status
            if slice_obj.filled_qty >= slice_obj.quantity:
                slice_obj.status = "filled"
                slice_obj.filled_at = datetime.now(tz=timezone.utc)

                if iceberg.remaining_qty > 0:
                    # Place next slice
                    self._place_next_slice(iceberg_id)
                else:
                    self._complete_iceberg(iceberg_id)

            elif partial:
                # Check if should replenish
                fill_ratio = slice_obj.filled_qty / slice_obj.quantity

                if fill_ratio >= iceberg.replenish_threshold:
                    # Cancel current slice and place new one
                    print(f"  Replenishing slice (filled {fill_ratio*100:.0f}%)")
                    self._place_next_slice(iceberg_id)

            self._log_event(iceberg_id, "slice_filled", {
                'slice_id': slice_obj.slice_id,
                'filled_qty': filled_qty,
                'fill_price': fill_price,
                'partial': partial,
                'total_filled': iceberg.total_filled,
                'remaining': iceberg.remaining_qty
            })

            print(f"  Slice filled: {filled_qty} @ {fill_price}")
            print(f"  Progress: {iceberg.total_filled}/{iceberg.total_quantity} ({iceberg.total_filled/iceberg.total_quantity*100:.1f}%)")

    def _complete_iceberg(self, iceberg_id: str):
        """Complete iceberg order"""
        iceberg = self.orders[iceberg_id]

        iceberg.status = IcebergStatus.COMPLETED

        # Calculate execution time
        if iceberg.slices:
            first_slice = iceberg.slices[0]
            last_slice = iceberg.slices[-1]
