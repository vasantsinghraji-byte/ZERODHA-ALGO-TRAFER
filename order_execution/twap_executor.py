            twap_slice = TWAPSlice(
                slice_num=i + 1,
                scheduled_time=scheduled_time,
                quantity=qty,
                target_price=0.0
            )

            twap_order.slices.append(twap_slice)

        self.orders[twap_id] = twap_order

        print(f"Created TWAP order: {twap_id}")
        print(f"  {num_slices} slices over {duration_minutes} minutes")
        print(f"  Slice size: ~{slice_qty}")

        return twap_order

    def start_execution(self, twap_id: str):
        """Start TWAP execution"""
        if twap_id not in self.orders:
            raise ValueError(f"TWAP order {twap_id} not found")

        twap = self.orders[twap_id]
        twap.status = TWAPStatus.ACTIVE

        print(f"Started TWAP execution: {twap_id}")

    def execute_slice(self, twap_id: str, slice_num: int, fill_price: float):
        """Execute a TWAP slice"""
        twap = self.orders.get(twap_id)
        if not twap:
            return

        if slice_num < 1 or slice_num > len(twap.slices):
            return

        slice_obj = twap.slices[slice_num - 1]
        slice_obj.executed = True
        slice_obj.actual_price = fill_price
        slice_obj.filled_qty = slice_obj.quantity
        slice_obj.execution_time = datetime.now(tz=timezone.utc)

        twap.total_filled += slice_obj.filled_qty

        from decimal import Decimal
        d_prev = Decimal(str(twap.avg_fill_price)) * (twap.total_filled - slice_obj.filled_qty)
        d_new = Decimal(str(fill_price)) * slice_obj.filled_qty
        twap.avg_fill_price = float((d_prev + d_new) / twap.total_filled)

        print(f"  Slice {slice_num} executed: {slice_obj.quantity} @ {fill_price}")

        if twap.total_filled >= twap.total_quantity:
            twap.status = TWAPStatus.COMPLETED
            print(f"TWAP completed: {twap_id}")


if __name__ == "__main__":
    print("TWAP - Time-Weighted Average Price")
    print("=" * 60)
