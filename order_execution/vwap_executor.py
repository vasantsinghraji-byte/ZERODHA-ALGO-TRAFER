            strategy=strategy
        )

        # Calculate volume profile and slices
        if historical_data is not None:
            self._calculate_volume_slices(vwap_order, historical_data)

        self.orders[vwap_id] = vwap_order

        print(f"Created VWAP order: {vwap_id}")
        print(f"  Participation rate: {participation_rate*100:.1f}%")
        print(f"  Slices: {len(vwap_order.slices)}")

        return vwap_order

    def _calculate_volume_slices(self, vwap_order: VWAPOrder, historical_data: pd.DataFrame):
        """Calculate volume-weighted slices"""
        if 'volume' not in historical_data.columns:
            return

        # Group by hour or period
        total_volume = historical_data['volume'].sum()

        if total_volume == 0:
            return

        # Create slices based on volume distribution
        num_periods = min(len(historical_data), 20)
        period_size = len(historical_data) // num_periods

        carried = 0.0
        for i in range(num_periods):
            start_idx = i * period_size
            end_idx = min((i + 1) * period_size, len(historical_data))

            period_volume = historical_data['volume'].iloc[start_idx:end_idx].sum()
            volume_pct = period_volume / total_volume

            exact_qty = vwap_order.total_quantity * volume_pct + carried
            slice_qty = int(exact_qty)
            carried = exact_qty - slice_qty

            if slice_qty > 0:
                vwap_slice = VWAPSlice(
                    slice_num=i + 1,
                    target_volume_pct=volume_pct,
                    quantity=slice_qty
                )
                vwap_order.slices.append(vwap_slice)

        # Add remainder to last slice to avoid dust quantity loss
        if vwap_order.slices:
            allocated = sum(s.quantity for s in vwap_order.slices)
            remainder = vwap_order.total_quantity - allocated
            if remainder > 0:
                vwap_order.slices[-1].quantity += remainder

    def start_execution(self, vwap_id: str):
        """Start VWAP execution"""
        if vwap_id not in self.orders:
            raise ValueError(f"VWAP order {vwap_id} not found")

        vwap = self.orders[vwap_id]
        vwap.status = VWAPStatus.ACTIVE

        print(f"Started VWAP execution: {vwap_id}")

    def execute_slice(self, vwap_id: str, slice_num: int, fill_price: float):
        """Execute a VWAP slice"""
        vwap = self.orders.get(vwap_id)
        if not vwap or slice_num < 1 or slice_num > len(vwap.slices):
            return

        with self._lock:
            slice_obj = vwap.slices[slice_num - 1]
            slice_obj.executed = True
            slice_obj.actual_price = fill_price
            slice_obj.filled_qty = slice_obj.quantity

            vwap.total_filled += slice_obj.filled_qty

            from decimal import Decimal
            d_prev = Decimal(str(vwap.avg_fill_price)) * (vwap.total_filled - slice_obj.filled_qty)
            d_new = Decimal(str(fill_price)) * slice_obj.filled_qty
            vwap.avg_fill_price = float((d_prev + d_new) / vwap.total_filled)

            print(f"  VWAP slice {slice_num} executed: {slice_obj.quantity} @ {fill_price}")

            if vwap.total_filled >= vwap.total_quantity:
                vwap.status = VWAPStatus.COMPLETED
                print(f"VWAP completed: {vwap_id}")


if __name__ == "__main__":
    print("VWAP - Volume-Weighted Average Price")
    print("=" * 60)
