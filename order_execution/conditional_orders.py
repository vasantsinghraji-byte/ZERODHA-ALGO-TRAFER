            conditional_id=conditional_id,
            symbol=symbol,
            conditions=conditions,
            logic=logic,
            actions_if_true=actions_if_true,
            actions_if_false=actions_if_false or [],
            strategy=strategy
        )

        self.conditional_orders[conditional_id] = conditional_order

        print(f"Created conditional order: {conditional_id}")
        print(f"  Conditions: {len(conditions)} ({logic} logic)")
        print(f"  Actions if true: {len(actions_if_true)}")
        print(f"  Actions if false: {len(actions_if_false or [])}")

        return conditional_order

    def create_price_condition(self,
                              condition_type: ConditionType,
                              price: float,
                              price2: Optional[float] = None) -> Condition:
        """Create price-based condition"""
        condition_id = f"COND_{uuid.uuid4().hex}"

        params = {'price': price}
        if price2:
            params['price2'] = price2

        return Condition(
            condition_id=condition_id,
            condition_type=condition_type,
            parameters=params
        )

    def create_time_condition(self,
                             condition_type: ConditionType,
                             target_time: datetime) -> Condition:
        """Create time-based condition"""
        condition_id = f"COND_{uuid.uuid4().hex}"

        return Condition(
            condition_id=condition_id,
            condition_type=condition_type,
            parameters={'target_time': target_time}
        )

    def evaluate_conditions(self, conditional_id: str, market_data: Dict[str, Any]) -> bool:
        """
        Evaluate all conditions

        Args:
            conditional_id: Conditional order ID
            market_data: Current market data

        Returns:
            True if conditions met
        """
        conditional = self.conditional_orders.get(conditional_id)
        if not conditional or not conditional.is_active:
            return False

        def _evaluate_and_record(condition):
            is_met = self._evaluate_single_condition(condition, market_data)
            condition.is_met = is_met
            condition.evaluated_at = datetime.now(tz=timezone.utc)
            return is_met

        # Short-circuit: AND stops at first False, OR stops at first True
        if conditional.logic == "AND":
            all_met = all(_evaluate_and_record(c) for c in conditional.conditions)
        else:  # OR
            all_met = any(_evaluate_and_record(c) for c in conditional.conditions)

        conditional.conditions_met = all_met

        # Execute actions if state changed
        if all_met and not conditional.actions_executed:
            self._execute_actions(conditional_id, conditional.actions_if_true)
            conditional.actions_executed = True
            conditional.executed_at = datetime.now(tz=timezone.utc)
            return True

        elif not all_met and conditional.actions_if_false and not conditional.actions_executed:
            self._execute_actions(conditional_id, conditional.actions_if_false)

        return all_met

    def _evaluate_single_condition(self, condition: Condition, market_data: Dict[str, Any]) -> bool:
        """Evaluate single condition"""
        current_price = market_data.get('ltp', 0)

        if condition.condition_type == ConditionType.PRICE_ABOVE:
            return current_price > condition.parameters['price']

        elif condition.condition_type == ConditionType.PRICE_BELOW:
            return current_price < condition.parameters['price']
