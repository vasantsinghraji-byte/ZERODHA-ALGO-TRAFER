    # Holdings breakdown
    print("\n" + "=" * 80)
    print("HOLDINGS BREAKDOWN")
    print("=" * 80)
    print("\n{:<15} {:>8} {:>12} {:>12} {:>12} {:>10}".format(
        "Symbol", "Qty", "Invested", "Current", "P&L", "Weight"
    ))
    print("-" * 80)

    for _, row in df.iterrows():
        pnl_sign = "+" if row['pnl'] >= 0 else ""
        print("{:<15} {:>8} Rs.{:>9,.2f} Rs.{:>9,.2f} {:>11} {:>9.2f}%".format(
            row['symbol'],
            int(row['qty']),
            row['invested'],
            row['current'],
            f"{pnl_sign}Rs.{row['pnl']:,.2f}",
            row['weight']
        ))
