                if 0 <= y < height:
                    chart[y][i] = body_char

        # Convert to string
        output = []
        output.append("ASCII Price Chart (Recent {} bars)".format(len(df)))
        output.append("─" * width)

        for y, row in enumerate(chart):
            price = price_max - (y / (height - 1) * price_range)
            output.append(f"{price:7.2f} │ {''.join(row)}")

        output.append("─" * width)

        return "\n".join(output)