# scripts/utils.py
def print_table(title, headers, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

    def line(cells):
        # first column left-aligned (labels), the rest right-aligned (numbers)
        return "  ".join(
            f"{str(c):{'<' if i == 0 else '>'}{w}}"
            for i, (c, w) in enumerate(zip(cells, widths))
        )

    print(title)
    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line(row))
    print()
