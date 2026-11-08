"""Stage 6: sudoku rule check of the recognised givens."""


def check_validity(grid):
    """
    grid: 9x9 list, 0 = empty. A grid is valid when no digit repeats in any
    row, column or 3x3 box. Returns (is_valid, conflicts) where conflicts is
    a sorted list of [row, col] cells that take part in a repetition.
    """
    units = []
    for i in range(9):
        units.append([(i, c) for c in range(9)])                      # row i
        units.append([(r, i) for r in range(9)])                      # column i
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            units.append([(br + dr, bc + dc) for dr in range(3) for dc in range(3)])

    conflicts = set()
    for unit in units:
        seen = {}
        for (r, c) in unit:
            d = grid[r][c]
            if d == 0:
                continue
            seen.setdefault(d, []).append((r, c))
        for cells in seen.values():
            if len(cells) > 1:
                conflicts.update(cells)

    return len(conflicts) == 0, sorted([list(cell) for cell in conflicts])
