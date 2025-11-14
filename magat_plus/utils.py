from pibt.pypibt.mapf_utils import is_valid_coord

def get_neighbors(grid, coord, moves):
    # coord: y, x
    neigh = []
    move_idx = []
    mask = []

    # check valid input
    if not is_valid_coord(grid, coord):
        return neigh, move_idx

    y, x = coord

    for i, (dy, dx) in enumerate(moves):
        if is_valid_coord(grid, (y + dy, x + dx)):
            neigh.append((y + dy, x + dx))
            move_idx.append(i)
            mask.append(True)
        else:
            mask.append(False)

    return neigh, move_idx, mask