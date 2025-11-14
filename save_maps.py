import argparse
import pathlib
import numpy as np

from grid_config_generator import add_grid_config_args, grid_config_generator_factory


def main():
    parser = argparse.ArgumentParser(description="Run Expert")
    parser = add_grid_config_args(parser)

    args = parser.parse_args()
    print(args)

    rng = np.random.default_rng(args.map_seed)
    seeds = rng.integers(10**10, size=args.num_maps)

    _grid_config_generator = grid_config_generator_factory(args)

    for map_id, seed in enumerate(seeds):
        grid_config = _grid_config_generator(seed)

        obstacles = np.array(grid_config.map)
        height = obstacles.shape[0]
        width = obstacles.shape[1]

        out_str = f"type octile\nheight {height}\nwidth {width}\nmap"
        for i in range(height):
            out_str += "\n"
            for j in range(width):
                if obstacles[i, j] == 0:
                    out_str += "."
                else:
                    out_str += "@"

        path = pathlib.Path(args.map_dir, f"{map_id}.map")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(out_str)

if __name__ == "__main__":
    main()
