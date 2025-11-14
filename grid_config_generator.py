from typing import Optional
import numpy as np
import argparse
from dataclasses import dataclass
import pathlib

from pogema import GridConfig, pogema_v0
from pogema.generator import bfs
from pogema_toolbox.generators.maze_generator import MazeGenerator


def add_grid_config_args_mixed(parser):
    parser.add_argument("--map_types", type=str, default="random=0.1+maze=0.9")
    parser.add_argument("--map_w_min", type=int, default=16)
    parser.add_argument("--map_w_max", type=int, default=20)
    parser.add_argument("--num_agents", type=str, default="16+24+32")
    parser.add_argument("--obstacle_density_min", type=float, default=0.2)
    parser.add_argument("--obstacle_density_max", type=float, default=1.0)
    parser.add_argument("--go_straight_min", type=float, default=0.75)
    parser.add_argument("--go_straight_max", type=float, default=0.85)

    parser.add_argument("--wall_width_min", type=int, default=4)
    parser.add_argument("--wall_width_max", type=int, default=7)
    parser.add_argument("--wall_height_min", type=int, default=2)
    parser.add_argument("--wall_height_max", type=int, default=2)
    parser.add_argument("--side_pad", type=int, default=2)
    parser.add_argument("--horizontal_gap", type=int, default=1)
    parser.add_argument("--vertical_gap", type=int, default=3)
    parser.add_argument("--vertical_gap_min", type=int, default=None)
    parser.add_argument("--vertical_gap_max", type=int, default=None)
    parser.add_argument("--num_wall_rows_min", type=int, default=None)
    parser.add_argument("--num_wall_rows_max", type=int, default=None)
    parser.add_argument("--num_wall_cols_min", type=int, default=None)
    parser.add_argument("--num_wall_cols_max", type=int, default=None)
    parser.add_argument("--wfi_instance", action="store_true", default=False)
    parser.add_argument(
        "--block_extra_space", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--room_width_min", type=int, default=5)
    parser.add_argument("--room_width_max", type=int, default=9)
    parser.add_argument("--room_height_min", type=int, default=5)
    parser.add_argument("--room_height_max", type=int, default=9)
    parser.add_argument("--num_rows_min", type=int, default=3)
    parser.add_argument("--num_rows_max", type=int, default=5)
    parser.add_argument("--num_cols_min", type=int, default=3)
    parser.add_argument("--num_cols_max", type=int, default=5)
    parser.add_argument("--room_grid_uniform", action="store_true", default=True)
    parser.add_argument(
        "--room_only_centre_obstacles", action="store_true", default=False
    )

    parser.add_argument(
        "--ensure_grid_config_is_generatable",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--regulate_obstacle_density_max",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Params for pre-generating maps
    parser.add_argument("--map_dir", type=str, default=None)
    parser.add_argument("--num_maps", type=int, default=1)
    parser.add_argument("--map_seed", type=int, default=17)

    return parser


def add_grid_config_args(parser):
    parser.add_argument("--map_type", type=str, default="RandomGrid")
    parser.add_argument("--map_h", type=int, default=20)
    parser.add_argument("--map_w", type=int, default=20)
    parser.add_argument("--robot_density", type=float, default=0.025)
    parser.add_argument("--obstacle_density", type=float, default=0.1)
    parser.add_argument("--max_episode_steps", type=int, default=128)
    parser.add_argument("--obs_radius", type=int, default=4)
    parser.add_argument("--collision_system", type=str, default="soft")
    parser.add_argument("--on_target", type=str, default="nothing")

    parser.add_argument("--min_dist", type=int, default=None)
    parser.add_argument("--max_dist", type=int, default=None)

    parser = add_grid_config_args_mixed(parser)

    return parser


def generate_grid_config_from_env(env, max_episode_steps=None):
    config = env.grid.config
    if max_episode_steps is None:
        max_episode_steps = config.max_episode_steps

    return GridConfig(
        num_agents=config.num_agents,  # number of agents
        size=config.size,  # size of the grid
        density=config.density,  # obstacle density
        seed=config.seed,
        max_episode_steps=max_episode_steps,  # horizon
        obs_radius=config.obs_radius,  # defines field of view
        observation_type=config.observation_type,
        collision_system=config.collision_system,
        on_target=config.on_target,
        map=env.grid.get_obstacles(ignore_borders=True).tolist(),
        agents_xy=env.grid.get_agents_xy(ignore_borders=True),
        targets_xy=env.grid.get_targets_xy(ignore_borders=True),
    )


MOVES = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]
START_ID = 2


def _get_components(obstacles, map_w, start_id=START_ID, moves=MOVES):
    grid = obstacles.copy()

    components = bfs(grid, moves, map_w, start_id, free_cell=0)
    return grid, components


def generate_start_target_pairs(obstacles, map_h, map_w, min_dist, max_dist=None):
    # All possible pairs [start_x, start_y, target_x, target_y]
    possible_pairs = np.ones((map_h, map_w, map_h, map_w), dtype=int)

    # Removing obstacles
    possible_pairs *= 1 - obstacles[:, :, None, None]
    possible_pairs *= 1 - obstacles[None, None, :, :]

    # Removing pairs that are too close
    locs = np.zeros((map_h, map_w, 2), dtype=int)
    locs[:, :, 0] = np.arange(map_h)[:, None]
    locs[:, :, 1] = np.arange(map_w)[None, :]

    pdist = np.sum(
        np.abs(locs[:, :, None, None] - locs[None, None, :, :]), axis=-1
    )  # Manhattan Distance

    possible_pairs *= pdist >= min_dist
    if max_dist is not None:
        possible_pairs *= pdist <= max_dist

    # Removing unreachable pairs
    component_map, components = _get_components(obstacles, map_w, start_id=START_ID)

    reachable_pairs = np.zeros((map_h, map_w, map_h, map_w), dtype=bool)
    for i in range(START_ID, len(components)):
        component = component_map == i
        reachable_pairs += component[:, :, None, None] * component[None, None, :, :]

    possible_pairs *= reachable_pairs

    return np.stack(np.nonzero(possible_pairs), axis=-1)


class GridConfigError(BaseException):
    def __init__(self, seed):
        message = f"Could not generate enough valid start and target positions for map with seed {seed}"
        super().__init__(message)


def generate_random_grid_with_min_dist(
    seed,
    map_w,
    num_agents,
    obstacle_density,
    obs_radius,
    collision_system,
    on_target,
    min_dist,
    max_episode_steps,
    num_tries=5,
):
    rng = np.random.default_rng(seed)

    obstacles = rng.binomial(1, obstacle_density, (map_w, map_w))

    # Generating start and target positions
    possible_pairs = generate_start_target_pairs(
        obstacles=obstacles, map_h=map_w, map_w=map_w, min_dist=min_dist
    )

    for _ in range(num_tries):
        # Trying out a random permutation
        possible_pairs = rng.permutation(possible_pairs, axis=0)

        start_positions = []
        target_positions = []
        for start_x, start_y, target_x, target_y in possible_pairs.tolist():
            start_pos = [start_x, start_y]
            target_pos = [target_x, target_y]
            if (
                start_pos in start_positions
                or start_pos in target_positions
                or target_pos in start_positions
                or target_pos in target_positions
            ):
                continue
            else:
                start_positions.append(start_pos)
                target_positions.append(target_pos)
            if len(start_positions) == num_agents:
                break

        if len(start_positions) == num_agents:
            # No need to retry
            break

    if len(start_positions) < num_agents:
        # Could not find a suitable configuration
        raise GridConfigError(seed)

    return GridConfig(
        num_agents=num_agents,  # number of agents
        size=map_w,  # size of the grid
        density=obstacle_density,  # obstacle density
        seed=seed,
        max_episode_steps=max_episode_steps,  # horizon
        obs_radius=obs_radius,  # defines field of view
        observation_type="MAPF",
        collision_system=collision_system,
        on_target=on_target,
        map=obstacles.tolist(),
        agents_xy=start_positions,
        targets_xy=target_positions,
    )


def num_agents_sampler(num_agents, probs, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(num_agents, p=probs)


@dataclass
class MazeRangeSettings:
    size_min: int = 5
    size_max: int = 9

    obstacle_density_min: float = 0.0
    obstacle_density_max: float = 1.0

    wall_components_min: int = 1
    wall_components_max: int = 8

    go_straight_min: float = 0.75
    go_straight_max: float = 0.85

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)

        # Generate a sample for each attribute
        size = rng.integers(self.size_min, self.size_max + 1)
        obstacle_density = rng.uniform(
            self.obstacle_density_min, self.obstacle_density_max
        )
        wall_components = rng.integers(
            self.wall_components_min, self.wall_components_max + 1
        )
        go_straight = rng.uniform(self.go_straight_min, self.go_straight_max)

        # Return a dictionary with the sampled values
        return {
            "width": size,
            "height": size,
            "obstacle_density": obstacle_density,
            "wall_components": wall_components,
            "go_straight": go_straight,
            "seed": seed,
        }


@dataclass
class RandomRangeSettings:
    size_min: int = 5
    size_max: int = 9

    obstacle_density_min: float = 0.0
    obstacle_density_max: float = 1.0

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)

        # Generate a sample for each attribute
        size = rng.integers(self.size_min, self.size_max + 1)
        if self.obstacle_density_min == self.obstacle_density_max:
            obstacle_density = self.obstacle_density_min
        else:
            obstacle_density = rng.uniform(
                self.obstacle_density_min, self.obstacle_density_max
            )

        # Return a dictionary with the sampled values
        return {
            "size": size,
            "obstacle_density": obstacle_density,
            "seed": seed,
        }


@dataclass
class WarehouseRangeSettings:
    size_min: int = 5
    size_max: int = 9

    wall_width_min: int = 5
    wall_width_max: int = 5

    wall_height_min: int = 2
    wall_height_max: int = 2

    side_pad: int = 1
    horizontal_gap: int = 1

    vertical_gap: int = 3
    vertical_gap_min: Optional[int] = None
    vertical_gap_max: Optional[int] = None

    num_wall_rows_min: Optional[int] = None
    num_wall_rows_max: Optional[int] = None

    num_wall_cols_min: Optional[int] = None
    num_wall_cols_max: Optional[int] = None

    wfi_instance: bool = False
    block_extra_space: bool = True

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)

        # Generate a sample for each attribute
        wall_width = rng.integers(self.wall_width_min, self.wall_width_max + 1)
        wall_height = rng.integers(self.wall_height_min, self.wall_height_max + 1)

        vertical_gap = self.vertical_gap
        if self.vertical_gap_min is not None and self.vertical_gap_max is not None:
            vertical_gap = rng.integers(
                self.vertical_gap_min, self.vertical_gap_max + 1
            )

        if self.num_wall_rows_min is not None:
            num_wall_rows = rng.integers(
                self.num_wall_rows_min, self.num_wall_rows_max + 1
            )
            num_wall_cols = rng.integers(
                self.num_wall_cols_min, self.num_wall_cols_max + 1
            )

            height = vertical_gap * (num_wall_rows + 1) + wall_height * num_wall_rows
            width = (
                self.side_pad * 2
                + wall_width * num_wall_cols
                + self.horizontal_gap * (num_wall_cols - 1)
            )

            size = max(width, height)
        else:
            size = rng.integers(self.size_min, self.size_max + 1)

            num_wall_rows = (size - vertical_gap) // (wall_height + vertical_gap)
            num_wall_cols = (size - self.side_pad * 2 + self.horizontal_gap) // (
                wall_width + self.horizontal_gap
            )

        # Return a dictionary with the sampled values
        return {
            "width": size,
            "height": size,
            "num_wall_rows": num_wall_rows,
            "num_wall_cols": num_wall_cols,
            "wall_width": wall_width,
            "wall_height": wall_height,
            "side_pad": self.side_pad,
            "horizontal_gap": self.horizontal_gap,
            "vertical_gap": vertical_gap,
            "wfi_instance": self.wfi_instance,
            "block_extra_space": self.block_extra_space,
            "seed": seed,
        }


@dataclass
class RoomRangeSettings:
    room_width_min: int = 5
    room_width_max: int = 9

    room_height_min: int = 5
    room_height_max: int = 9

    num_rows_min: int = 3
    num_rows_max: int = 5

    num_cols_min: int = 3
    num_cols_max: int = 5

    obstacle_density_min: float = 0.0
    obstacle_density_max: float = 0.4

    uniform: bool = True
    only_centre_obstacles: bool = False

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)

        # Generate a sample for each attribute
        room_height = rng.integers(self.room_height_min, self.room_height_max + 1)
        num_rows = rng.integers(self.num_rows_min, self.num_rows_max + 1)

        if self.uniform:
            room_width = room_height
            num_cols = num_rows
        else:
            room_width = rng.integers(self.room_width_min, self.room_width_max + 1)
            num_cols = rng.integers(self.num_cols_min, self.num_cols_max + 1)

        obstacle_density = rng.uniform(
            self.obstacle_density_min, self.obstacle_density_max
        )

        # Return a dictionary with the sampled values
        return {
            "room_width": room_width,
            "room_height": room_height,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "obstacle_density": obstacle_density,
            "only_centre_obstacles": self.only_centre_obstacles,
            "seed": seed,
        }


def generate_maze(
    width, height, obstacle_density, wall_components, go_straight, seed=None
):
    rng = np.random.default_rng(seed)
    assert width > 0 and height > 0, "Width and height must be positive integers"
    maze_shape = ((height // 2) * 2 + 3, (width // 2) * 2 + 3)
    density = (
        int(maze_shape[0] * maze_shape[1] * obstacle_density // wall_components)
        if wall_components != 0
        else 0
    )

    maze_grid = np.zeros(maze_shape, dtype="int")
    maze_grid[0, :] = maze_grid[-1, :] = 1
    maze_grid[:, 0] = maze_grid[:, -1] = 1

    for i in range(density):
        x = rng.integers(0, maze_shape[1] // 2) * 2
        y = rng.integers(0, maze_shape[0] // 2) * 2
        maze_grid[y, x] = 1
        last_direction = (0, 0)  # Initial direction is null
        for j in range(wall_components):
            next_x, next_y, last_direction = MazeGenerator.select_random_neighbor(
                x, y, maze_grid, maze_shape, rng, last_direction, go_straight
            )
            if next_x is not None and maze_grid[next_y, next_x] == 0:
                maze_grid[next_y, next_x] = 1
                maze_grid[next_y + (y - next_y) // 2, next_x + (x - next_x) // 2] = 1
                x, y = next_x, next_y
    return maze_grid[1:-1, 1:-1]


def generate_maze_grid_config(
    size_min,
    size_max,
    obstacle_density_min,
    obstacle_density_max,
    go_straight_min,
    go_straight_max,
    regulate_obstacle_density_max,
    num_agents,
    probs,
    max_episode_steps,
    obs_radius,
    collision_system,
    on_target,
    seed,
):
    num_agents = num_agents_sampler(num_agents=num_agents, probs=probs, seed=seed)

    if regulate_obstacle_density_max:
        robot_density_max = num_agents / (size_min * size_min)
        obstacle_density_max = min(
            obstacle_density_max, 1 - 2 * robot_density_max - 0.05
        )

    setting = MazeRangeSettings(
        size_min=size_min,
        size_max=size_max,
        obstacle_density_min=obstacle_density_min,
        obstacle_density_max=obstacle_density_max,
        go_straight_min=go_straight_min,
        go_straight_max=go_straight_max,
    ).sample(seed=seed)

    maze = generate_maze(**setting)

    return GridConfig(
        num_agents=num_agents,
        size=maze.shape[0],
        density=setting["obstacle_density"],  # obstacle density
        seed=setting["seed"],
        max_episode_steps=max_episode_steps,  # horizon
        obs_radius=obs_radius,  # defines field of view
        observation_type="MAPF",
        collision_system=collision_system,
        on_target=on_target,
        map=maze.tolist(),
    )


def generate_random_grid_config(
    size_min,
    size_max,
    obstacle_density_min,
    obstacle_density_max,
    regulate_obstacle_density_max,
    num_agents,
    probs,
    max_episode_steps,
    obs_radius,
    collision_system,
    on_target,
    seed,
):
    num_agents = num_agents_sampler(num_agents=num_agents, probs=probs, seed=seed)

    if regulate_obstacle_density_max:
        robot_density_max = num_agents / (size_min * size_min)
        obstacle_density_max = min(
            obstacle_density_max, 1 - 2 * robot_density_max - 0.05
        )

    setting = RandomRangeSettings(
        size_min=size_min,
        size_max=size_max,
        obstacle_density_min=obstacle_density_min,
        obstacle_density_max=obstacle_density_max,
    ).sample(seed=seed)

    return GridConfig(
        num_agents=num_agents,
        size=setting["size"],
        density=setting["obstacle_density"],  # obstacle density
        seed=setting["seed"],
        max_episode_steps=max_episode_steps,  # horizon
        obs_radius=obs_radius,  # defines field of view
        observation_type="MAPF",
        collision_system=collision_system,
        on_target=on_target,
    )


def forced_generation_of_random_map(size, obstacle_density, seed):
    rng = np.random.default_rng(seed)

    random_map = rng.uniform(size=(size, size))
    random_map = (random_map < obstacle_density).astype(int)

    return random_map


def generate_force_random_grid_config(
    size_min,
    size_max,
    obstacle_density_min,
    obstacle_density_max,
    regulate_obstacle_density_max,
    num_agents,
    probs,
    max_episode_steps,
    obs_radius,
    collision_system,
    on_target,
    seed,
):
    num_agents = num_agents_sampler(num_agents=num_agents, probs=probs, seed=seed)

    if regulate_obstacle_density_max:
        robot_density_max = num_agents / (size_min * size_min)
        obstacle_density_max = min(
            obstacle_density_max, 1 - 2 * robot_density_max - 0.05
        )

    setting = RandomRangeSettings(
        size_min=size_min,
        size_max=size_max,
        obstacle_density_min=obstacle_density_min,
        obstacle_density_max=obstacle_density_max,
    ).sample(seed=seed)

    random_map = forced_generation_of_random_map(**setting)

    return GridConfig(
        num_agents=num_agents,
        size=setting["size"],
        density=setting["obstacle_density"],  # obstacle density
        seed=setting["seed"],
        max_episode_steps=max_episode_steps,  # horizon
        obs_radius=obs_radius,  # defines field of view
        observation_type="MAPF",
        collision_system=collision_system,
        on_target=on_target,
        map=random_map.tolist(),
    )


def block_extra_warehouse_space(
    grid,
    num_wall_rows,
    num_wall_cols,
    wall_width,
    wall_height,
    side_pad,
    horizontal_gap,
    vertical_gap,
):
    wh_height = vertical_gap * (num_wall_rows + 1) + wall_height * num_wall_rows
    wh_width = (
        side_pad * 2 + wall_width * num_wall_cols + horizontal_gap * (num_wall_cols - 1)
    )

    grid[wh_height:, :] = 1
    grid[:, wh_width:] = 1

    return grid


def generate_warehouse(
    width,
    height,
    num_wall_rows,
    num_wall_cols,
    wall_width,
    wall_height,
    side_pad,
    horizontal_gap,
    vertical_gap,
    wfi_instance=False,
    block_extra_space=True,
    seed=None,
):
    grid = np.zeros((height, width), dtype=int)

    for row in range(num_wall_rows):
        row_start = vertical_gap * (row + 1) + wall_height * row
        for col in range(num_wall_cols):
            col_start = side_pad + col * (wall_width + horizontal_gap)
            grid[
                row_start : row_start + wall_height, col_start : col_start + wall_width
            ] = 1

    if block_extra_space:
        grid = block_extra_warehouse_space(
            grid,
            num_wall_rows,
            num_wall_cols,
            wall_width,
            wall_height,
            side_pad,
            horizontal_gap,
            vertical_gap,
        )

    return grid


def generate_warehouse_grid_config(
    size_min,
    size_max,
    num_wall_rows_min,
    num_wall_rows_max,
    num_wall_cols_min,
    num_wall_cols_max,
    wall_width_min,
    wall_width_max,
    wall_height_min,
    wall_height_max,
    side_pad,
    horizontal_gap,
    vertical_gap,
    vertical_gap_min,
    vertical_gap_max,
    wfi_instance,
    block_extra_space,
    num_agents,
    probs,
    max_episode_steps,
    obs_radius,
    collision_system,
    on_target,
    seed,
):
    assert not wfi_instance, "Not yet supported."

    num_agents = num_agents_sampler(num_agents=num_agents, probs=probs, seed=seed)

    setting = WarehouseRangeSettings(
        size_min=size_min,
        size_max=size_max,
        num_wall_rows_min=num_wall_rows_min,
        num_wall_rows_max=num_wall_rows_max,
        num_wall_cols_min=num_wall_cols_min,
        num_wall_cols_max=num_wall_cols_max,
        wall_width_min=wall_width_min,
        wall_width_max=wall_width_max,
        wall_height_min=wall_height_min,
        wall_height_max=wall_height_max,
        side_pad=side_pad,
        horizontal_gap=horizontal_gap,
        vertical_gap=vertical_gap,
        vertical_gap_min=vertical_gap_min,
        vertical_gap_max=vertical_gap_max,
        wfi_instance=wfi_instance,
        block_extra_space=block_extra_space,
    ).sample(seed=seed)

    warehouse = generate_warehouse(**setting)
    density = np.sum(warehouse) / (warehouse.shape[0] * warehouse.shape[1])

    return GridConfig(
        num_agents=num_agents,
        size=warehouse.shape[0],
        density=density,  # obstacle density
        seed=setting["seed"],
        max_episode_steps=max_episode_steps,  # horizon
        obs_radius=obs_radius,  # defines field of view
        observation_type="MAPF",
        collision_system=collision_system,
        on_target=on_target,
        map=warehouse.tolist(),
    )


def generate_room(
    room_width,
    room_height,
    num_rows,
    num_cols,
    obstacle_density,
    only_centre_obstacles=False,
    seed=None,
):
    rng = np.random.default_rng(seed)

    room = np.zeros(
        (room_height * num_rows + num_rows - 1, room_width * num_cols + num_cols - 1),
        dtype="int",
    )

    obs_prob = rng.uniform(0, 1, size=room.shape)
    room[obs_prob < obstacle_density] = 1

    if only_centre_obstacles:
        # Remove obstacles in room edges
        room[0 :: room_height + 1, :] = 0
        room[:, 0 :: room_width + 1] = 0
        room[room_height - 1 :: room_height + 1, :] = 0
        room[:, room_width - 1 :: room_width + 1] = 0

    # Setting walls
    room[room_height :: room_height + 1, :] = 1
    room[:, room_width :: room_width + 1] = 1

    # Setting doors
    row_doors = rng.integers(low=0, high=room_width, size=(num_rows - 1, num_cols))
    offset = np.arange(num_cols) * (room_width + 1)
    offset = np.expand_dims(offset, axis=0)
    row_doors = offset + row_doors

    np.put_along_axis(room[room_height :: room_height + 1, :], row_doors, 0, axis=1)

    col_doors = rng.integers(low=0, high=room_height, size=(num_rows, num_cols - 1))
    offset = np.arange(num_rows) * (room_height + 1)
    offset = np.expand_dims(offset, axis=1)
    col_doors = offset + col_doors

    np.put_along_axis(room[:, room_width :: room_width + 1], col_doors, 0, axis=0)

    return room


def generate_room_grid_config(
    room_width_min,
    room_width_max,
    room_height_min,
    room_height_max,
    num_rows_min,
    num_rows_max,
    num_cols_min,
    num_cols_max,
    obstacle_density_min,
    obstacle_density_max,
    uniform,
    only_centre_obstacles,
    num_agents,
    probs,
    max_episode_steps,
    obs_radius,
    collision_system,
    on_target,
    seed,
):
    num_agents = num_agents_sampler(num_agents=num_agents, probs=probs, seed=seed)

    setting = RoomRangeSettings(
        room_width_min=room_width_min,
        room_width_max=room_width_max,
        room_height_min=room_height_min,
        room_height_max=room_height_max,
        num_rows_min=num_rows_min,
        num_rows_max=num_rows_max,
        num_cols_min=num_cols_min,
        num_cols_max=num_cols_max,
        obstacle_density_min=obstacle_density_min,
        obstacle_density_max=obstacle_density_max,
        uniform=uniform,
        only_centre_obstacles=only_centre_obstacles,
    ).sample(seed=seed)

    room = generate_room(**setting)
    density = np.sum(room) / (room.shape[0] * room.shape[1])

    return GridConfig(
        num_agents=num_agents,
        density=density,  # obstacle density
        seed=setting["seed"],
        max_episode_steps=max_episode_steps,  # horizon
        obs_radius=obs_radius,  # defines field of view
        observation_type="MAPF",
        collision_system=collision_system,
        on_target=on_target,
        map=room.tolist(),
    )


def generate_predefined_map_grid_config(
    map_dir,
    num_maps,
    num_agents,
    probs,
    max_episode_steps,
    obs_radius,
    collision_system,
    on_target,
    seed,
):
    num_agents = num_agents_sampler(num_agents=num_agents, probs=probs, seed=seed)

    rng = np.random.default_rng(seed)
    map_id = rng.integers(0, num_maps)

    map_path = pathlib.Path(map_dir, f"{map_id}.map")

    with open(map_path, "r") as f:
        lines = f.readlines()

    map_start = None
    for i, line in enumerate(lines):
        if line.strip() == "map":
            map_start = i + 1
            break

    if map_start is None:
        raise ValueError(f"No 'map' section found in {map_path}")

    map_lines = lines[map_start:]
    obstacles = []
    for line in map_lines:
        obs_line = []
        for cell in line.strip():
            if cell == "@" or cell == "T" or cell == "O":
                obs_line.append(1)
            else:
                obs_line.append(0)
        obstacles.append(obs_line)

    density = np.mean(obstacles)

    return GridConfig(
        num_agents=num_agents,
        density=density,  # obstacle density
        seed=seed,
        max_episode_steps=max_episode_steps,  # horizon
        obs_radius=obs_radius,  # defines field of view
        observation_type="MAPF",
        collision_system=collision_system,
        on_target=on_target,
        map=obstacles,
    )


def generate_min_dist_grid_config(grid_config, min_dist, max_dist, num_tries):
    rng = np.random.default_rng(grid_config.seed)
    obstacles = np.array(grid_config.map)

    # Generating start and target positions
    possible_pairs = generate_start_target_pairs(
        obstacles=obstacles,
        map_h=obstacles.shape[0],
        map_w=obstacles.shape[1],
        min_dist=min_dist,
        max_dist=max_dist,
    )

    num_agents = grid_config.num_agents

    for _ in range(num_tries):
        # Trying out a random permutation
        possible_pairs = rng.permutation(possible_pairs, axis=0)

        start_positions = []
        target_positions = []
        for start_x, start_y, target_x, target_y in possible_pairs.tolist():
            start_pos = [start_x, start_y]
            target_pos = [target_x, target_y]
            if start_pos in start_positions or target_pos in target_positions:
                continue
            else:
                start_positions.append(start_pos)
                target_positions.append(target_pos)
            if len(start_positions) == num_agents:
                break

        if len(start_positions) == num_agents:
            # No need to retry
            break

    if len(start_positions) < num_agents:
        # Could not find a suitable configuration
        raise GridConfigError(grid_config.seed)

    return GridConfig(
        num_agents=num_agents,  # number of agents
        size=grid_config.size,  # size of the grid
        density=grid_config.density,  # obstacle density
        seed=grid_config.seed,
        max_episode_steps=grid_config.max_episode_steps,  # horizon
        obs_radius=grid_config.obs_radius,  # defines field of view
        observation_type=grid_config.observation_type,
        collision_system=grid_config.collision_system,
        on_target=grid_config.on_target,
        map=obstacles.tolist(),
        agents_xy=start_positions,
        targets_xy=target_positions,
    )


def grid_config_generator_factory_single_config(
    map_type,
    map_w,
    map_h,
    num_agents,
    obstacle_density,
    obs_radius,
    collision_system,
    on_target,
    min_dist,
    max_episode_steps=None,
    num_agents_generator=None,
):
    if map_type == "RandomGrid":
        assert map_h == map_w, (
            f"Expect height and width of random grid to be the same, "
            f"but got height {map_h} and width {map_w}"
        )

        if (min_dist is None) or (min_dist <= 2):

            def _grid_config_generator(
                seed,
                max_episode_steps=max_episode_steps,
                num_agents=num_agents,
                map_id=None,
            ):
                if num_agents_generator:
                    num_agents = num_agents_generator(map_id)
                return GridConfig(
                    num_agents=num_agents,  # number of agents
                    size=map_w,  # size of the grid
                    density=obstacle_density,  # obstacle density
                    seed=seed,  # set to None for random
                    # obstacles, agents and targets
                    # positions at each reset
                    max_episode_steps=max_episode_steps,  # horizon
                    obs_radius=obs_radius,  # defines field of view
                    observation_type="MAPF",
                    collision_system=collision_system,
                    on_target=on_target,
                )

        else:
            # We need a custom generator to enforce the min dist between
            # start and target pos for each agent
            def _grid_config_generator(
                seed,
                max_episode_steps=max_episode_steps,
                num_agents=num_agents,
                map_id=None,
            ):
                if num_agents_generator:
                    num_agents = num_agents_generator(map_id)
                return generate_random_grid_with_min_dist(
                    seed,
                    map_w=map_w,
                    num_agents=num_agents,
                    obstacle_density=obstacle_density,
                    obs_radius=obs_radius,
                    collision_system=collision_system,
                    on_target=on_target,
                    min_dist=min_dist,
                    max_episode_steps=max_episode_steps,
                )

    elif map_type == "Maze" or map_type == "maze":
        pass
    else:
        raise ValueError(f"Unsupported map type: {map_type}.")
    return _grid_config_generator


def _grid_config_generator_factory_mixed_config(
    map_types,
    num_agents,
    max_episode_steps,
    map_w_min,
    map_w_max,
    obstacle_density_min,
    obstacle_density_max,
    obs_radius,
    collision_system,
    go_straight_min,
    go_straight_max,
    num_wall_rows_min,
    num_wall_rows_max,
    num_wall_cols_min,
    num_wall_cols_max,
    wall_width_min,
    wall_width_max,
    wall_height_min,
    wall_height_max,
    side_pad,
    horizontal_gap,
    vertical_gap,
    vertical_gap_min,
    vertical_gap_max,
    wfi_instance,
    block_extra_space,
    room_width_min,
    room_width_max,
    room_height_min,
    room_height_max,
    num_rows_min,
    num_rows_max,
    num_cols_min,
    num_cols_max,
    uniform,
    room_only_centre_obstacles,
    regulate_obstacle_density_max,
    min_dist,
    max_dist,
    map_dir,
    num_maps,
    on_target,
):
    map_types = map_types.split("+")
    map_types = [mt.split("=") for mt in map_types]
    map_types, map_type_probs = tuple(zip(*map_types))

    map_type_probs = np.array(map_type_probs, dtype=float)
    assert np.sum(map_type_probs) == 1, "Probabilities should sum to 1"

    number_agents = num_agents.split("+")
    number_agents = [na.split("=") for na in number_agents]
    if len(number_agents[0]) == 1:
        num_agents_probs = np.ones(len(number_agents), dtype=float) / len(number_agents)
        number_agents = np.array(number_agents, dtype=int).squeeze()
    else:
        number_agents, num_agents_probs = tuple(zip(*number_agents))
        number_agents = np.array(number_agents, dtype=int)
        num_agents_probs = np.array(num_agents_probs, dtype=float)
        # assert np.sum(num_agents_probs) == 1, "Probabilities should sum to 1"

    def _grid_config_generator(
        seed,
        max_episode_steps=max_episode_steps,
        num_agents=None,
        map_id=None,
    ):
        rng = np.random.default_rng(seed)
        map_type = rng.choice(map_types, p=map_type_probs)

        if map_type == "random":
            return generate_random_grid_config(
                size_min=map_w_min,
                size_max=map_w_max,
                obstacle_density_min=obstacle_density_min,
                obstacle_density_max=obstacle_density_max,
                regulate_obstacle_density_max=regulate_obstacle_density_max,
                num_agents=number_agents,
                probs=num_agents_probs,
                max_episode_steps=max_episode_steps,
                obs_radius=obs_radius,
                collision_system=collision_system,
                on_target=on_target,
                seed=seed,
            )
        elif map_type == "force_random":
            return generate_force_random_grid_config(
                size_min=map_w_min,
                size_max=map_w_max,
                obstacle_density_min=obstacle_density_min,
                obstacle_density_max=obstacle_density_max,
                regulate_obstacle_density_max=regulate_obstacle_density_max,
                num_agents=number_agents,
                probs=num_agents_probs,
                max_episode_steps=max_episode_steps,
                obs_radius=obs_radius,
                collision_system=collision_system,
                on_target=on_target,
                seed=seed,
            )
        elif map_type == "maze":
            return generate_maze_grid_config(
                size_min=map_w_min,
                size_max=map_w_max,
                obstacle_density_min=obstacle_density_min,
                obstacle_density_max=obstacle_density_max,
                go_straight_min=go_straight_min,
                go_straight_max=go_straight_max,
                regulate_obstacle_density_max=regulate_obstacle_density_max,
                num_agents=number_agents,
                probs=num_agents_probs,
                max_episode_steps=max_episode_steps,
                obs_radius=obs_radius,
                collision_system=collision_system,
                on_target=on_target,
                seed=seed,
            )
        elif map_type == "warehouse":
            return generate_warehouse_grid_config(
                size_min=map_w_min,
                size_max=map_w_max,
                num_wall_rows_min=num_wall_rows_min,
                num_wall_rows_max=num_wall_rows_max,
                num_wall_cols_min=num_wall_cols_min,
                num_wall_cols_max=num_wall_cols_max,
                wall_width_min=wall_width_min,
                wall_width_max=wall_width_max,
                wall_height_min=wall_height_min,
                wall_height_max=wall_height_max,
                side_pad=side_pad,
                horizontal_gap=horizontal_gap,
                vertical_gap=vertical_gap,
                vertical_gap_min=vertical_gap_min,
                vertical_gap_max=vertical_gap_max,
                wfi_instance=wfi_instance,
                block_extra_space=block_extra_space,
                num_agents=number_agents,
                probs=num_agents_probs,
                max_episode_steps=max_episode_steps,
                obs_radius=obs_radius,
                collision_system=collision_system,
                on_target=on_target,
                seed=seed,
            )
        elif map_type == "room":
            return generate_room_grid_config(
                room_width_min=room_width_min,
                room_width_max=room_width_max,
                room_height_min=room_height_min,
                room_height_max=room_height_max,
                num_rows_min=num_rows_min,
                num_rows_max=num_rows_max,
                num_cols_min=num_cols_min,
                num_cols_max=num_cols_max,
                obstacle_density_min=obstacle_density_min,
                obstacle_density_max=obstacle_density_max,
                uniform=uniform,
                only_centre_obstacles=room_only_centre_obstacles,
                num_agents=number_agents,
                probs=num_agents_probs,
                max_episode_steps=max_episode_steps,
                obs_radius=obs_radius,
                collision_system=collision_system,
                on_target=on_target,
                seed=seed,
            )
        elif map_type == "pregen":
            return generate_predefined_map_grid_config(
                map_dir=map_dir,
                num_maps=num_maps,
                num_agents=number_agents,
                probs=num_agents_probs,
                max_episode_steps=max_episode_steps,
                obs_radius=obs_radius,
                collision_system=collision_system,
                on_target=on_target,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported map type: {map_type}.")

    if min_dist is None:
        return _grid_config_generator

    def _gcgen(seed, max_episode_steps=max_episode_steps, num_agents=None, map_id=None):
        grid_config = _grid_config_generator(
            seed, max_episode_steps, num_agents, map_id
        )
        return generate_min_dist_grid_config(
            grid_config=grid_config, min_dist=min_dist, max_dist=max_dist, num_tries=5
        )

    return _gcgen


def grid_config_generator_factory_mixed_config(
    map_types,
    num_agents,
    max_episode_steps,
    map_w_min,
    map_w_max,
    obstacle_density_min,
    obstacle_density_max,
    obs_radius,
    collision_system,
    go_straight_min,
    go_straight_max,
    num_wall_rows_min,
    num_wall_rows_max,
    num_wall_cols_min,
    num_wall_cols_max,
    wall_width_min,
    wall_width_max,
    wall_height_min,
    wall_height_max,
    side_pad,
    horizontal_gap,
    vertical_gap,
    vertical_gap_min,
    vertical_gap_max,
    wfi_instance,
    block_extra_space,
    room_width_min,
    room_width_max,
    room_height_min,
    room_height_max,
    num_rows_min,
    num_rows_max,
    num_cols_min,
    num_cols_max,
    uniform,
    room_only_centre_obstacles,
    regulate_obstacle_density_max,
    min_dist,
    max_dist,
    map_dir,
    num_maps,
    on_target,
    ensure_grid_config_is_generatable=False,
):
    grid_config_generator = _grid_config_generator_factory_mixed_config(
        map_types=map_types,
        num_agents=num_agents,
        max_episode_steps=max_episode_steps,
        map_w_min=map_w_min,
        map_w_max=map_w_max,
        obstacle_density_min=obstacle_density_min,
        obstacle_density_max=obstacle_density_max,
        obs_radius=obs_radius,
        collision_system=collision_system,
        go_straight_min=go_straight_min,
        go_straight_max=go_straight_max,
        num_wall_rows_min=num_wall_rows_min,
        num_wall_rows_max=num_wall_rows_max,
        num_wall_cols_min=num_wall_cols_min,
        num_wall_cols_max=num_wall_cols_max,
        wall_width_min=wall_width_min,
        wall_width_max=wall_width_max,
        wall_height_min=wall_height_min,
        wall_height_max=wall_height_max,
        side_pad=side_pad,
        horizontal_gap=horizontal_gap,
        vertical_gap=vertical_gap,
        vertical_gap_min=vertical_gap_min,
        vertical_gap_max=vertical_gap_max,
        wfi_instance=wfi_instance,
        block_extra_space=block_extra_space,
        room_width_min=room_width_min,
        room_width_max=room_width_max,
        room_height_min=room_height_min,
        room_height_max=room_height_max,
        num_rows_min=num_rows_min,
        num_rows_max=num_rows_max,
        num_cols_min=num_cols_min,
        num_cols_max=num_cols_max,
        uniform=uniform,
        room_only_centre_obstacles=room_only_centre_obstacles,
        regulate_obstacle_density_max=regulate_obstacle_density_max,
        min_dist=min_dist,
        max_dist=max_dist,
        map_dir=map_dir,
        num_maps=num_maps,
        on_target=on_target,
    )
    number_agents = num_agents
    if ensure_grid_config_is_generatable:

        def _ensure_valid_grid_config(
            seed,
            max_episode_steps=max_episode_steps,
            num_agents=None,
            map_id=None,
        ):
            grid_config = grid_config_generator(
                seed, max_episode_steps, num_agents, map_id
            )
            try:
                env = pogema_v0(grid_config=grid_config)
                env.reset()
            except:
                # Lowering max obstacle density
                grid_config = _grid_config_generator_factory_mixed_config(
                    map_types=map_types,
                    num_agents=number_agents,
                    max_episode_steps=max_episode_steps,
                    map_w_min=map_w_min,
                    map_w_max=map_w_max,
                    obstacle_density_min=obstacle_density_min,
                    obstacle_density_max=0.4,
                    obs_radius=obs_radius,
                    collision_system=collision_system,
                    go_straight_min=go_straight_min,
                    go_straight_max=go_straight_max,
                    num_wall_rows_min=num_wall_rows_min,
                    num_wall_rows_max=num_wall_rows_min,
                    num_wall_cols_min=num_wall_cols_min,
                    num_wall_cols_max=num_wall_cols_min,
                    wall_width_min=wall_width_min,
                    wall_width_max=wall_width_min,
                    wall_height_min=wall_height_min,
                    wall_height_max=wall_height_min,
                    side_pad=side_pad,
                    horizontal_gap=horizontal_gap,
                    vertical_gap=vertical_gap,
                    vertical_gap_min=vertical_gap_min,
                    vertical_gap_max=vertical_gap_max,
                    wfi_instance=wfi_instance,
                    block_extra_space=block_extra_space,
                    room_width_min=room_width_min,
                    room_width_max=room_width_max,
                    room_height_min=room_height_min,
                    room_height_max=room_height_max,
                    num_rows_min=num_rows_min,
                    num_rows_max=num_rows_max,
                    num_cols_min=num_cols_min,
                    num_cols_max=num_cols_max,
                    uniform=uniform,
                    room_only_centre_obstacles=room_only_centre_obstacles,
                    regulate_obstacle_density_max=regulate_obstacle_density_max,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    map_dir=map_dir,
                    num_maps=num_maps,
                    on_target=on_target,
                )(seed, max_episode_steps, num_agents, map_id)
            return grid_config

        return _ensure_valid_grid_config
    return grid_config_generator


def grid_config_generator_factory(args, testing=False, num_agents_generator=None):
    if testing:
        map_type = args.test_map_type
    else:
        map_type = args.map_type
    if map_type == "mixed":
        if testing:
            return grid_config_generator_factory_mixed_config(
                map_types=args.test_map_types,
                num_agents=args.test_num_agents,
                max_episode_steps=args.test_max_episode_steps,
                map_w_min=args.test_map_w_min,
                map_w_max=args.test_map_w_max,
                obstacle_density_min=args.test_obstacle_density_min,
                obstacle_density_max=args.test_obstacle_density_max,
                obs_radius=args.test_obs_radius,
                collision_system=args.collision_system,
                go_straight_min=args.test_go_straight_min,
                go_straight_max=args.test_go_straight_max,
                num_wall_rows_min=args.test_num_wall_rows_min,
                num_wall_rows_max=args.test_num_wall_rows_max,
                num_wall_cols_min=args.test_num_wall_cols_min,
                num_wall_cols_max=args.test_num_wall_cols_max,
                wall_width_min=args.test_wall_width_min,
                wall_width_max=args.test_wall_width_max,
                wall_height_min=args.test_wall_height_min,
                wall_height_max=args.test_wall_height_max,
                side_pad=args.test_side_pad,
                horizontal_gap=args.test_horizontal_gap,
                vertical_gap=args.test_vertical_gap,
                vertical_gap_min=args.test_vertical_gap_min,
                vertical_gap_max=args.test_vertical_gap_max,
                wfi_instance=args.test_wfi_instance,
                block_extra_space=args.test_block_extra_space,
                room_width_min=args.test_room_width_min,
                room_width_max=args.test_room_width_max,
                room_height_min=args.test_room_height_min,
                room_height_max=args.test_room_height_max,
                num_rows_min=args.test_num_rows_min,
                num_rows_max=args.test_num_rows_max,
                num_cols_min=args.test_num_cols_min,
                num_cols_max=args.test_num_cols_max,
                uniform=args.test_room_grid_uniform,
                room_only_centre_obstacles=args.test_room_only_centre_obstacles,
                regulate_obstacle_density_max=args.test_regulate_obstacle_density_max,
                min_dist=args.test_min_dist,
                max_dist=args.test_max_dist,
                map_dir=args.test_map_dir,
                num_maps=args.test_num_maps,
                on_target=args.test_on_target,
                ensure_grid_config_is_generatable=args.test_ensure_grid_config_is_generatable,
            )
        else:
            return grid_config_generator_factory_mixed_config(
                map_types=args.map_types,
                num_agents=args.num_agents,
                max_episode_steps=args.max_episode_steps,
                map_w_min=args.map_w_min,
                map_w_max=args.map_w_max,
                obstacle_density_min=args.obstacle_density_min,
                obstacle_density_max=args.obstacle_density_max,
                obs_radius=args.obs_radius,
                collision_system=args.collision_system,
                go_straight_min=args.go_straight_min,
                go_straight_max=args.go_straight_max,
                num_wall_rows_min=args.num_wall_rows_min,
                num_wall_rows_max=args.num_wall_rows_max,
                num_wall_cols_min=args.num_wall_cols_min,
                num_wall_cols_max=args.num_wall_cols_max,
                wall_width_min=args.wall_width_min,
                wall_width_max=args.wall_width_max,
                wall_height_min=args.wall_height_min,
                wall_height_max=args.wall_height_max,
                side_pad=args.side_pad,
                horizontal_gap=args.horizontal_gap,
                vertical_gap=args.vertical_gap,
                vertical_gap_min=args.vertical_gap_min,
                vertical_gap_max=args.vertical_gap_max,
                wfi_instance=args.wfi_instance,
                block_extra_space=args.block_extra_space,
                room_width_min=args.room_width_min,
                room_width_max=args.room_width_max,
                room_height_min=args.room_height_min,
                room_height_max=args.room_height_max,
                num_rows_min=args.num_rows_min,
                num_rows_max=args.num_rows_max,
                num_cols_min=args.num_cols_min,
                num_cols_max=args.num_cols_max,
                uniform=args.room_grid_uniform,
                room_only_centre_obstacles=args.room_only_centre_obstacles,
                regulate_obstacle_density_max=args.regulate_obstacle_density_max,
                min_dist=args.min_dist,
                max_dist=args.max_dist,
                map_dir=args.map_dir,
                num_maps=args.num_maps,
                on_target=args.on_target,
                ensure_grid_config_is_generatable=args.ensure_grid_config_is_generatable,
            )
    else:
        if num_agents_generator is None:
            if testing:
                num_agents = int(
                    args.test_robot_density * args.test_map_h * args.test_map_w
                )
            else:
                num_agents = int(args.robot_density * args.map_h * args.map_w)

        if testing:
            return grid_config_generator_factory_single_config(
                map_type=args.test_map_type,
                map_w=args.test_map_w,
                map_h=args.test_map_h,
                num_agents=num_agents,
                obstacle_density=args.test_obstacle_density,
                obs_radius=args.test_obs_radius,
                collision_system=args.collision_system,
                on_target=args.test_on_target,
                min_dist=args.test_min_dist,
                max_episode_steps=args.test_max_episode_steps,
                num_agents_generator=num_agents_generator,
            )
        else:
            return grid_config_generator_factory_single_config(
                map_type=args.map_type,
                map_w=args.map_w,
                map_h=args.map_h,
                num_agents=num_agents,
                obstacle_density=args.obstacle_density,
                obs_radius=args.obs_radius,
                collision_system=args.collision_system,
                on_target=args.on_target,
                min_dist=args.min_dist,
                max_episode_steps=args.max_episode_steps,
                num_agents_generator=num_agents_generator,
            )
