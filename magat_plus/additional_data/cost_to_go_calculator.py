from typing import Optional
from collections import deque
from dataclasses import dataclass, field
import numpy as np

from pibt.pypibt.mapf_utils import Coord, Grid, get_neighbors, is_valid_coord


@dataclass
class DistTable:
    grid: Grid
    goal: Coord
    Q: deque = field(init=False)  # lazy distance evaluation
    table: np.ndarray = field(init=False)  # distance matrix

    def __post_init__(self):
        self.Q = deque([self.goal])
        self.table = np.full(self.grid.shape, self.grid.size, dtype=int)
        self.table[self.goal] = 0

    def get(self, target: Coord, invalid_result_value: Optional[int] = None) -> int:
        # check valid input
        if not is_valid_coord(self.grid, target):
            if invalid_result_value is None:
                return self.grid.size
            return invalid_result_value

        # distance has been known
        if self.table[target] < self.table.size:
            return self.table[target]

        # BFS with lazy evaluation
        while len(self.Q) > 0:
            u = self.Q.popleft()
            d = int(self.table[u])
            for v in get_neighbors(self.grid, u):
                if d + 1 < self.table[v]:
                    self.table[v] = d + 1
                    self.Q.append(v)
            if u == target:
                return d

        return self.grid.size


class CostToGoCalculator:
    def __init__(
        self,
        env,
        obs_radius,
        dtype="float32",
        pad_cost_to_go=True,
        clamp_value=None,
        clamp_values_doubled=False,
    ):
        obstacles = env.grid.get_obstacles(ignore_borders=True)
        obstacles = obstacles == 0

        goals = env.grid.get_targets_xy(ignore_borders=True)
        goals = [tuple(g) for g in goals]

        self.dist_tables = [DistTable(obstacles, goal) for goal in goals]
        self.obs_radius = obs_radius
        self.dtype = dtype

        self.feature_size = 2 * obs_radius + 1
        self.pad_cost_to_go = pad_cost_to_go

        self.clamp_value = clamp_value
        self.clamp_values_doubled = clamp_values_doubled

    def generate_single_agent_cost_to_go(self, agent_id, position):
        cost_to_go_grid = np.zeros(
            (self.feature_size, self.feature_size), dtype=self.dtype
        )

        position_x, position_y = position

        base_value = self.dist_tables[agent_id].get((position_x, position_y))
        invalid_result_value = base_value + 2 * self.obs_radius
        for i in range(self.feature_size):
            for j in range(self.feature_size):
                cost_to_go_grid[i, j] = (
                    self.dist_tables[agent_id].get(
                        (
                            position_x + i - self.obs_radius,
                            position_y + j - self.obs_radius,
                        ),
                        invalid_result_value,
                    )
                    - base_value
                )
        if self.pad_cost_to_go:
            cost_to_go_grid = np.pad(cost_to_go_grid, pad_width=1)
        return cost_to_go_grid

    def generate_cost_to_go(self, env, normalized=True):
        if self.pad_cost_to_go:
            cost_to_go = np.zeros(
                (env.num_agents, self.feature_size + 2, self.feature_size + 2),
                dtype=self.dtype,
            )
        else:
            cost_to_go = np.zeros(
                (env.num_agents, self.feature_size, self.feature_size), dtype=self.dtype
            )
        agent_pos = env.grid.get_agents_xy(ignore_borders=True)
        for agent_id, pos in enumerate(agent_pos):
            cost_to_go[agent_id] = self.generate_single_agent_cost_to_go(agent_id, pos)
        if normalized:
            cost_to_go = cost_to_go / (2 * self.obs_radius)
        if self.clamp_value is not None:
            if self.clamp_values_doubled:
                mask = cost_to_go > self.clamp_value
                cost_to_go[mask] = 2 * self.clamp_value
                mask = cost_to_go < -self.clamp_value
                cost_to_go[mask] = -2 * self.clamp_value
            else:
                cost_to_go = np.clip(cost_to_go, -self.clamp_value, self.clamp_value)
        return cost_to_go


def get_greedy_actions(cost_to_go, move_results, obs_radius, dtype="int"):
    idx = np.expand_dims(obs_radius + move_results, axis=0)
    idx = np.broadcast_to(idx, (cost_to_go.shape[0], *idx.shape[1:]))

    rows, cols = idx[..., 0], idx[..., 1]

    result = cost_to_go[np.arange(cost_to_go.shape[0])[:, None], rows, cols]

    return (result == np.min(result, axis=-1, keepdims=True)).astype(dtype)
