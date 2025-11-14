import numpy as np
from additional_data.cost_to_go_calculator import DistTable


class CostToGoCalculator:
    def __init__(
        self,
        env,
        dtype="int64",
        obstacle_value=2**30 - 1,
    ):
        obstacles = env.grid.get_obstacles(ignore_borders=True)
        obstacles = obstacles == 0

        goals = env.grid.get_targets_xy(ignore_borders=True)
        goals = [tuple(g) for g in goals]

        self.dist_tables = [DistTable(obstacles, goal) for goal in goals]
        self.dtype = dtype

        self.obstacle_value = obstacle_value

    def generate_cost_to_go_grid(self):
        cost_to_go_grid = np.zeros(
            (len(self.dist_tables), *self.dist_tables[0].grid.shape),
            dtype=self.dtype,
        )

        for agent_idx, dist_table in enumerate(self.dist_tables):
            for i in range(cost_to_go_grid.shape[1]):
                for j in range(cost_to_go_grid.shape[2]):
                    cost_to_go_grid[agent_idx, i, j] = dist_table.get(
                        (i, j), invalid_result_value=self.obstacle_value
                    )
        return cost_to_go_grid
