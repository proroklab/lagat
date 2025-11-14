import numpy as np

class HeuristicMapGenerator:
    def __init__(self, map, num_agents, targets, obs_radius):
        self.map = np.array(map).astype(int)
        self.num_agents = num_agents
        self.targets = targets
        self.obs_radius = obs_radius
        self.heuristic_map = None
        self._get_heuristic_map()

    def _get_heuristic_map(self):
        map_size = self.map.shape
        dist_map = self._initialize_dist_map(map_size)
        empty_pos = self._get_empty_positions()

        for i in range(self.num_agents):
            self._compute_distances(i, dist_map, empty_pos)

        self.heuristic_map = self._initialize_heuristic_map(map_size)
        self._populate_heuristic_map(dist_map, empty_pos, map_size)
        #self._pad_heuristic_map()

    def _initialize_dist_map(self, map_size):
        return (
            np.ones((self.num_agents, *map_size), dtype=np.int32)
            * np.iinfo(np.int32).max
        )

    def _get_empty_positions(self):
        empty_pos = np.argwhere(self.map == 0)
        empty_pos = [
            (x, y) for x, y in empty_pos
            if self.obs_radius <= x < self.map.shape[0] - self.obs_radius
            and self.obs_radius <= y < self.map.shape[1] - self.obs_radius
        ]
        return set(empty_pos)

    def _compute_distances(self, agent_index, dist_map, empty_pos):
        open_list = set()
        x, y = tuple(self.targets[agent_index])
        open_list.add((x, y))
        dist_map[agent_index, x, y] = 0

        while open_list:
            x, y = open_list.pop()
            dist = dist_map[agent_index, x, y]

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in empty_pos and dist_map[agent_index, nx, ny] > dist + 1:
                    dist_map[agent_index, nx, ny] = dist + 1
                    open_list.add((nx, ny))

    def _initialize_heuristic_map(self, map_size):
        return np.zeros((self.num_agents, 4, *map_size), dtype=bool)

    def _populate_heuristic_map(self, dist_map, empty_pos, map_size):
        for x, y in empty_pos:
            for i in range(self.num_agents):
                if x > 0 and dist_map[i, x - 1, y] < dist_map[i, x, y]:
                    self.heuristic_map[i, 0, x, y] = 1
                if x < map_size[0] - 1 and dist_map[i, x + 1, y] < dist_map[i, x, y]:
                    self.heuristic_map[i, 1, x, y] = 1
                if y > 0 and dist_map[i, x, y - 1] < dist_map[i, x, y]:
                    self.heuristic_map[i, 2, x, y] = 1
                if y < map_size[1] - 1 and dist_map[i, x, y + 1] < dist_map[i, x, y]:
                    self.heuristic_map[i, 3, x, y] = 1

    def _pad_heuristic_map(self):
        self.heuristic_map = np.pad(
            self.heuristic_map,
            (
                (0, 0),
                (0, 0),
                (self.obs_radius, self.obs_radius),
                (self.obs_radius, self.obs_radius),
            ),
        )