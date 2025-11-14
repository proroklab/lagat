import numpy as np
import pathlib

from typing import Literal, Optional
from pogema_toolbox.algorithm_config import AlgoBase

from pogema import GridConfig

import os
import subprocess

lib_path = os.path.join(os.path.dirname(__file__), "build", "lns")
lib_path = pathlib.Path(lib_path)
lib_path.parent.mkdir(parents=True, exist_ok=True)

if not os.path.exists(lib_path):
    calling_script_dir = os.path.dirname(lib_path.parent)
    cmake_cmd = [
        "cmake",
        "-DCMAKE_C_COMPILER=/usr/bin/gcc-11",
        "-DCMAKE_CXX_COMPILER=/usr/bin/g++-11",
        "-DCMAKE_BUILD_TYPE=RELEASE",
        "-B",
        "build",
    ]
    make_cmd = [
        "make",
        "-C",
        "build",
        "-j4",
    ]

    subprocess.run(cmake_cmd, check=True, cwd=calling_script_dir)
    subprocess.run(make_cmd, check=True, cwd=calling_script_dir)

DEFAULT_TMP = os.path.join(os.path.dirname(__file__), "tmp")


class LNS2InferenceConfig(AlgoBase):
    name: Literal["LNS2"] = "LNS2"
    time_limit: float = 30
    tmp_dir: str = DEFAULT_TMP
    args: list[str] = []


class LNS2Lib:
    def __init__(self, config: LNS2InferenceConfig):
        self.config = config
        tmp_dir = config.tmp_dir
        self.input_map = os.path.abspath(os.path.join(tmp_dir, "input.map"))
        self.input_scen = os.path.abspath(os.path.join(tmp_dir, "input.scen"))
        self.output_file = os.path.abspath(os.path.join(tmp_dir, "output.txt"))
        tmp_dir = pathlib.Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self.num_agents = None

    def prepare_input(self, env):
        # Generating the map
        obstacles = np.array(env.grid.get_obstacles(ignore_borders=True))
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

        with open(self.input_map, "w") as f:
            f.write(out_str)

        # Generating the scenario
        start_locs = env.grid.get_agents_xy(ignore_borders=True)
        target_locs = env.grid.get_targets_xy(ignore_borders=True)

        self.num_agents = len(start_locs)

        out_str = "version 1\n"
        for agent_id, (start, goal) in enumerate(zip(start_locs, target_locs)):
            cur_str = f"{agent_id}\tinput.map\t{width}\t{height}\t{start[1]}\t{start[0]}\t{goal[1]}\t{goal[0]}\t1"
            out_str += cur_str + "\n"
        with open(self.input_scen, "w") as f:
            f.write(out_str)

    def parse_output(self):
        if not os.path.exists(self.output_file):
            return None

        with open(self.output_file, "r") as f:
            output_data = f.readlines()

        offset = 0
        columns = []
        for line in output_data[offset:]:
            line = line.strip()
            line = line.split(":")[1]

            tuples = line.split(")->")
            tuples = [t.strip("\n") for t in tuples]
            tuples = [t.strip() for t in tuples]
            if len(tuples[-1]) == 0:
                tuples = tuples[:-1]
            tuples = [tuple(map(int, t[1:].split(","))) for t in tuples]

            columns.append(tuples)

        if len(columns) == 0:
            return None

        max_len = max(len(col) for col in columns)
        for i in range(len(columns)):
            if len(columns[i]) < max_len:
                columns[i] += [columns[i][-1]] * (max_len - len(columns[i]))

        return np.array(columns)

    def run_lns2(self, env):
        self.prepare_input(env)

        calling_script_dir = lib_path.parent
        lns2_command = [
            "./lns",
            "-m",
            self.input_map,
            "-a",
            self.input_scen,
            f"--outputPaths={self.output_file}",
            "-t",
            str(self.config.time_limit),
            "-k",
            str(self.num_agents),
        ]
        lns2_command = lns2_command + self.config.args

        try:
            subprocess.run(
                lns2_command,
                check=True,
                cwd=calling_script_dir,
                stdout=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return None

        return self.parse_output()


class LNS2Inference:
    def __init__(self, config: LNS2InferenceConfig, env=None):
        self.config = config
        self.lns2_lib = LNS2Lib(config)
        self.output_data = None
        self.step = 1
        self.env = env
        if env is not None:
            self.MOVES = np.array(self.env.grid_config.MOVES)
        self.timed_out = False

    def reset_states(self, env=None):
        self.step = 1
        self.timed_out = False
        if env is not None:
            self.env = env
            self.MOVES = np.array(self.env.grid_config.MOVES)

    def _get_next_move_single_agent(self, agent_id, step):
        agent_path = self.output_data[agent_id]
        if step >= len(agent_path):
            return 0

        old_pos = agent_path[step - 1]
        new_pos = agent_path[step]

        return np.nonzero(np.all(self.MOVES == (new_pos - old_pos), axis=-1))[0].item()

    def _get_next_move(self, step):
        return [
            self._get_next_move_single_agent(agent_id, step)
            for agent_id in range(self.env.grid_config.num_agents)
        ]

    def act(
        self, observations=None, rewards=None, dones=None, info=None, skip_agents=None
    ):
        if self.output_data is None:
            if not self.timed_out:
                self.output_data = self.lns2_lib.run_lns2(self.env)
                if self.output_data is None:
                    self.timed_out = True
                    return [0] * self.env.grid_config.num_agents
            else:
                # If timed out, then just waiting (maybe change to something else?)
                return [0] * self.env.grid_config.num_agents
        actions = self._get_next_move(self.step)
        self.step += 1
        return actions
