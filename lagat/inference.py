import numpy as np
import pathlib
import ctypes

from typing import Literal, Optional
from pogema_toolbox.algorithm_config import AlgoBase

from pogema import GridConfig

import os
import subprocess

lib_path = os.path.join(os.path.dirname(__file__), "build", "liblagat.so")
lib_path = pathlib.Path(lib_path)
lib_path.parent.mkdir(parents=True, exist_ok=True)

if not os.path.exists(lib_path):
    calling_script_dir = os.path.dirname(lib_path.parent)
    cmake_cmd = [
        "cmake",
        "-DCMAKE_BUILD_TYPE=Debug" "-DCAFFE2_USE_CUDNN=True",
        "-DCMAKE_C_COMPILER=/usr/bin/gcc-11",
        "-DCMAKE_CXX_COMPILER=/usr/bin/g++-11",
        "-B",
        "build",
    ]
    make_cmd = [
        "make",
        "-C",
        "build",
        "-j4",
    ]
    cmd_env = os.environ.copy()
    cmd_env["Torch_DIR"] = "/workspace/libtorch"

    subprocess.run(cmake_cmd, check=True, cwd=calling_script_dir, env=cmd_env)
    subprocess.run(make_cmd, check=True, cwd=calling_script_dir, env=cmd_env)

DEFAULT_TMP = os.path.join(os.path.dirname(__file__), "tmp")


class LaGATInferenceConfig(AlgoBase):
    name: Literal["LaGAT"] = "LaGAT"
    time_limit: float = 30
    tmp_dir: str = DEFAULT_TMP
    model_path: Optional[str] = "models/lagat.pt"
    args: list[str] = []
    enable_max_timesteps: bool = False


class LaGATLib:
    LIBRARY = None
    MODEL_LOADED = False
    MODEL_LOAD_PATH = None
    MODEL = None

    def __init__(self, config: LaGATInferenceConfig):
        self.config = config
        tmp_dir = config.tmp_dir
        self.input_map = os.path.abspath(os.path.join(tmp_dir, "input.map"))
        self.input_scen = os.path.abspath(os.path.join(tmp_dir, "input.scen"))
        self.output_file = os.path.abspath(os.path.join(tmp_dir, "output.txt"))
        tmp_dir = pathlib.Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self.num_agents = None
        self.max_timesteps = None

        if LaGATLib.LIBRARY is None:
            self.load_library()
        self.load_model()

    def load_library(self):
        LaGATLib.LIBRARY = ctypes.CDLL(lib_path)

        LaGATLib.LIBRARY.load_model.argtypes = [ctypes.c_char_p]  # model_path
        LaGATLib.LIBRARY.load_model.restype = ctypes.c_void_p

        LaGATLib.LIBRARY.free_model.argtypes = [ctypes.c_void_p]  # model pointer
        LaGATLib.LIBRARY.free_model.restype = ctypes.c_int

        LaGATLib.LIBRARY.run_lacam.argtypes = [
            ctypes.c_int,  # argc
            ctypes.POINTER(ctypes.c_char_p),  # argv
            ctypes.c_void_p,  # model pointer
        ]
        LaGATLib.LIBRARY.run_lacam.restype = ctypes.c_int

    def load_model(self):
        if LaGATLib.MODEL_LOADED:
            if self.config.model_path == LaGATLib.MODEL_LOAD_PATH:
                return 0
            LaGATLib.LIBRARY.free_model(LaGATLib.MODEL)
        if self.config.model_path is not None:
            print("Loaded Model")
            model_path = self.config.model_path.encode("utf-8")
            LaGATLib.MODEL = LaGATLib.LIBRARY.load_model(model_path)
            LaGATLib.MODEL_LOADED = True
            LaGATLib.MODEL_LOAD_PATH = self.config.model_path

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
            cur_str = f"{agent_id}\tinput.map\t{width}\t{height}\t{start[1]}\t{start[0]}\t{goal[1]}\t{goal[0]}\t1\n"
            # cur_str = f"{agent_id}       input.map  {width}      {height}      {start[1]}       {start[0]}      {goal[1]}       {goal[0]}      1"
            out_str += cur_str + "\n"
        with open(self.input_scen, "w") as f:
            f.write(out_str)

        if self.config.enable_max_timesteps:
            self.max_timesteps = env.grid_config.max_episode_steps

    def parse_output(self):
        if not os.path.exists(self.output_file):
            return None

        with open(self.output_file, "r") as f:
            output_data = f.readlines()

        offset = 0
        for line in output_data:
            offset += 1
            if line.startswith("solution"):
                break

        columns = None
        for line in output_data[offset:]:
            line = line.strip()
            line = line.split(":")[1]

            tuples = line.split("),")
            tuples = [t.strip("\n") for t in tuples]
            tuples = [t.strip() for t in tuples]
            if len(tuples[-1]) == 0:
                tuples = tuples[:-1]
            tuples = [tuple(map(int, t[1:].split(","))) for t in tuples]

            if columns is None:
                columns = [[] for _ in range(len(tuples))]

            for i, t in enumerate(tuples):
                columns[i].append(t[::-1])

        if columns is None:
            return None

        return np.array(columns)

    def run_lagat(self, env):
        self.prepare_input(env)

        calling_script_dir = lib_path.parent
        lacam_command = [
            "./main",
            "-m",
            self.input_map,
            "-i",
            self.input_scen,
            "-o",
            self.output_file,
            "-t",
            str(self.config.time_limit),
            "-N",
            str(self.num_agents),
        ]
        if self.config.model_path is not None:
            lacam_command = lacam_command + [
                "--model",
                self.config.model_path,
            ]
        if self.max_timesteps is not None:
            lacam_command = lacam_command + [
                "--enable_max_timesteps",
                "--max_timesteps",
                str(self.max_timesteps),
            ]
        lacam_command = lacam_command + self.config.args

        ret_code = self.run_library(lacam_command)

        return self.parse_output()

    def run_library(self, args):
        argv = (ctypes.c_char_p * len(args))(*[arg.encode("utf-8") for arg in args])
        argc = len(args)

        return LaGATLib.LIBRARY.run_lacam(argc, argv, LaGATLib.MODEL)


class LaGATInference:
    def __init__(self, config: LaGATInferenceConfig, env=None):
        self.config = config
        self.lacam_lib = LaGATLib(config)
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
                self.output_data = self.lacam_lib.run_lagat(self.env)
                if self.output_data is None:
                    self.timed_out = True
                    return [0] * self.env.grid_config.num_agents
            else:
                # If timed out, then just waiting (maybe change to something else?)
                return [0] * self.env.grid_config.num_agents
        actions = self._get_next_move(self.step)
        self.step += 1
        return actions
