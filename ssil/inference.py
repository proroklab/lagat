import numpy as np
import pathlib

from typing import Literal
from pogema_toolbox.algorithm_config import AlgoBase

from ssil.cost_to_go_generator import CostToGoCalculator

from pogema import GridConfig

import os
import subprocess

import ruamel.yaml

yaml = ruamel.yaml.YAML()

lib_path = os.path.dirname(__file__)

DEFAULT_TMP = os.path.join(os.path.dirname(__file__), "tmp")


class SSILInferenceConfig(AlgoBase):
    name: Literal["SSIL"] = "SSIL"
    tmp_dir: str = DEFAULT_TMP
    max_steps: int = 1000


class SSILLib:
    def __init__(self, config: SSILInferenceConfig):
        self.config = config
        tmp_dir = config.tmp_dir
        self.input_file = os.path.abspath(os.path.join(tmp_dir, "input.yaml"))
        self.input_cost_to_gos = os.path.abspath(os.path.join(tmp_dir, "input_bds.npz"))
        self.output_file = os.path.abspath(os.path.join(tmp_dir, "output.yaml"))
        tmp_dir = pathlib.Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

    def prepare_input(self, env):
        start_locs = env.grid.get_agents_xy(ignore_borders=True)
        target_locs = env.grid.get_targets_xy(ignore_borders=True)
        obstacle_locs = np.stack(
            np.nonzero(env.grid.get_obstacles(ignore_borders=True))
        ).T

        input_data = {"agents": [], "map": {"dimensions": [], "obstacles": []}}
        for agent_id, (start, goal) in enumerate(zip(start_locs, target_locs)):
            s, g = ruamel.yaml.comments.CommentedSeq(
                start
            ), ruamel.yaml.comments.CommentedSeq(goal)
            s.fa.set_flow_style()
            g.fa.set_flow_style()
            input_data["agents"].append(
                {"start": s, "goal": g, "name": f"agent{agent_id}"}
            )
        input_data["map"]["dimensions"] = ruamel.yaml.comments.CommentedSeq(
            [env.grid_config.size, env.grid_config.size]
        )
        input_data["map"]["dimensions"].fa.set_flow_style()
        for obstacle in obstacle_locs:
            o = ruamel.yaml.comments.CommentedSeq(obstacle.tolist())
            o.fa.set_flow_style()
            input_data["map"]["obstacles"].append(o)
        with open(self.input_file, "w") as f:
            yaml.dump(input_data, f)

        # Generating cost-to-gos
        c2g = CostToGoCalculator(env)
        cost_to_go_grid = c2g.generate_cost_to_go_grid()
        np.savez(self.input_cost_to_gos, single_map=cost_to_go_grid)

        return input_data

    def parse_output(self):
        with open(self.output_file, "r") as f:
            output_data = yaml.load(f)
        return output_data

    def run_ssil(self, env):
        self.prepare_input(env)

        calling_script_dir = lib_path
        ssil_command = [
            "conda",
            "activate",
            "mlmapf",
            "&&",
            "python",
            "custom_run_generator.py",
            "--input-yaml",
            self.input_file,
            "--bds-file",
            self.input_cost_to_gos,
            "--output-dir",
            self.config.tmp_dir,
            "--shieldType",
            "CS-PIBT",
            "--useGPU",
            "--maxSteps",
            str(self.config.max_steps),
        ]
        # ssil_command = [
        #     "conda activate mlmapf && python custom_run_generator.py --input-yaml {self.input_file} --bds-file {self.input_cost_to_gos} --output-dir {self.config.tmp_dir} --shieldType CS-PIBT --useGPU --maxSteps {self.config.max_steps}"
        # ]
        ssil_command = f'bash -c "source activate mlmapf; python custom_run_generator.py --input-yaml {self.input_file} --bds-file {self.input_cost_to_gos} --output-dir {self.config.tmp_dir} --shieldType CS-PIBT --useGPU --maxSteps {self.config.max_steps}"'

        try:
            subprocess.run(
                ssil_command,
                check=True,
                cwd=calling_script_dir,
                stdout=subprocess.DEVNULL,
                shell=True,
            )
        except subprocess.TimeoutExpired:
            return None

        return self.parse_output()


class SSILInference:
    def __init__(self, config: SSILInferenceConfig, env=None):
        self.config = config
        self.ssil_lib = SSILLib(config)
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
            self.ssil_lib.config.max_steps = env.grid_config.max_episode_steps

    def _get_pos_from_idx(self, agent_id, idx):
        return np.array(
            [
                self.output_data["schedule"][f"agent{agent_id}"][idx]["x"],
                self.output_data["schedule"][f"agent{agent_id}"][idx]["y"],
            ]
        )

    def _get_next_move_single_agent(self, agent_id, step):
        idx = 0
        for data in self.output_data["schedule"][f"agent{agent_id}"]:
            if data["t"] >= step:
                break
            idx += 1
        if idx == len(self.output_data["schedule"][f"agent{agent_id}"]):
            return 0
        if self.output_data["schedule"][f"agent{agent_id}"][idx]["t"] == step:
            new_pos = self._get_pos_from_idx(agent_id, idx)
            old_pos = self._get_pos_from_idx(agent_id, idx - 1)
            return np.nonzero(np.all(self.MOVES == (new_pos - old_pos), axis=-1))[0][0]
        else:
            return 0

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
                self.output_data = self.ssil_lib.run_ssil(self.env)
                if self.output_data is None:
                    self.timed_out = True
                    return [0] * self.env.grid_config.num_agents
            else:
                # If timed out, then just waiting (maybe change to something else?)
                return [0] * self.env.grid_config.num_agents
        actions = self._get_next_move(self.step)
        self.step += 1
        return actions
