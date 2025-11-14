import torch

from scrimp.alg_parameters import *
from scrimp.episodic_buffer import EpisodicBuffer
from scrimp.mapf_gym import MAPFEnv
from scrimp.model import Model
from scrimp.util import reset_env
from scrimp.eval_model import one_step

from typing import Optional, Literal
from pydantic import BaseModel
from pydantic import Extra
from pogema_toolbox.algorithm_config import AlgoBase


class SCRIMPInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['SCRIMP'] = 'SCRIMP'
    path_to_weights: str = "scrimp/model"
    parallel_backend: Literal[
        'multiprocessing', 'dask', 'sequential', 'balanced_multiprocessing', 'balanced_dask',
        'balanced_dask_gpu'] = 'balanced_dask'


class SCRIMPInference:
    def __init__(self, cfg: SCRIMPInferenceConfig):
        self.cfg: SCRIMPInferenceConfig = cfg
        self.env = None
        self.buffer = None
        self.obs = None
        self.vector = None
        self.hidden_state = None
        self.message = None
        self.cur_xy = None
        path_checkpoint = self.cfg.path_to_weights + "/net_checkpoint.pkl"
        self.model = Model(0, torch.device(cfg.device))
        self.model.network.load_state_dict(torch.load(path_checkpoint, map_location=torch.device(cfg.device))['model'])
        self.model.network.eval()
        self.offsets = None
        self.global_obs = None

    def act(self, observations):
        num_agents = len(observations)
        cur_goals = self.env.get_goals()
        need_to_update = False
        goals = []
        for i in range(len(observations)):
            goal = (
            observations[i]['target_xy'][0] + self.offsets[i][0], observations[i]['target_xy'][1] + self.offsets[i][1])
            if cur_goals[i] != goal:
                need_to_update = True
            goals.append(goal)
        if need_to_update:
            self.env.set_state(self.global_obs, self.env.get_positions(), goals)
            _, _, self.obs, self.vector, _ = reset_env(self.env, num_agents)

        actions, self.hidden_state, v_all, ps, self.message = self.model.final_evaluate(self.obs, self.vector,
                                                                                        self.hidden_state, self.message,
                                                                                        num_agents, greedy=False)

        one_episode_perf = {'episode_len': 0, 'max_goals': 0, 'collide': 0, 'success_rate': 0}
        rewards, self.obs, self.vector, done, one_episode_perf, max_on_goals, on_goal = one_step(self.env, actions,
                                                                                                 self.model, v_all,
                                                                                                 self.hidden_state, ps,
                                                                                                 one_episode_perf,
                                                                                                 self.message,
                                                                                                 self.buffer)
        moves = {(0, 0): 0, (-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
        new_xy = self.env.get_positions()
        actions = []
        for a in range(num_agents):
            actions.append(moves[(new_xy[a][0] - self.cur_xy[a][0], new_xy[a][1] - self.cur_xy[a][1])])
        self.cur_xy = new_xy
        return actions

    def reset_states(self, env):
        torch.manual_seed(self.cfg.seed)
        path_checkpoint = self.cfg.path_to_weights + "/net_checkpoint.pkl"
        self.model = Model(0, torch.device(self.cfg.device))
        self.model.network.load_state_dict(torch.load(path_checkpoint, map_location=torch.device(self.cfg.device))['model'])

        starts = env.grid.get_agents_xy()
        goals = env.grid.get_targets_xy(ignore_borders=True)
        obs = env.grid.get_obstacles()
        num_agents = len(starts)

        self.env = MAPFEnv(num_agents=num_agents, size=64, prob=0.2)
        self.buffer = EpisodicBuffer(100, num_agents)
        self.offsets = starts
        self.env.set_state(obs, starts, goals)
        self.message = torch.zeros((1, num_agents, NetParameters.NET_SIZE)).to(torch.device(self.cfg.device))
        self.hidden_state = (torch.zeros((num_agents, NetParameters.NET_SIZE // 2)).to(torch.device(self.cfg.device)),
                                torch.zeros((num_agents, NetParameters.NET_SIZE // 2)).to(torch.device(self.cfg.device)))
        _, _, self.obs, self.vector, _ = reset_env(self.env, num_agents)
        self.cur_xy = self.env.get_positions()
        self.buffer.batch_add(self.cur_xy)
        self.global_obs = obs.copy()
