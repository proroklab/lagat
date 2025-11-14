from typing import Optional, Literal
from pydantic import BaseModel
from pydantic import Extra
from pogema_toolbox.algorithm_config import AlgoBase
import torch
import numpy as np
from dcc.model import Network
from dcc.heuristic_map import HeuristicMapGenerator

class DCCInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['DCC'] = 'DCC'
    path_to_weights: str = "dcc/weights/128000.pth"
    parallel_backend: Literal[
        'multiprocessing', 'dask', 'sequential', 'balanced_multiprocessing', 'balanced_dask',
        'balanced_dask_gpu'] = 'balanced_dask'
    device: str = 'cpu'
    num_process: int = 8
    
    
class DCCInference:
    def __init__(self, cfg: DCCInferenceConfig):
        self.cfg = cfg
        self.agent = None
        self.last_actions = None
        self.positions = None
        self.h_map = None
        
    def _get_heuristic_observations(self, observations):
        agents_positions = [obs['global_xy'] for obs in observations]
        num_agents = len(observations)
        obs_size = len(observations[0]['obstacles'])//2
        h_obs = np.zeros((num_agents, 4, obs_size*2+1, obs_size*2+1),dtype=bool,)
        for i, agent_pos in enumerate(agents_positions):
            x, y = agent_pos
            h_obs[i] = self.h_map.heuristic_map[i, :, x - obs_size : x + obs_size + 1, y - obs_size : y + obs_size + 1]

        return h_obs
        
    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):
        num_agents = len(observations)
        if self.last_actions is None:
            self.last_actions = np.zeros((num_agents, 5))
        if self.h_map is None:
            self.h_map = HeuristicMapGenerator(observations[0]['global_obstacles'], 
                                               num_agents, 
                                               [obs['global_target_xy'] for obs in observations], 
                                               len(observations[0]['obstacles'])//2)
        self.positions = np.array([obs['global_xy'] for obs in observations])
        agents = [[obs['agents']] for obs in observations]
        for a in agents:
            a[0][len(a[0])//2, len(a[0])//2] = 0
        dcc_observations = np.concatenate((np.array(agents),
                                          np.array([[obs['obstacles']] for obs in observations]),
                                          self._get_heuristic_observations(observations)), axis=1)
        actions, *_ = self.agent.step(
            torch.as_tensor(dcc_observations.astype(np.float32)).to(self.cfg.device),
            torch.as_tensor(self.last_actions.astype(np.float32)).to(self.cfg.device),
            torch.as_tensor(self.positions.astype(int)).to(self.cfg.device),
        )
        self.last_actions = np.zeros((num_agents, 5))
        self.last_actions[np.arange(num_agents), np.array(actions)] = 1
        actions = [(a + 1)%5 for a in actions]  # mapping from DCC actions to pogema ones
        return actions

    def reset_states(self):
        self.last_actions = None
        self.positions = None
        self.h_map = None
        self.agent = Network()
        state_dict = torch.load(self.cfg.path_to_weights, map_location=self.cfg.device)
        self.agent.load_state_dict(state_dict)
        self.agent.to(self.cfg.device)
        self.agent.eval()