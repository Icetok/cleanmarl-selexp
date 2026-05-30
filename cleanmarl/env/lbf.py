import numpy as np
from .common_interface import CommonInterface

import lbforaging  # needed so Gymnasium registers LBF envs
import gymnasium as gym
from gymnasium.spaces import flatdim


class LBFWrapper(CommonInterface):
    def __init__(
        self,
        map_name,
        reward_aggr="sum",
        seed=0,
        time_limit=150,
        agent_ids=False,
        **kwargs,
    ):
        super().__init__()

        self.env = gym.make(map_name, max_episode_steps=time_limit, **kwargs)

        self.agent_ids = bool(agent_ids)
        self.reward_aggr = reward_aggr
        self.episode_limit = int(time_limit)
        self.current_step = 0

        self.n_agents = int(self.env.unwrapped.n_agents)
        self.agents = list(range(self.n_agents))

        self._base_obs_size = int(flatdim(self.env.observation_space[0]))
        self._obs_size = self._base_obs_size + (self.n_agents if self.agent_ids else 0)

        self._action_size = max(space.n for space in self.env.action_space)

        self.state = np.zeros((self.n_agents * self._base_obs_size,), dtype=np.float32)

        obs, _ = self.env.reset(seed=seed)
        self.process_obs(obs)

    def step(self, actions):
        actions = np.asarray(actions).reshape(-1)

        if actions.shape[0] != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {actions.shape[0]}")

        actions = np.clip(actions, 0, self._action_size - 1)
        actions = [int(a) for a in actions]

        obs, rewards, terminated, truncated, info = self.env.step(actions)
        self.current_step += 1

        obs = self.process_obs(obs)

        reward_vec = np.asarray(rewards, dtype=np.float32).reshape(-1)

        if self.reward_aggr == "sum":
            reward = float(np.sum(reward_vec))
        elif self.reward_aggr == "mean":
            reward = float(np.mean(reward_vec))
        else:
            raise ValueError(f"Unsupported reward_aggr: {self.reward_aggr}")

        info = dict(info) if isinstance(info, dict) else {}
        info["reward_vec"] = reward_vec
        info["reward_team"] = reward
        info["reward_agents"] = reward_vec

        return obs, np.float32(reward), bool(terminated), bool(truncated), info

    def reset(self, seed=None):
        self.current_step = 0
        if seed is None:
            obs, info = self.env.reset()
        else:
            obs, info = self.env.reset(seed=seed)

        obs = self.process_obs(obs)
        return obs, info if isinstance(info, dict) else {}

    def get_obs_size(self):
        return self._obs_size

    def get_state_size(self):
        return self.n_agents * self._base_obs_size

    def get_state(self):
        return self.state.astype(np.float32, copy=False)

    def get_action_size(self):
        return self._action_size

    def get_avail_actions(self):
        return np.ones((self.n_agents, self._action_size), dtype=bool)

    def get_avail_agent_actions(self, agent_id):
        return np.ones((self._action_size,), dtype=bool)

    def sample(self):
        return np.asarray(self.env.action_space.sample(), dtype=np.int64)

    def process_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)

        if obs.ndim != 2:
            obs = np.stack(obs).astype(np.float32)

        self.state = obs.reshape(-1).astype(np.float32, copy=False)

        if self.agent_ids:
            ids = np.eye(self.n_agents, dtype=np.float32)
            obs = np.concatenate([obs, ids], axis=1)

        return obs.astype(np.float32, copy=False)

    def close(self):
        self.env.close()