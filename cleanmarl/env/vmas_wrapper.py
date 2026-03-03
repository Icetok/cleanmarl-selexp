# env/vmas_wrapper.py

import numpy as np
import torch
import vmas


class VMASWrapper:
    """
    Wraps a VMAS environment to match the interface expected by the MADDPG script.

    Key responsibilities:
      - Convert VMAS torch outputs -> numpy
      - Provide unified methods: reset, step, get_state, get_avail_actions
      - Optionally compute a simple "semantic score" (interaction/reward),
        but NOT advantage (advantage is computed in maddpg.py).
      - Provide render(mode="rgb_array") for evaluation videos (best-effort).
    """

    def __init__(
        self,
        scenario="discovery",
        n_agents=3,
        max_steps=100,
        agent_ids=True,
        device="cpu",
        continuous_actions=False,
        # wrapper semantic knobs (only for interaction/reward)
        semantic_enabled=True,
        semantic_threshold=0.0,
        semantic_mode="interaction",  # "interaction" | "reward"
        interaction_radius=0.5,
        interaction_use_inverse_mean_dist=True,
        **kwargs,
    ):
        self.scenario = scenario
        self.max_steps = max_steps
        self.agent_ids = agent_ids
        self.device = device

        # semantic configuration
        self.semantic_enabled = bool(semantic_enabled)
        self.semantic_threshold = float(semantic_threshold)
        self.semantic_mode = semantic_mode
        self.interaction_radius = float(interaction_radius)
        self.interaction_use_inverse_mean_dist = bool(interaction_use_inverse_mean_dist)

        self._last_semantic_score = 0.0
        self._last_semantic_keep = True

        # Create the VMAS environment
        self.env = vmas.make_env(
            scenario=scenario,
            num_envs=1,
            device=device,
            continuous_actions=continuous_actions,
            dict_spaces=False,
            max_steps=max_steps,
            **kwargs,
        )

        self.t = 0

        # Reset once to infer dimensions
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = self._to_np_obs(obs)

        self.n_agents = obs.shape[0]
        self._base_obs_size = obs.shape[-1]

        act_space0 = self.env.action_space[0]
        self._act_size = act_space0.n if hasattr(act_space0, "n") else act_space0.shape[-1]

        self._obs_size = self._base_obs_size + (self.n_agents if self.agent_ids else 0)

        obs0 = obs
        if self.agent_ids:
            obs0 = self._append_agent_ids(obs0)
        self._last_obs = obs0

    def get_obs_size(self):
        return self._obs_size

    def get_action_size(self):
        return self._act_size

    def get_state_size(self):
        return self.n_agents * self.get_obs_size()

    def get_last_semantic(self):
        return self._last_semantic_score, self._last_semantic_keep

    def reset(self):
        self.t = 0
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = self._to_np_obs(obs)

        self.n_agents = obs.shape[0]
        self._obs_size = self._base_obs_size + (self.n_agents if self.agent_ids else 0)

        if self.agent_ids:
            obs = self._append_agent_ids(obs)

        self._last_obs = obs
        self._last_semantic_score = 0.0
        self._last_semantic_keep = True
        return obs, {}

    def step(self, actions):
        self.t += 1

        obs, rew, done, info = self.env.step(self._format_actions(actions))
        obs = self._to_np_obs(obs)

        self.n_agents = obs.shape[0]
        self._obs_size = self._base_obs_size + (self.n_agents if self.agent_ids else 0)

        r_vec = self._to_np_rew(rew)
        r = float(np.mean(r_vec))

        if self.semantic_enabled:
            s_score = self._compute_semantic_score(obs_raw=obs, reward_scalar=r)
            s_keep = (s_score >= self.semantic_threshold)
        else:
            s_score = 0.0
            s_keep = True

        self._last_semantic_score = float(s_score)
        self._last_semantic_keep = bool(s_keep)

        if self.agent_ids:
            obs = self._append_agent_ids(obs)
        self._last_obs = obs

        terminated = bool(done.item()) if hasattr(done, "item") else bool(done)
        truncated = False
        if self.max_steps is not None:
            truncated = self.t >= self.max_steps

        infos = self._to_info_dict(info)
        infos["semantic_score"] = self._last_semantic_score
        infos["semantic_keep"] = self._last_semantic_keep
        infos["semantic_threshold"] = self.semantic_threshold
        infos["semantic_mode"] = self.semantic_mode

        return obs, r, terminated, truncated, infos

    def get_state(self):
        return self._last_obs.reshape(-1)

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.get_action_size()), dtype=bool)

    def render(self, mode="rgb_array"):
        """
        Best-effort rendering.
        For eval videos we prefer mode="rgb_array" which should return HxWx3 uint8.
        If VMAS backend doesn't support it, returns None.
        """
        # Try gym-like signature
        try:
            frame = self.env.render(mode=mode)
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            return frame
        except TypeError:
            pass
        except Exception:
            return None

        # Try no-arg render
        try:
            frame = self.env.render()
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            return frame
        except Exception:
            return None

    def close(self):
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except TypeError:
                pass

    # -----------------------
    # Wrapper semantic computation (interaction/reward)
    # -----------------------
    def _compute_semantic_score(self, obs_raw: np.ndarray, reward_scalar: float) -> float:
        if not self.semantic_enabled:
            return 0.0

        mode = (self.semantic_mode or "interaction").lower().strip()

        if mode == "reward":
            return float(abs(reward_scalar))

        if obs_raw.ndim != 2 or obs_raw.shape[1] < 2:
            return float(abs(reward_scalar))

        pos = obs_raw[:, :2].astype(np.float32, copy=False)

        diffs = pos[:, None, :] - pos[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)

        n = dists.shape[0]
        if n <= 1:
            return 0.0

        mask = ~np.eye(n, dtype=bool)

        if self.interaction_use_inverse_mean_dist:
            mean_dist = float(np.mean(dists[mask]))
            return float(1.0 / (mean_dist + 1e-6))

        rad = self.interaction_radius
        neighbors = (dists < rad) & mask
        count = np.sum(neighbors, axis=1)
        return float(np.mean(count))

    # -----------------------
    # Conversions / helpers
    # -----------------------
    def _to_np_obs(self, obs):
        if isinstance(obs, list):
            obs = torch.stack(obs, dim=0)
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu()
        obs = obs.numpy()

        if obs.ndim == 3:
            if obs.shape[1] == 1:
                obs = obs[:, 0, :]
            elif obs.shape[0] == 1:
                obs = obs[0, :, :]
            else:
                raise ValueError(f"Unexpected VMAS obs shape: {obs.shape}")

        self._last_obs = obs
        return obs

    def _to_np_rew(self, rew):
        if isinstance(rew, list):
            rew = torch.stack(rew, dim=0)
        if isinstance(rew, torch.Tensor):
            rew = rew.detach().cpu()
        rew = rew.numpy()

        if rew.ndim == 2 and rew.shape[1] == 1:
            rew = rew[:, 0]
        if rew.ndim == 3:
            if rew.shape[1] == 1:
                rew = rew[:, 0, :]
            elif rew.shape[0] == 1:
                rew = rew[0, :, :]
            rew = np.squeeze(rew)

        return rew

    def _append_agent_ids(self, obs):
        n = obs.shape[0]
        ids = np.eye(n, dtype=np.float32)
        return np.concatenate([obs, ids], axis=-1)

    def _format_actions(self, actions):
        """
        VMAS expects per-agent tensors with batch dimension (1, ...).

        - Discrete: actions is (N,) ints -> returns list of (1,1) int tensors
        - Continuous: actions is (N, act_dim) floats -> returns list of (1, act_dim) float tensors
        """
        a = np.asarray(actions)

        # Continuous case: (N, act_dim)
        if a.ndim == 2:
            return [torch.tensor(a[i:i + 1], device=self.device, dtype=torch.float32) for i in range(self.n_agents)]

        # Discrete case: scalar or (N,)
        if a.ndim == 0:
            a = np.full((self.n_agents,), a, dtype=np.int64)
        if a.ndim == 1:
            a = a[:, None]
        return [torch.tensor(a[i:i + 1], device=self.device, dtype=torch.int64) for i in range(self.n_agents)]

    def _to_info_dict(self, info):
        if isinstance(info, dict):
            out = {}
            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.detach().cpu().numpy()
                else:
                    out[k] = v
            return out
        return {}