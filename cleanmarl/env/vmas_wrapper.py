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
        but NOT advantage (advantage is computed in maddpg_continuous.py).
      - Provide render(mode="rgb_array") for evaluation videos (best-effort).

    Notes / fixes:
      - Keep TWO versions of observations:
          * obs returned to the policy can include agent IDs (if agent_ids=True)
          * "state" for the critic should NOT include agent IDs (they are redundant + can hurt learning)
      - Reward handling: VMAS can return a per-agent reward vector; MADDPG typically uses a team reward.
        We therefore aggregate by SUM (stronger learning signal than mean in many VMAS scenarios).
      - Action formatting avoids extra tensor copies where possible (torch.as_tensor + views).
    """

    def __init__(
        self,
        scenario="discovery",
        n_agents=3,
        max_steps=200,
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
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.agent_ids = bool(agent_ids)
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
            max_steps=self.max_steps,
            **kwargs,
        )

        self.t = 0

        # Reset once to infer dimensions
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs_raw = self._to_np_obs(obs)  # (N, obs_dim) on CPU as numpy

        self.n_agents = int(obs_raw.shape[0])
        self._base_obs_size = int(obs_raw.shape[-1])

        act_space0 = self.env.action_space[0]
        self._act_size = int(act_space0.n) if hasattr(act_space0, "n") else int(act_space0.shape[-1])

        # obs returned to actor/policy may include agent IDs
        self._obs_size = self._base_obs_size + (self.n_agents if self.agent_ids else 0)

        # cache raw + augmented
        self._last_obs_raw = obs_raw
        obs0 = obs_raw
        if self.agent_ids:
            obs0 = self._append_agent_ids(obs0)
        self._last_obs = obs0

    def get_obs_size(self):
        return self._obs_size

    def get_action_size(self):
        return self._act_size

    def get_state_size(self):
        # IMPORTANT: critic "state" should use RAW obs (no agent IDs)
        return self.n_agents * self._base_obs_size

    def get_last_semantic(self):
        return self._last_semantic_score, self._last_semantic_keep

    def reset(self, seed=None):
        self.t = 0

        # VMAS reset signature differs by version; best-effort.
        try:
            obs = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        except TypeError:
            obs = self.env.reset()

        if isinstance(obs, tuple):
            obs = obs[0]

        obs_raw = self._to_np_obs(obs)  # (N, obs_dim)

        self.n_agents = int(obs_raw.shape[0])
        self._obs_size = self._base_obs_size + (self.n_agents if self.agent_ids else 0)

        # cache raw + augmented
        self._last_obs_raw = obs_raw
        obs_out = obs_raw
        if self.agent_ids:
            obs_out = self._append_agent_ids(obs_out)

        self._last_obs = obs_out
        self._last_semantic_score = 0.0
        self._last_semantic_keep = True
        return obs_out, {}

    def step(self, actions):
        self.t += 1

        obs, rew, done, info = self.env.step(self._format_actions(actions))

        obs_raw = self._to_np_obs(obs)  # (N, obs_dim)
        self.n_agents = int(obs_raw.shape[0])
        self._obs_size = self._base_obs_size + (self.n_agents if self.agent_ids else 0)

        # reward: VMAS often returns per-agent rewards. Aggregate to a team scalar.
        r_vec = self._to_np_rew(rew)
        r = self._aggregate_reward(r_vec)

        # semantic score computed on RAW obs (positions in first 2 dims)
        if self.semantic_enabled:
            s_score = self._compute_semantic_score(obs_raw=obs_raw, reward_scalar=r)
            s_keep = s_score >= self.semantic_threshold
        else:
            s_score = 0.0
            s_keep = True

        self._last_semantic_score = float(s_score)
        self._last_semantic_keep = bool(s_keep)

        # cache raw + augmented
        self._last_obs_raw = obs_raw
        obs_out = obs_raw
        if self.agent_ids:
            obs_out = self._append_agent_ids(obs_out)
        self._last_obs = obs_out

        terminated = bool(done.item()) if hasattr(done, "item") else bool(done)

        truncated = False
        if self.max_steps is not None:
            truncated = self.t >= self.max_steps

        infos = self._to_info_dict(info)
        infos["semantic_score"] = self._last_semantic_score
        infos["semantic_keep"] = self._last_semantic_keep
        infos["semantic_threshold"] = float(self.semantic_threshold)
        infos["semantic_mode"] = self.semantic_mode
        infos["reward_vec"] = r_vec  # debug: helps verify reward wiring
        infos["reward_team"] = float(r)

        return obs_out, float(r), terminated, bool(truncated), infos

    def get_state(self):
        # IMPORTANT: critic should see RAW obs only (no agent IDs)
        return self._last_obs_raw.reshape(-1)

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.get_action_size()), dtype=bool)

    def render(self, mode="rgb_array"):
        """
        Best-effort rendering.
        For eval videos we prefer mode="rgb_array" which should return HxWx3 uint8.
        If VMAS backend doesn't support it, returns None.
        """
        try:
            frame = self.env.render(mode=mode)
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            return frame
        except TypeError:
            pass
        except Exception:
            return None

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
    # Reward aggregation
    # -----------------------
    def _aggregate_reward(self, r_vec):
        """
        VMAS rewards can be:
          - scalar
          - shape (N,) per-agent
          - shape (num_envs, N) or (N, num_envs) depending on version
        We want a TEAM scalar reward for MADDPG -> use sum over agents.
        """
        r_np = np.asarray(r_vec)

        # squeeze away singleton env dims
        r_np = np.squeeze(r_np)

        if r_np.ndim == 0:
            return float(r_np)

        # if vector length matches agents -> sum
        if r_np.ndim == 1:
            if r_np.shape[0] == self.n_agents:
                return float(np.sum(r_np))
            # otherwise, just sum everything
            return float(np.sum(r_np))

        # any higher dims -> sum all
        return float(np.sum(r_np))

    # -----------------------
    # Conversions / helpers
    # -----------------------
    def _to_np_obs(self, obs):
        """
        Returns obs as numpy array of shape (N, obs_dim).
        """
        if isinstance(obs, list):
            # list of (num_envs, obs_dim) tensors for each agent
            obs = torch.stack(obs, dim=0)  # (N, num_envs, obs_dim)

        if isinstance(obs, torch.Tensor):
            obs = obs.detach()

            # move to cpu only once here
            if obs.device.type != "cpu":
                obs = obs.cpu()

        obs = obs.numpy()

        # Normalize to (N, obs_dim)
        if obs.ndim == 3:
            # common cases:
            # (N, 1, obs_dim) or (1, N, obs_dim)
            if obs.shape[1] == 1:
                obs = obs[:, 0, :]
            elif obs.shape[0] == 1:
                obs = obs[0, :, :]
            else:
                raise ValueError(f"Unexpected VMAS obs shape: {obs.shape}")

        if obs.ndim != 2:
            raise ValueError(f"Unexpected VMAS obs ndim after squeeze: {obs.shape}")

        return obs

    def _to_np_rew(self, rew):
        if isinstance(rew, list):
            rew = torch.stack(rew, dim=0)

        if isinstance(rew, torch.Tensor):
            rew = rew.detach()
            if rew.device.type != "cpu":
                rew = rew.cpu()

        rew = np.asarray(rew)

        # squeeze common singleton dims
        rew = np.squeeze(rew)

        return rew

    def _append_agent_ids(self, obs):
        n = obs.shape[0]
        ids = np.eye(n, dtype=np.float32)
        return np.concatenate([obs, ids], axis=-1)

    def _format_actions(self, actions):
        """
        VMAS expects per-agent tensors with batch dimension (num_envs, ...), where num_envs=1.

        - Discrete: actions is (N,) ints -> list of (1,1) int64 tensors
        - Continuous: actions is (N, act_dim) floats -> list of (1, act_dim) float32 tensors

        This implementation avoids unnecessary copies by using torch.as_tensor + views.
        """
        a = np.asarray(actions)

        # Continuous case: (N, act_dim)
        if a.ndim == 2:
            a_t = torch.as_tensor(a, device=self.device, dtype=torch.float32)  # (N, act_dim)
            # return list of (1, act_dim) views
            return [a_t[i].unsqueeze(0) for i in range(self.n_agents)]

        # Discrete case: scalar or (N,)
        if a.ndim == 0:
            a = np.full((self.n_agents,), a, dtype=np.int64)
        if a.ndim == 1:
            a = a[:, None]  # (N,1)

        a_t = torch.as_tensor(a, device=self.device, dtype=torch.int64)  # (N,1)
        return [a_t[i].unsqueeze(0) for i in range(self.n_agents)]

    def _to_info_dict(self, info):
        if isinstance(info, dict):
            out = {}
            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    vv = v.detach()
                    if vv.device.type != "cpu":
                        vv = vv.cpu()
                    out[k] = vv.numpy()
                else:
                    out[k] = v
            return out
        return {}