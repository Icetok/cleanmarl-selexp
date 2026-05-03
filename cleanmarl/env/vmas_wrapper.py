import numpy as np
import torch
import vmas


class VMASWrapper:
    """
    Wraps a VMAS environment to match the interface expected by the training scripts.

    Supports:
      - parallel VMAS collection via num_envs
      - actor obs with optional agent IDs
      - critic state from RAW obs only
      - team reward aggregation
      - best-effort rgb rendering for num_envs=1
      - per-env reset_at support for vectorised envs
    """

    def __init__(
        self,
        scenario="balance",
        n_agents=4,  # kept for interface compatibility; VMAS determines actual n_agents from obs
        max_steps=100,
        num_envs=1,
        agent_ids=True,
        device="cpu",
        continuous_actions=False,
        semantic_enabled=True,
        semantic_threshold=0.0,
        semantic_mode="interaction",
        interaction_radius=0.5,
        interaction_use_inverse_mean_dist=True,
        **kwargs,
    ):
        self.scenario = scenario
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.num_envs = int(num_envs)
        self.agent_ids = bool(agent_ids)
        self.device = device

        self.semantic_enabled = bool(semantic_enabled)
        self.semantic_threshold = float(semantic_threshold)
        self.semantic_mode = semantic_mode
        self.interaction_radius = float(interaction_radius)
        self.interaction_use_inverse_mean_dist = bool(interaction_use_inverse_mean_dist)

        self._last_semantic_score = np.zeros((self.num_envs,), dtype=np.float32)
        self._last_semantic_keep = np.ones((self.num_envs,), dtype=bool)

        self.env = vmas.make_env(
            scenario=scenario,
            num_envs=self.num_envs,
            device=device,
            continuous_actions=continuous_actions,
            dict_spaces=False,
            max_steps=self.max_steps,
            **kwargs,
        )

        self.t = np.zeros((self.num_envs,), dtype=np.int32)

        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs_raw = self._to_np_obs(obs)

        self.n_agents = int(obs_raw.shape[1])
        self._base_obs_size = int(obs_raw.shape[-1])

        act_space0 = self.env.action_space[0]
        self._act_size = int(act_space0.n) if hasattr(act_space0, "n") else int(act_space0.shape[-1])

        self._obs_size = self._base_obs_size + (self.n_agents if self.agent_ids else 0)

        self._last_obs_raw = obs_raw
        obs0 = obs_raw
        if self.agent_ids:
            obs0 = self._append_agent_ids(obs0)
        self._last_obs = obs0

        if hasattr(act_space0, "low") and hasattr(act_space0, "high"):
            self.act_low = np.asarray(act_space0.low, dtype=np.float32)
            self.act_high = np.asarray(act_space0.high, dtype=np.float32)
        else:
            self.act_low = -np.ones((self._act_size,), dtype=np.float32)
            self.act_high = np.ones((self._act_size,), dtype=np.float32)

    def get_obs_size(self):
        return self._obs_size

    def get_action_size(self):
        return self._act_size

    def get_state_size(self):
        return self.n_agents * self._base_obs_size

    def get_last_semantic(self):
        return self._last_semantic_score, self._last_semantic_keep

    def reset(self, seed=None):
        self.t = np.zeros((self.num_envs,), dtype=np.int32)

        try:
            obs = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        except TypeError:
            obs = self.env.reset()

        if isinstance(obs, tuple):
            obs = obs[0]

        obs_raw = self._to_np_obs(obs)

        self.n_agents = int(obs_raw.shape[1])
        self._obs_size = self._base_obs_size + (self.n_agents if self.agent_ids else 0)

        self._last_obs_raw = obs_raw
        obs_out = obs_raw
        if self.agent_ids:
            obs_out = self._append_agent_ids(obs_out)

        self._last_obs = obs_out
        self._last_semantic_score = np.zeros((self.num_envs,), dtype=np.float32)
        self._last_semantic_keep = np.ones((self.num_envs,), dtype=bool)

        return obs_out, {}

    def step(self, actions):
        self.t += 1

        obs, rew, done, info = self.env.step(self._format_actions(actions))

        obs_raw = self._to_np_obs(obs)

        self.n_agents = int(obs_raw.shape[1])
        self._obs_size = self._base_obs_size + (self.n_agents if self.agent_ids else 0)

        r_vec = self._to_np_rew(rew)
        reward_team = self._aggregate_reward(r_vec)

        if self.semantic_enabled:
            s_score = self._compute_semantic_score(obs_raw=obs_raw, reward_scalar=reward_team)
            s_keep = s_score >= self.semantic_threshold
        else:
            s_score = np.zeros((self.num_envs,), dtype=np.float32)
            s_keep = np.ones((self.num_envs,), dtype=bool)

        self._last_semantic_score = np.asarray(s_score, dtype=np.float32).reshape(-1)
        self._last_semantic_keep = np.asarray(s_keep, dtype=bool).reshape(-1)

        self._last_obs_raw = obs_raw
        obs_out = obs_raw
        if self.agent_ids:
            obs_out = self._append_agent_ids(obs_out)
        self._last_obs = obs_out

        done_arr = self._to_done_np(done)
        terminated = done_arr.astype(bool)

        truncated = np.zeros((self.num_envs,), dtype=bool)
        if self.max_steps is not None:
            truncated = self.t >= self.max_steps

        infos = self._to_info_dict(info)
        infos["semantic_score"] = self._last_semantic_score
        infos["semantic_keep"] = self._last_semantic_keep
        infos["semantic_threshold"] = float(self.semantic_threshold)
        infos["semantic_mode"] = self.semantic_mode
        infos["reward_vec"] = r_vec
        infos["reward_team"] = reward_team

        return obs_out, reward_team.astype(np.float32), terminated, truncated, infos

    def reset_at(self, indices=None):
        if indices is None:
            return self.reset()

        indices = np.asarray(indices, dtype=np.int32).reshape(-1)

        if len(indices) == 0:
            return self._last_obs.copy(), {}

        for idx in indices:
            idx = int(idx)

            try:
                obs_idx = self.env.reset_at(idx)
            except AttributeError:
                obs_idx = self.env.reset()

            if isinstance(obs_idx, tuple):
                obs_idx = obs_idx[0]

            if isinstance(obs_idx, list):
                obs_idx = torch.stack(obs_idx, dim=0)

            if isinstance(obs_idx, torch.Tensor):
                obs_idx = obs_idx.detach().cpu().numpy()

            obs_idx = np.asarray(obs_idx)

            if obs_idx.ndim == 2 and obs_idx.shape[0] == self.n_agents:
                self._last_obs_raw[idx] = obs_idx.astype(np.float32, copy=False)
            elif obs_idx.ndim == 3 and obs_idx.shape[0] == 1 and obs_idx.shape[1] == self.n_agents:
                self._last_obs_raw[idx] = obs_idx[0].astype(np.float32, copy=False)
            elif obs_idx.ndim == 3 and obs_idx.shape[0] == self.num_envs and obs_idx.shape[1] == self.n_agents:
                self._last_obs_raw[idx] = obs_idx[idx].astype(np.float32, copy=False)
            else:
                obs_idx_raw = self._to_np_obs(obs_idx)
                if obs_idx_raw.shape[0] == 1:
                    self._last_obs_raw[idx] = obs_idx_raw[0]
                elif obs_idx_raw.shape[0] == self.num_envs:
                    self._last_obs_raw[idx] = obs_idx_raw[idx]

            self.t[idx] = 0
            self._last_semantic_score[idx] = 0.0
            self._last_semantic_keep[idx] = True

        obs_out = self._last_obs_raw
        if self.agent_ids:
            obs_out = self._append_agent_ids(obs_out)

        self._last_obs = obs_out
        return obs_out.copy(), {}

    def get_state(self):
        return self._last_obs_raw.reshape(self.num_envs, -1)

    def get_avail_actions(self):
        return np.ones((self.num_envs, self.n_agents, self.get_action_size()), dtype=bool)

    def render(self, mode="rgb_array"):
        if self.num_envs != 1:
            return None

        try:
            frame = self.env.render(mode=mode)

            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()

            if isinstance(frame, list) and len(frame) > 0:
                frame = frame[0]
            elif isinstance(frame, tuple) and len(frame) > 0:
                frame = frame[0]

            return frame

        except TypeError:
            pass

        except Exception as e:
            if not hasattr(self, "_render_warned"):
                self._render_warned = True
                print(f"[vmas_render] render(mode={mode}) failed: {type(e).__name__}: {e}")
            return None

        try:
            frame = self.env.render()

            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()

            if isinstance(frame, list) and len(frame) > 0:
                frame = frame[0]
            elif isinstance(frame, tuple) and len(frame) > 0:
                frame = frame[0]

            return frame

        except Exception:
            return None

    def close(self):
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except TypeError:
                pass

    def _compute_semantic_score(self, obs_raw: np.ndarray, reward_scalar) -> np.ndarray:
        if not self.semantic_enabled:
            return np.zeros((self.num_envs,), dtype=np.float32)

        mode = (self.semantic_mode or "interaction").lower().strip()

        reward_scalar = np.asarray(reward_scalar, dtype=np.float32).reshape(-1)

        if mode == "reward":
            return np.abs(reward_scalar)

        if obs_raw.ndim != 3 or obs_raw.shape[-1] < 2:
            return np.abs(reward_scalar)

        scores = np.zeros((obs_raw.shape[0],), dtype=np.float32)

        for e in range(obs_raw.shape[0]):
            pos = obs_raw[e, :, :2].astype(np.float32, copy=False)

            diffs = pos[:, None, :] - pos[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)

            n = dists.shape[0]
            if n <= 1:
                scores[e] = 0.0
                continue

            mask = ~np.eye(n, dtype=bool)

            if self.interaction_use_inverse_mean_dist:
                mean_dist = float(np.mean(dists[mask]))
                scores[e] = float(1.0 / (mean_dist + 1e-6))
            else:
                rad = self.interaction_radius
                neighbors = (dists < rad) & mask
                count = np.sum(neighbors, axis=1)
                scores[e] = float(np.mean(count))

        return scores

    def _aggregate_reward(self, r_vec):
        r_np = np.asarray(r_vec, dtype=np.float32)
        r_np = np.squeeze(r_np)

        if r_np.ndim == 0:
            return np.full((self.num_envs,), float(r_np), dtype=np.float32)

        if r_np.ndim == 1:
            if r_np.shape[0] == self.num_envs:
                return r_np.astype(np.float32)

            if r_np.shape[0] == self.n_agents and self.num_envs == 1:
                return np.array([float(np.sum(r_np))], dtype=np.float32)

            return np.full((self.num_envs,), float(np.sum(r_np)), dtype=np.float32)

        if r_np.ndim == 2:
            if r_np.shape[0] == self.num_envs and r_np.shape[1] == self.n_agents:
                return np.sum(r_np, axis=1).astype(np.float32)

            if r_np.shape[1] == self.num_envs and r_np.shape[0] == self.n_agents:
                return np.sum(r_np, axis=0).astype(np.float32)

        try:
            return np.sum(r_np, axis=tuple(range(1, r_np.ndim))).astype(np.float32)
        except Exception:
            return np.full((self.num_envs,), float(np.sum(r_np)), dtype=np.float32)

    def _to_np_obs(self, obs):
        if isinstance(obs, list):
            obs = torch.stack(obs, dim=0)

        if isinstance(obs, torch.Tensor):
            obs = obs.detach()
            if obs.device.type != "cpu":
                obs = obs.cpu()

        obs = np.asarray(obs)

        if obs.ndim == 3:
            if obs.shape[0] == self.n_agents if hasattr(self, "n_agents") else False:
                obs = np.transpose(obs, (1, 0, 2))
            elif obs.shape[1] == self.num_envs:
                obs = np.transpose(obs, (1, 0, 2))
            elif obs.shape[0] == self.num_envs:
                pass
            else:
                if obs.shape[0] < obs.shape[1]:
                    obs = np.transpose(obs, (1, 0, 2))

        elif obs.ndim == 2:
            obs = obs[None, :, :]

        else:
            raise ValueError(f"Unexpected VMAS obs shape after conversion: {obs.shape}")

        if obs.ndim != 3:
            raise ValueError(f"Unexpected VMAS obs ndim after squeeze: {obs.shape}")

        return obs.astype(np.float32, copy=False)

    def _to_np_rew(self, rew):
        if isinstance(rew, list):
            rew = torch.stack(rew, dim=0)

        if isinstance(rew, torch.Tensor):
            rew = rew.detach()
            if rew.device.type != "cpu":
                rew = rew.cpu()

        rew = np.asarray(rew)
        rew = np.squeeze(rew)

        return rew

    def _to_done_np(self, done):
        if isinstance(done, list):
            done = torch.stack(done, dim=0)

        if isinstance(done, torch.Tensor):
            done = done.detach()
            if done.device.type != "cpu":
                done = done.cpu()

        done = np.asarray(done)
        done = np.squeeze(done)

        if done.ndim == 0:
            return np.full((self.num_envs,), bool(done), dtype=bool)

        if done.ndim == 1:
            if done.shape[0] == self.num_envs:
                return done.astype(bool)

            if done.shape[0] == self.n_agents and self.num_envs == 1:
                return np.array([bool(np.any(done))], dtype=bool)

        if done.ndim == 2:
            if done.shape[0] == self.num_envs:
                return np.any(done, axis=1).astype(bool)

            if done.shape[1] == self.num_envs:
                return np.any(done, axis=0).astype(bool)

        return np.full((self.num_envs,), bool(np.any(done)), dtype=bool)

    def _append_agent_ids(self, obs):
        e, n, _ = obs.shape
        ids = np.eye(n, dtype=np.float32)[None, :, :]
        ids = np.repeat(ids, e, axis=0)

        return np.concatenate([obs, ids], axis=-1)

    def _format_actions(self, actions):
        a = np.asarray(actions)

        if a.ndim == 2:
            a = a[None, :, :]

        if a.ndim != 3:
            raise ValueError(f"Unexpected action shape for VMAS continuous actions: {a.shape}")

        a_t = torch.as_tensor(a, device=self.device, dtype=torch.float32)

        return [a_t[:, i, :] for i in range(self.n_agents)]

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