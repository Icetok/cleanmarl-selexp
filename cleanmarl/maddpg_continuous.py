import copy
from pathlib import Path
import datetime
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from env.vmas_wrapper import VMASWrapper

# Optional dependency for video writing.
try:
    import imageio.v2 as imageio

    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False

print("[boot] maddpg_continuous.py imported")


# -------------------------
# Command line arguments
# -------------------------
@dataclass
class Args:
    # Environment selection
    env_type: str = "vmas"
    env_name: str = "balance"
    env_family: str = "vmas"
    agent_ids: bool = True

    # VMAS parallel rollout
    vmas_num_envs: int = 32

    # -------------------------
    # VMAS balance config
    # -------------------------
    vmas_max_steps: int = 100
    vmas_n_agents: int = 4
    vmas_random_package_pos_on_line: bool = True
    vmas_package_mass: float = 5.0

    # -------------------------
    # VMAS discovery config
    # -------------------------
    vmas_n_targets: int = 7
    vmas_lidar_range: float = 0.35
    vmas_covering_range: float = 0.25
    vmas_agents_per_target: int = 2
    vmas_targets_respawn: bool = True
    vmas_shared_reward: bool = True
    vmas_agent_collision_penalty: float = 0.0

    # RL hyperparameters
    gamma: float = 0.99
    buffer_size: int = 500000
    batch_size: int = 256
    normalize_reward: bool = True

    # Actor network
    actor_hidden_dim: int = 128
    actor_num_layers: int = 2

    # Critic network
    critic_hidden_dim: int = 128
    critic_num_layers: int = 2

    # Training / logging
    learning_starts: int = 5000
    train_freq: int = 1
    updates_per_step: int = 4
    optimizer: str = "Adam"
    learning_rate_actor: float = 5e-5
    learning_rate_critic: float = 1e-4
    total_timesteps: int = 150000
    target_network_update_freq: int = 1
    polyak: float = 0.005
    clip_gradients: float = 1.0
    log_every: int = 10
    eval_steps: int = 5000
    num_eval_ep: int = 5

    # Exploration schedule
    exploration_noise_start: float = 0.3
    exploration_noise_end: float = 0.05
    exploration_anneal_frac: float = 0.5

    # Evaluation rendering / video
    eval_render: bool = False
    eval_save_video: bool = False
    eval_video_dir: str = "eval_videos"
    eval_video_fps: int = 20
    eval_video_format: str = "mp4"
    eval_video_max_frames: int = 2000

    # W&B / device / seed
    use_wnb: bool = False
    wnb_project: str = ""
    wnb_entity: str = ""
    device: str = "cuda"
    seed: int = 1

    # Semantic selection
    semantic_enabled: bool = False
    semantic_mode: str = "advantage"
    semantic_log_every: int = 50

    # soft discard / second-stage sampling
    semantic_sampling_enabled: bool = True
    semantic_candidate_multiplier: int = 4
    semantic_priority_fraction: float = 0.5

    # advantage gating
    adv_keep_frac: float = 0.2
    adv_alpha_ema_beta: float = 0.9
    adv_alpha_window: int = 5000
    adv_warmup_steps: int = 5000

    # clustering
    cluster_enabled: bool = False
    cluster_mode: str = "proximity"
    cluster_proximity_radius: float = 0.75
    cluster_policy_dist_thresh: float = 0.25
    cluster_keep_rule: str = "any"


# -------------------------
# Actor network (continuous)
# -------------------------
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layer):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

    def act(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# -------------------------
# Centralised critic network
# -------------------------
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim, num_agents) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layer):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, 1)))

    def forward(self, state, actions, grad_processing=False, batch_action=None):
        x = self.maddpg_inputs(state, actions, grad_processing, batch_action)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)  # (B, N)

    def maddpg_inputs(self, state, actions, grad_processing, batch_action):
        maddpg_inputs = torch.zeros((state.size(0), self.num_agents, self.input_dim), device=state.device)
        maddpg_inputs[:, :, : state.size(-1)] = state.unsqueeze(1)

        joint_actions = actions.unsqueeze(1).expand(-1, self.num_agents, -1, -1).reshape(
            state.size(0), self.num_agents, -1
        )

        if grad_processing:
            if batch_action is None:
                raise ValueError("batch_action must be provided when grad_processing=True")
            batch_joint = batch_action.unsqueeze(1).expand(-1, self.num_agents, -1, -1).reshape(
                state.size(0), self.num_agents, -1
            )
            mask = (
                torch.eye(self.num_agents, device=state.device)
                .unsqueeze(-1)
                .expand(-1, -1, actions.size(-1))
                .reshape(self.num_agents, -1)
            )
            joint_actions = torch.where(mask.bool(), joint_actions, batch_joint)

        maddpg_inputs[:, :, state.size(-1):] = joint_actions
        return maddpg_inputs


# -------------------------
# Step-based replay buffer
# -------------------------
class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        num_agents,
        obs_space,
        state_space,
        action_space,
        normalize_reward=False,
        device="cpu",
    ):
        self.buffer_size = int(buffer_size)
        self.num_agents = int(num_agents)
        self.obs_space = int(obs_space)
        self.state_space = int(state_space)
        self.action_space = int(action_space)
        self.normalize_reward = bool(normalize_reward)
        self.device = device

        self.obs = np.zeros((self.buffer_size, self.num_agents, self.obs_space), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.num_agents, self.action_space), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.next_obs = np.zeros((self.buffer_size, self.num_agents, self.obs_space), dtype=np.float32)
        self.states = np.zeros((self.buffer_size, self.state_space), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, self.state_space), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)

        self.keep_mask = np.ones((self.buffer_size,), dtype=bool)
        self.semantic_score = np.zeros((self.buffer_size,), dtype=np.float32)

        self.pos = 0
        self.size = 0

        self.reward_count = 0
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0

    def _update_reward_stats(self, rewards: np.ndarray):
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if rewards.size == 0:
            return
        self.reward_count += int(rewards.size)
        self.reward_sum += float(np.sum(rewards))
        self.reward_sq_sum += float(np.sum(rewards**2))

    @property
    def reward_mean(self) -> float:
        if self.reward_count == 0:
            return 0.0
        return self.reward_sum / self.reward_count

    @property
    def reward_std(self) -> float:
        if self.reward_count <= 1:
            return 1.0
        mean = self.reward_mean
        var = max(self.reward_sq_sum / self.reward_count - mean * mean, 1e-8)
        return float(np.sqrt(var))

    def store_batch(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        states,
        next_states,
        dones,
        keep_mask=None,
        semantic_score=None,
    ):
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1)

        num_transitions = rewards.shape[0]

        if keep_mask is None:
            keep_mask = np.ones((num_transitions,), dtype=bool)
        else:
            keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)

        if semantic_score is None:
            semantic_score = np.zeros((num_transitions,), dtype=np.float32)
        else:
            semantic_score = np.asarray(semantic_score, dtype=np.float32).reshape(-1)

        self._update_reward_stats(rewards)

        for i in range(num_transitions):
            self.obs[self.pos] = obs[i]
            self.actions[self.pos] = actions[i]
            self.rewards[self.pos] = rewards[i]
            self.next_obs[self.pos] = next_obs[i]
            self.states[self.pos] = states[i]
            self.next_states[self.pos] = next_states[i]
            self.dones[self.pos] = dones[i]
            self.keep_mask[self.pos] = keep_mask[i]
            self.semantic_score[self.pos] = semantic_score[i]

            self.pos = (self.pos + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)

    def sample(
        self,
        batch_size: int,
        semantic_sampling_enabled: bool = True,
        candidate_multiplier: int = 4,
        priority_fraction: float = 0.5,
    ):
        if self.size == 0:
            raise RuntimeError("ReplayBuffer is empty; cannot sample")

        batch_size = int(batch_size)
        candidate_size = min(self.size, max(batch_size, batch_size * int(candidate_multiplier)))
        candidate_idx = np.random.randint(0, self.size, size=candidate_size)

        if semantic_sampling_enabled:
            keep_candidates = candidate_idx[self.keep_mask[candidate_idx]]
            non_keep_candidates = candidate_idx[~self.keep_mask[candidate_idx]]

            n_keep = min(len(keep_candidates), int(round(batch_size * float(priority_fraction))))
            n_rand = batch_size - n_keep

            chosen = []
            if n_keep > 0:
                keep_pick = np.random.choice(keep_candidates, size=n_keep, replace=(len(keep_candidates) < n_keep))
                chosen.append(keep_pick)

            if len(non_keep_candidates) > 0 and n_rand > 0:
                rand_pick = np.random.choice(
                    non_keep_candidates,
                    size=n_rand,
                    replace=(len(non_keep_candidates) < n_rand),
                )
                chosen.append(rand_pick)
            elif n_rand > 0:
                fallback_pick = np.random.choice(candidate_idx, size=n_rand, replace=(len(candidate_idx) < n_rand))
                chosen.append(fallback_pick)

            idx = np.concatenate(chosen, axis=0)
            if idx.shape[0] < batch_size:
                extra = np.random.choice(candidate_idx, size=batch_size - idx.shape[0], replace=True)
                idx = np.concatenate([idx, extra], axis=0)
        else:
            idx = np.random.choice(candidate_idx, size=batch_size, replace=(len(candidate_idx) < batch_size))

        obs = torch.from_numpy(self.obs[idx]).float().to(self.device)
        actions = torch.from_numpy(self.actions[idx]).float().to(self.device)
        rewards = torch.from_numpy(self.rewards[idx]).float().to(self.device)
        next_obs = torch.from_numpy(self.next_obs[idx]).float().to(self.device)
        states = torch.from_numpy(self.states[idx]).float().to(self.device)
        next_states = torch.from_numpy(self.next_states[idx]).float().to(self.device)
        dones = torch.from_numpy(self.dones[idx]).float().to(self.device)
        keep_mask = torch.from_numpy(self.keep_mask[idx]).bool().to(self.device)
        semantic_score = torch.from_numpy(self.semantic_score[idx]).float().to(self.device)

        if self.normalize_reward:
            rewards = rewards / max(self.reward_std, 1e-6)

        return obs, actions, rewards, next_obs, states, next_states, dones, keep_mask, semantic_score


# -------------------------
# Environment factory
# -------------------------
def environment(env_type, env_name, env_family, agent_ids, kwargs):
    if env_type == "vmas":
        vmas_kwargs = dict(kwargs)

        device = vmas_kwargs.pop("device", "cpu")
        n_agents = vmas_kwargs.pop("n_agents", 3)
        max_steps = vmas_kwargs.pop("max_steps", 100)
        num_envs = vmas_kwargs.pop("num_envs", 1)

        semantic_enabled = vmas_kwargs.pop("wrapper_semantic_enabled", False)
        semantic_threshold = vmas_kwargs.pop("semantic_threshold", 0.0)
        semantic_mode = vmas_kwargs.pop("semantic_mode", "interaction")
        interaction_radius = vmas_kwargs.pop("interaction_radius", 0.5)
        interaction_use_inverse_mean_dist = vmas_kwargs.pop("interaction_use_inverse_mean_dist", True)

        return VMASWrapper(
            scenario=env_name,
            n_agents=n_agents,
            max_steps=max_steps,
            num_envs=num_envs,
            agent_ids=agent_ids,
            device=device,
            continuous_actions=True,
            semantic_enabled=semantic_enabled,
            semantic_threshold=semantic_threshold,
            semantic_mode=semantic_mode,
            interaction_radius=interaction_radius,
            interaction_use_inverse_mean_dist=interaction_use_inverse_mean_dist,
            **vmas_kwargs,
        )

    raise ValueError(f"Unknown env_type: {env_type}")


# -------------------------
# Utility
# -------------------------
def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads if g is not None]
    if len(norms) == 0:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.tensor(norms), d)


def soft_update(target_net, utility_net, polyak):
    for target_param, param in zip(target_net.parameters(), utility_net.parameters()):
        target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)


# -------------------------
# Semantic helpers
# -------------------------
def _strip_agent_ids(obs_with_ids: np.ndarray, n_agents: int, agent_ids: bool) -> np.ndarray:
    if not agent_ids:
        return obs_with_ids
    if obs_with_ids.ndim != 2:
        return obs_with_ids
    base_dim = obs_with_ids.shape[1] - n_agents
    if base_dim <= 0:
        return obs_with_ids
    return obs_with_ids[:, :base_dim]


def _pairwise_l2_distance(X: np.ndarray) -> np.ndarray:
    diffs = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diffs, axis=-1).astype(np.float32)


def _connected_components_from_adjacency(adj: np.ndarray) -> list[list[int]]:
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            nbrs = np.where(adj[u])[0]
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def _build_clusters_continuous(
    obs_base: np.ndarray,
    policy_vecs: np.ndarray,
    mode: str,
    prox_radius: float,
    policy_dist_thresh: float,
) -> tuple[list[list[int]], dict[int, int], dict[str, float]]:
    n = policy_vecs.shape[0]
    mode = (mode or "proximity").lower().strip()

    Dpol = _pairwise_l2_distance(policy_vecs)
    policy_edges = Dpol <= float(policy_dist_thresh)

    prox_edges = np.zeros((n, n), dtype=bool)
    if obs_base.ndim == 2 and obs_base.shape[1] >= 2:
        pos = obs_base[:, :2].astype(np.float32, copy=False)
        diffs = pos[:, None, :] - pos[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        prox_edges = dists <= float(prox_radius)

    if mode == "policy":
        adj = policy_edges
    elif mode == "hybrid":
        adj = policy_edges | prox_edges
    else:
        adj = prox_edges if prox_edges.any() else policy_edges

    adj = adj | np.eye(n, dtype=bool)

    clusters = _connected_components_from_adjacency(adj)

    agent_to_cluster = {}
    for cid, comp in enumerate(clusters):
        for a in comp:
            agent_to_cluster[a] = cid

    mask = ~np.eye(n, dtype=bool)
    mean_pairwise_policy_dist = float(np.mean(Dpol[mask])) if n > 1 else 0.0
    stats = {
        "mean_pairwise_policy_dist": mean_pairwise_policy_dist,
        "num_clusters": float(len(clusters)),
    }
    return clusters, agent_to_cluster, stats


def _shannon_entropy_of_cluster_sizes(clusters: list[list[int]], n_agents: int, eps: float = 1e-12) -> float:
    if n_agents <= 0 or len(clusters) == 0:
        return 0.0
    sizes = np.asarray([len(c) for c in clusters], dtype=np.float32)
    p = sizes / (float(n_agents) + eps)
    return -float(np.sum(p * np.log(p + eps)))


def _compute_advantage_per_agent_continuous(
    critic,
    actor,
    obs_np,
    state_np,
    actions_taken_np,
    device,
    action_scale,
    action_bias,
):
    obs_t = torch.from_numpy(obs_np).float().to(device)
    state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
    a_taken = torch.from_numpy(actions_taken_np).float().to(device).unsqueeze(0)

    with torch.no_grad():
        a_pi = action_scale * torch.tanh(actor.act(obs_t).unsqueeze(0)) + action_bias
        q_taken = critic(state_t, a_taken)
        q_pi = critic(state_t, a_pi)

        if q_taken.ndim == 1:
            q_taken = q_taken.unsqueeze(0)
        if q_pi.ndim == 1:
            q_pi = q_pi.unsqueeze(0)

        adv = (q_taken - q_pi).reshape(-1)
        abs_adv = torch.abs(adv).detach().cpu().numpy().astype(np.float32)

    return abs_adv


# -------------------------
# Video / schedules
# -------------------------
def _as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return None
    f = np.asarray(frame)
    if f.dtype == np.uint8:
        return f
    f = f.astype(np.float32)
    if f.max() <= 1.5:
        f = f * 255.0
    f = np.clip(f, 0.0, 255.0).astype(np.uint8)
    return f


def _maybe_write_video(frames, out_path: Path, fps: int, fmt: str):
    if not frames:
        return
    if not _HAS_IMAGEIO:
        print("[warn] imageio not available; skipping video save. (pip install imageio imageio-ffmpeg)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or "gif").lower().strip()

    cleaned = []
    for fr in frames:
        fr_u8 = _as_uint8_rgb(fr)
        if fr_u8 is not None:
            cleaned.append(fr_u8)
    if not cleaned:
        return

    try:
        if fmt == "mp4":
            imageio.mimsave(str(out_path.with_suffix(".mp4")), cleaned, fps=int(fps))
        else:
            imageio.mimsave(str(out_path.with_suffix(".gif")), cleaned, fps=int(fps))
    except Exception as e:
        print(f"[warn] failed to save video to {out_path}: {e}")


def _noise_schedule(step: int, total_steps: int, start: float, end: float, anneal_frac: float) -> float:
    anneal_steps = max(1, int(float(total_steps) * float(anneal_frac)))
    if step >= anneal_steps:
        return float(end)
    alpha = float(step) / float(anneal_steps)
    return float(start + alpha * (end - start))


def _build_vmas_kwargs(args: Args, num_envs: int) -> dict:
    wrapper_semantic_enabled = args.semantic_enabled and (args.semantic_mode in ("interaction", "reward"))

    kwargs = {
        "device": args.device,
        "num_envs": int(num_envs),
        "n_agents": int(args.vmas_n_agents),
        "max_steps": int(args.vmas_max_steps),
        "wrapper_semantic_enabled": wrapper_semantic_enabled,
        "semantic_mode": args.semantic_mode if wrapper_semantic_enabled else "interaction",
        "semantic_threshold": 0.0,
    }

    if args.env_name == "discovery":
        kwargs.update(
            {
                "n_targets": int(args.vmas_n_targets),
                "lidar_range": float(args.vmas_lidar_range),
                "covering_range": float(args.vmas_covering_range),
                "agents_per_target": int(args.vmas_agents_per_target),
                "targets_respawn": bool(args.vmas_targets_respawn),
                "shared_reward": bool(args.vmas_shared_reward),
            }
        )
        if args.vmas_agent_collision_penalty != 0.0:
            kwargs["agent_collision_penalty"] = float(args.vmas_agent_collision_penalty)

    elif args.env_name == "balance":
        kwargs.update(
            {
                "random_package_pos_on_line": bool(args.vmas_random_package_pos_on_line),
                "package_mass": float(args.vmas_package_mass),
            }
        )

    return kwargs


# -------------------------
# Single training update
# -------------------------
def _do_training_updates(
    args,
    rb,
    actor,
    critic,
    target_actor,
    target_critic,
    actor_optimizer,
    critic_optimizer,
    action_scale,
    action_bias,
    device,
    step,
    writer,
    num_updates,
):
    last_actor_loss = None
    last_critic_loss = None
    last_actor_grad = None
    last_critic_grad = None

    for _ in range(int(args.updates_per_step)):
        (
            batch_obs,
            batch_action,
            batch_reward,
            batch_next_obs,
            batch_states,
            batch_next_states,
            batch_done,
            batch_keep,
            batch_semantic_score,
        ) = rb.sample(
            batch_size=args.batch_size,
            semantic_sampling_enabled=(args.semantic_enabled and args.semantic_sampling_enabled),
            candidate_multiplier=args.semantic_candidate_multiplier,
            priority_fraction=args.semantic_priority_fraction,
        )

        with torch.no_grad():
            a_next = action_scale * torch.tanh(target_actor.act(batch_next_obs)) + action_bias
            q_next = target_critic(batch_next_states, a_next)
            q_next = torch.nan_to_num(q_next, nan=0.0)

            expanded_reward = batch_reward.unsqueeze(-1).expand(-1, critic.num_agents)
            expanded_done = batch_done.unsqueeze(-1).expand(-1, critic.num_agents)
            targets = expanded_reward + args.gamma * (1 - expanded_done) * q_next

        q_values = critic(batch_states, batch_action)
        critic_loss = F.mse_loss(q_values, targets)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_gradients = norm_d([p.grad for p in critic.parameters()], 2)
        if args.clip_gradients > 0:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.clip_gradients)
        critic_optimizer.step()

        a_pi = action_scale * torch.tanh(actor.act(batch_obs)) + action_bias
        qvals_pi = critic(batch_states, a_pi, grad_processing=True, batch_action=batch_action)
        actor_loss = -qvals_pi.mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_gradients = norm_d([p.grad for p in actor.parameters()], 2)
        if args.clip_gradients > 0:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.clip_gradients)
        actor_optimizer.step()

        num_updates += 1

        if num_updates % args.target_network_update_freq == 0:
            soft_update(target_net=target_actor, utility_net=actor, polyak=args.polyak)
            soft_update(target_net=target_critic, utility_net=critic, polyak=args.polyak)

        last_actor_loss = actor_loss
        last_critic_loss = critic_loss
        last_actor_grad = actor_gradients
        last_critic_grad = critic_gradients

    if last_actor_loss is not None:
        writer.add_scalar("train/critic_loss", float(last_critic_loss.detach().cpu().item()), step)
        writer.add_scalar("train/actor_loss", float(last_actor_loss.detach().cpu().item()), step)
        writer.add_scalar("train/actor_gradients", float(last_actor_grad.detach().cpu().item()), step)
        writer.add_scalar("train/critic_gradients", float(last_critic_grad.detach().cpu().item()), step)
        writer.add_scalar("train/num_updates", num_updates, step)
        writer.add_scalar("train/reward_std_running", float(rb.reward_std), step)

    return num_updates


# -------------------------
# Main training loop
# -------------------------
if __name__ == "__main__":
    print("[boot] entering main")
    args = tyro.cli(Args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device)

    kwargs = _build_vmas_kwargs(args, num_envs=args.vmas_num_envs if args.env_type == "vmas" else 1)
    eval_kwargs = _build_vmas_kwargs(args, num_envs=1 if args.env_type == "vmas" else 1)

    env = environment(args.env_type, args.env_name, args.env_family, args.agent_ids, kwargs)
    eval_env = environment(args.env_type, args.env_name, args.env_family, args.agent_ids, eval_kwargs)

    print(
        "[sanity] n_agents:",
        env.n_agents,
        "num_envs:",
        getattr(env, "num_envs", 1),
        "obs_size:",
        env.get_obs_size(),
        "state_size:",
        env.get_state_size(),
    )

    init_obs, _ = env.reset(seed=args.seed)
    init_states = env.get_state()
    rounded_states = np.round(init_states, 6)
    unique_states = np.unique(rounded_states, axis=0).shape[0]
    print(f"[sanity] unique initial states: {unique_states}/{env.num_envs}")
    if unique_states < env.num_envs:
        print("[warn] Some VMAS vector envs appear identical at reset.")

    if hasattr(env, "act_low") and hasattr(env, "act_high"):
        act_low_t = torch.from_numpy(np.asarray(env.act_low, dtype=np.float32)).to(device)
        act_high_t = torch.from_numpy(np.asarray(env.act_high, dtype=np.float32)).to(device)
    else:
        act_dim = env.get_action_size()
        act_low_t = torch.full((act_dim,), -1.0, device=device)
        act_high_t = torch.full((act_dim,), 1.0, device=device)

    action_scale = (act_high_t - act_low_t) / 2.0
    action_bias = (act_high_t + act_low_t) / 2.0

    actor = Actor(env.get_obs_size(), args.actor_hidden_dim, args.actor_num_layers, env.get_action_size()).to(device)
    target_actor = copy.deepcopy(actor).to(device)

    maddpg_input_dim = env.get_state_size() + env.n_agents * env.get_action_size()
    critic = Critic(maddpg_input_dim, args.critic_hidden_dim, args.critic_num_layers, env.get_action_size(), env.n_agents).to(device)
    target_critic = copy.deepcopy(critic).to(device)

    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"

    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"MADDPG-continuous-{run_name}",
        )

    writer = SummaryWriter(f"runs/MADDPG-continuous-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space=env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        normalize_reward=args.normalize_reward,
        device=device,
    )

    cluster_score_windows: dict[tuple, list[float]] = {}
    cluster_alpha: dict[tuple, float] = {}

    ep_rewards, ep_lengths = [], []
    num_episode, num_updates, step = 0, 0, 0
    collector_step = 0

    semantic_total_steps = 0
    semantic_kept_steps = 0

    next_eval_step = int(args.eval_steps)

    obs, _ = env.reset(seed=args.seed)

    if step < 5:
        print("obs shape:", obs.shape, "min/max:", float(obs.min()), float(obs.max()))
        st = env.get_state()
        print("state shape:", st.shape, "min/max:", float(st.min()), float(st.max()))

    while step < args.total_timesteps:
        env_batch = obs.shape[0]
        done_env = np.zeros((env_batch,), dtype=bool)
        ep_reward = np.zeros((env_batch,), dtype=np.float32)
        ep_len = np.zeros((env_batch,), dtype=np.int32)

        last_cluster_entropy = 0.0
        last_num_clusters = 1.0
        last_mean_policy_dist = 0.0

        while not bool(np.all(done_env)) and step < args.total_timesteps:
            state = env.get_state()

            obs_t = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                raw_action = actor.act(obs_t)
                a_det = action_scale * torch.tanh(raw_action) + action_bias

                sigma = _noise_schedule(
                    step=step,
                    total_steps=args.total_timesteps,
                    start=args.exploration_noise_start,
                    end=args.exploration_noise_end,
                    anneal_frac=args.exploration_anneal_frac,
                )

                noisy_action = a_det + sigma * action_scale * torch.randn_like(a_det)
                noisy_action = torch.max(torch.min(noisy_action, act_high_t), act_low_t)

                actions_np = noisy_action.detach().cpu().numpy().astype(np.float32)
                policy_np = a_det.detach().cpu().numpy().astype(np.float32)

            next_obs, reward, done, truncated, infos = env.step(actions_np)

            reward = np.asarray(reward, dtype=np.float32).reshape(-1)
            done = np.asarray(done, dtype=bool).reshape(-1)
            truncated = np.asarray(truncated, dtype=bool).reshape(-1)
            terminal = np.logical_or(done, truncated)

            next_state = env.get_state()

            keep_mask = np.ones((env_batch,), dtype=bool)
            semantic_score = np.zeros((env_batch,), dtype=np.float32)

            if args.semantic_enabled:
                if args.semantic_mode == "advantage":
                    for e in range(env_batch):
                        obs_base = _strip_agent_ids(obs[e], env.n_agents, args.agent_ids)
                        policy_vecs = policy_np[e]

                        if args.cluster_enabled:
                            clusters, _, cstats = _build_clusters_continuous(
                                obs_base=obs_base,
                                policy_vecs=policy_vecs,
                                mode=args.cluster_mode,
                                prox_radius=args.cluster_proximity_radius,
                                policy_dist_thresh=args.cluster_policy_dist_thresh,
                            )
                        else:
                            clusters = [list(range(env.n_agents))]
                            cstats = {"mean_pairwise_policy_dist": 0.0, "num_clusters": 1.0}

                        abs_adv = _compute_advantage_per_agent_continuous(
                            critic=critic,
                            actor=actor,
                            obs_np=obs[e],
                            state_np=state[e],
                            actions_taken_np=actions_np[e],
                            device=device,
                            action_scale=action_scale,
                            action_bias=action_bias,
                        )

                        cluster_scores = {}
                        for comp in clusters:
                            cluster_key = tuple(sorted(comp))
                            score = float(np.mean(abs_adv[np.asarray(comp, dtype=np.int64)]))
                            cluster_scores[cluster_key] = score

                            if cluster_key not in cluster_score_windows:
                                cluster_score_windows[cluster_key] = []
                                cluster_alpha[cluster_key] = 0.0

                            cluster_score_windows[cluster_key].append(score)
                            if len(cluster_score_windows[cluster_key]) > args.adv_alpha_window:
                                cluster_score_windows[cluster_key] = cluster_score_windows[cluster_key][-args.adv_alpha_window :]

                            if len(cluster_score_windows[cluster_key]) >= 100:
                                q = float(
                                    np.quantile(
                                        np.asarray(cluster_score_windows[cluster_key], dtype=np.float32),
                                        1.0 - float(args.adv_keep_frac),
                                    )
                                )
                                cluster_alpha[cluster_key] = (
                                    float(args.adv_alpha_ema_beta) * float(cluster_alpha[cluster_key])
                                    + (1.0 - float(args.adv_alpha_ema_beta)) * q
                                )

                        if step < args.adv_warmup_steps:
                            keep_mask[e] = True
                        else:
                            passed = [
                                float(score) >= float(cluster_alpha.get(cluster_key, 0.0))
                                for cluster_key, score in cluster_scores.items()
                            ]
                            keep_mask[e] = bool(all(passed)) if args.cluster_keep_rule == "all" else bool(any(passed))

                        semantic_score[e] = float(np.mean(abs_adv))
                        last_cluster_entropy = _shannon_entropy_of_cluster_sizes(clusters, env.n_agents)
                        last_num_clusters = float(len(clusters))
                        last_mean_policy_dist = float(cstats["mean_pairwise_policy_dist"])
                else:
                    semantic_score = np.asarray(infos.get("semantic_score", np.zeros((env_batch,), dtype=np.float32))).reshape(-1)
                    keep_mask = np.asarray(infos.get("semantic_keep", np.ones((env_batch,), dtype=bool))).reshape(-1)

            rb.store_batch(
                obs=obs,
                actions=actions_np,
                rewards=reward,
                next_obs=next_obs,
                states=state,
                next_states=next_state,
                dones=terminal.astype(np.float32),
                keep_mask=keep_mask,
                semantic_score=semantic_score,
            )

            semantic_total_steps += int(env_batch)
            semantic_kept_steps += int(np.sum(keep_mask))

            ep_reward += reward
            ep_len += (~done_env).astype(np.int32)

            done_env = np.logical_or(done_env, terminal)
            obs = next_obs

            step += int(env_batch)
            collector_step += 1

            enough_data = rb.size >= max(args.batch_size, args.learning_starts)
            if enough_data and (collector_step % args.train_freq == 0):
                num_updates = _do_training_updates(
                    args=args,
                    rb=rb,
                    actor=actor,
                    critic=critic,
                    target_actor=target_actor,
                    target_critic=target_critic,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    action_scale=action_scale,
                    action_bias=action_bias,
                    device=device,
                    step=step,
                    writer=writer,
                    num_updates=num_updates,
                )

            if step >= args.total_timesteps:
                break

        num_episode += 1
        ep_rewards.extend(ep_reward.tolist())
        ep_lengths.extend(ep_len.tolist())

        if num_episode % args.log_every == 0:
            writer.add_scalar("rollout/ep_reward", float(np.mean(ep_rewards)), step)
            writer.add_scalar("rollout/ep_length", float(np.mean(ep_lengths)), step)
            writer.add_scalar("rollout/num_episodes", num_episode, step)
            writer.add_scalar("rollout/reward_mean_running", float(rb.reward_mean), step)
            writer.add_scalar("rollout/reward_std_running", float(rb.reward_std), step)
            ep_rewards, ep_lengths = [], []

        if args.semantic_enabled and num_episode % args.semantic_log_every == 0:
            writer.add_scalar("semantic/step_keep_rate", float(semantic_kept_steps) / max(1, semantic_total_steps), step)
            writer.add_scalar("semantic/adv_keep_frac_target", float(args.adv_keep_frac), step)

            if len(cluster_alpha) > 0:
                writer.add_scalar("semantic/adv_alpha_mean", float(np.mean(list(cluster_alpha.values()))), step)
                writer.add_scalar("semantic/adv_alpha_max", float(np.max(list(cluster_alpha.values()))), step)

            if args.cluster_enabled:
                writer.add_scalar("cluster/entropy", float(last_cluster_entropy), step)
                writer.add_scalar("cluster/num_clusters", float(last_num_clusters), step)
                writer.add_scalar("cluster/mean_pairwise_policy_dist", float(last_mean_policy_dist), step)

        if step >= next_eval_step:
            video_root = Path(args.eval_video_dir) / run_name / f"step_{step}"
            video_root.mkdir(parents=True, exist_ok=True)
            print(f"[eval] num_episode={num_episode} step={step} saving={args.eval_save_video}")

            eval_obs, _ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward, eval_ep_length = [], []
            current_reward, current_ep_length = 0.0, 0
            frames = []

            while eval_ep < args.num_eval_ep:
                if args.eval_render or args.eval_save_video:
                    frame = None
                    if hasattr(eval_env, "render"):
                        frame = eval_env.render(mode="rgb_array")
                    if frame is not None and len(frames) < int(args.eval_video_max_frames):
                        frames.append(frame)

                with torch.no_grad():
                    eval_obs_t = torch.from_numpy(eval_obs).float().to(device)
                    eval_actions = action_scale * torch.tanh(actor.act(eval_obs_t)) + action_bias

                next_obs_, reward, done, truncated, _ = eval_env.step(eval_actions.cpu().numpy())
                reward_scalar = float(np.asarray(reward).reshape(-1)[0])
                done_scalar = bool(np.asarray(done).reshape(-1)[0])
                truncated_scalar = bool(np.asarray(truncated).reshape(-1)[0])

                current_reward += reward_scalar
                current_ep_length += 1
                eval_obs = next_obs_

                if done_scalar or truncated_scalar:
                    if args.eval_save_video:
                        out_path = video_root / f"eval_ep_{eval_ep}"
                        _maybe_write_video(frames, out_path, fps=int(args.eval_video_fps), fmt=args.eval_video_format)

                    frames = []
                    eval_obs, _ = eval_env.reset()
                    eval_ep_reward.append(current_reward)
                    eval_ep_length.append(current_ep_length)
                    current_reward, current_ep_length = 0.0, 0
                    eval_ep += 1

            writer.add_scalar("eval/ep_reward", float(np.mean(eval_ep_reward)), step)
            writer.add_scalar("eval/std_ep_reward", float(np.std(eval_ep_reward)), step)
            writer.add_scalar("eval/ep_length", float(np.mean(eval_ep_length)), step)

            next_eval_step += int(args.eval_steps)

        if step < args.total_timesteps:
            obs, _ = env.reset()

    writer.close()
    if args.use_wnb:
        import wandb
        wandb.finish()

    env.close()
    eval_env.close()