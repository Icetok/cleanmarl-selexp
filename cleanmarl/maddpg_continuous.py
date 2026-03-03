# cleanmarl/maddpg_continuous.py

import copy
import os
from pathlib import Path
import datetime
import random
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from env.pettingzoo_wrapper import PettingZooWrapper
from env.smaclite_wrapper import SMACliteWrapper
from env.lbf import LBFWrapper
from env.vmas_wrapper import VMASWrapper

# Optional dependency for video writing.
try:
    import imageio.v2 as imageio
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False


# -------------------------
# Command line arguments
# -------------------------
@dataclass
class Args:
    # Environment selection
    env_type: str = "pz"  # "pz" | "smaclite" | "lbf" | "vmas"
    env_name: str = "simple_spread_v3"
    env_family: str = "mpe"
    agent_ids: bool = True

    # RL hyperparameters
    gamma: float = 0.99
    buffer_size: int = 10000      # number of episodes stored (NOT transitions)
    batch_size: int = 10          # number of episodes sampled for training
    normalize_reward: bool = True

    # Actor network size
    actor_hidden_dim: int = 64
    actor_num_layers: int = 1

    # Critic network size
    critic_hidden_dim: int = 256
    critic_num_layers: int = 1

    # Training / logging
    train_freq: int = 1
    optimizer: str = "Adam"
    learning_rate_actor: float = 1e-5
    learning_rate_critic: float = 1e-5
    total_timesteps: int = 500000
    target_network_update_freq: int = 1
    polyak: float = 0.005
    clip_gradients: float = 1.0
    log_every: int = 10
    eval_steps: int = 50
    num_eval_ep: int = 5

    # Exploration noise (continuous)
    exploration_noise: float = 0.05

    # -------------------------
    # Evaluation rendering / video
    # -------------------------
    eval_render: bool = False
    eval_save_video: bool = False
    eval_video_dir: str = "eval_videos"
    eval_video_fps: int = 20
    eval_video_format: str = "gif"   # "gif" or "mp4"
    eval_video_max_frames: int = 2000

    # W&B / device / seed
    use_wnb: bool = False
    wnb_project: str = ""
    wnb_entity: str = ""
    device: str = "cpu"
    seed: int = 1

    # -------------------------
    # "Semantic layer" discarding
    # -------------------------
    semantic_enabled: bool = True
    semantic_mode: str = "advantage"  # "interaction" | "reward" | "advantage"
    semantic_discard_mode: str = "step"  # "episode" | "step"
    semantic_min_keep_frac: float = 0.05
    semantic_log_every: int = 50

    # -------------------------
    # Advantage-based gating details
    # -------------------------
    adv_keep_frac: float = 0.2
    adv_alpha_ema_beta: float = 0.9
    adv_alpha_window: int = 5000
    adv_warmup_steps: int = 5000

    # -------------------------
    # Clustering
    # -------------------------
    cluster_enabled: bool = True
    cluster_mode: str = "proximity"  # "proximity" | "policy" | "hybrid"
    cluster_proximity_radius: float = 0.75
    cluster_policy_dist_thresh: float = 0.25
    cluster_action_hist_window: int = 50
    cluster_keep_rule: str = "any"  # "all" | "any"


# -------------------------
# Actor network (continuous)
# -------------------------
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
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
        return x.squeeze()  # (B, N)

    def maddpg_inputs(self, state, actions, grad_processing, batch_action):
        # state: (B, state_dim)
        # actions: (B, N, act_dim)
        maddpg_inputs = torch.zeros((state.size(0), self.num_agents, self.input_dim), device=state.device)
        maddpg_inputs[:, :, : state.size(-1)] = state.unsqueeze(1)

        # replicate joint actions for each agent head (same structure as your discrete version)
        oh = actions.unsqueeze(1).expand(-1, self.num_agents, -1, -1).reshape(state.size(0), self.num_agents, -1)

        if grad_processing:
            b_oh = batch_action.unsqueeze(1).expand(-1, self.num_agents, -1, -1).reshape(state.size(0), self.num_agents, -1)
            mask = torch.eye(self.num_agents, device=state.device).unsqueeze(-1).expand(-1, -1, actions.size(-1)).reshape(self.num_agents, -1)
            oh = torch.where(mask.bool(), oh, b_oh)

        maddpg_inputs[:, :, state.size(-1) :] = oh
        return maddpg_inputs


# -------------------------
# Episode-based replay buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, buffer_size, num_agents, obs_space, state_space, action_space, normalize_reward=False, device="cpu"):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.normalize_reward = normalize_reward
        self.device = device

        self.episodes = [None] * buffer_size
        self.pos = 0
        self.size = 0

    def store(self, episode):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values)).float().to(self.device)
        self.episodes[self.pos] = episode
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = [self.episodes[i] for i in indices]
        lengths = [len(ep["obs"]) for ep in batch]
        max_length = max(lengths)

        obs = torch.zeros((batch_size, max_length, self.num_agents, self.obs_space), device=self.device)
        actions = torch.zeros((batch_size, max_length, self.num_agents, self.action_space), device=self.device)
        reward = torch.zeros((batch_size, max_length), device=self.device)
        states = torch.zeros((batch_size, max_length, self.state_space), device=self.device)
        done = torch.ones((batch_size, max_length), device=self.device)
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=self.device)

        for i in range(batch_size):
            L = lengths[i]
            obs[i, :L] = batch[i]["obs"]
            actions[i, :L] = batch[i]["actions"]
            reward[i, :L] = batch[i]["reward"]
            states[i, :L] = batch[i]["states"]
            done[i, :L] = batch[i]["done"]
            mask[i, :L] = True

        if self.normalize_reward and mask.any():
            mu = torch.mean(reward[mask])
            std = torch.std(reward[mask])
            reward[mask] = (reward[mask] - mu) / (std + 1e-6)

        return obs.float(), actions.float(), reward.float(), states.float(), done.float(), mask


# -------------------------
# Environment factory
# -------------------------
def environment(env_type, env_name, env_family, agent_ids, kwargs):
    if env_type == "pz":
        return PettingZooWrapper(family=env_family, env_name=env_name, agent_ids=agent_ids, **kwargs)
    if env_type == "smaclite":
        return SMACliteWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    if env_type == "lbf":
        return LBFWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    if env_type == "vmas":
        return VMASWrapper(
            scenario=env_name,
            n_agents=kwargs.get("n_agents", 3),
            max_steps=kwargs.get("max_steps", 100),
            agent_ids=agent_ids,
            device=kwargs.get("device", "cpu"),
            continuous_actions=True,  # IMPORTANT: continuous
            semantic_enabled=kwargs.get("wrapper_semantic_enabled", False),
            semantic_threshold=kwargs.get("semantic_threshold", 0.0),
            semantic_mode=kwargs.get("semantic_mode", "interaction"),
            interaction_radius=kwargs.get("interaction_radius", 0.5),
            interaction_use_inverse_mean_dist=kwargs.get("interaction_use_inverse_mean_dist", True),
        )
    raise ValueError(f"Unknown env_type: {env_type}")


# -------------------------
# Utility: gradient norm
# -------------------------
def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads if g is not None]
    if len(norms) == 0:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.tensor(norms), d)


# -------------------------
# Utility: Polyak update
# -------------------------
def soft_update(target_net, utility_net, polyak):
    for target_param, param in zip(target_net.parameters(), utility_net.parameters()):
        target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)


# -------------------------
# Semantic gating helper
# -------------------------
def _episode_semantic_gate(episode, mode, threshold, min_keep_frac):
    scores = np.asarray(episode.get("semantic_score", []), dtype=np.float32)
    keeps = np.asarray(episode.get("semantic_keep", []), dtype=bool)

    if scores.size == 0 or keeps.size == 0:
        return True, None

    if mode == "episode":
        avg_score = float(np.mean(scores))
        ok = avg_score >= float(threshold)
        return ok, None

    keep_idx = np.where(keeps)[0]
    keep_frac = float(len(keep_idx) / max(1, len(keeps)))
    if keep_frac < float(min_keep_frac):
        return False, None
    return True, keep_idx


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
    # X: (N, D)
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
    obs_base: np.ndarray,            # (N, obs_dim_base)
    policy_vecs: np.ndarray,         # (N, act_dim) e.g. current actor outputs or rolling mean actions
    mode: str,
    prox_radius: float,
    policy_dist_thresh: float,
) -> tuple[list[list[int]], dict[int, int], dict[str, float]]:
    n = policy_vecs.shape[0]
    mode = (mode or "proximity").lower().strip()

    # policy distance edges via L2 distance in action space
    Dpol = _pairwise_l2_distance(policy_vecs)  # (N,N)
    policy_edges = (Dpol <= float(policy_dist_thresh))

    # proximity edges (if obs contains 2D positions)
    prox_edges = np.zeros((n, n), dtype=bool)
    if obs_base.ndim == 2 and obs_base.shape[1] >= 2:
        pos = obs_base[:, :2].astype(np.float32, copy=False)
        diffs = pos[:, None, :] - pos[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        prox_edges = (dists <= float(prox_radius))

    if mode == "policy":
        adj = policy_edges
    elif mode == "hybrid":
        adj = policy_edges | prox_edges
    else:
        adj = prox_edges if prox_edges.any() else policy_edges

    # always include self-edges
    adj = adj | np.eye(n, dtype=bool)

    clusters = _connected_components_from_adjacency(adj)

    agent_to_cluster = {}
    for cid, comp in enumerate(clusters):
        for a in comp:
            agent_to_cluster[a] = cid

    mask = ~np.eye(n, dtype=bool)
    mean_pairwise_policy_dist = float(np.mean(Dpol[mask])) if n > 1 else 0.0
    stats = {"mean_pairwise_policy_dist": mean_pairwise_policy_dist, "num_clusters": float(len(clusters))}
    return clusters, agent_to_cluster, stats


def _shannon_entropy_of_cluster_sizes(clusters: list[list[int]], n_agents: int, eps: float = 1e-12) -> float:
    if n_agents <= 0 or len(clusters) == 0:
        return 0.0
    sizes = np.asarray([len(c) for c in clusters], dtype=np.float32)
    p = sizes / (float(n_agents) + eps)
    return -float(np.sum(p * np.log(p + eps)))


def _compute_advantage_per_agent_continuous(critic, actor, obs_np, state_np, actions_taken_np, device, act_low_t, act_high_t):
    # obs_np: (N, obs_dim)
    # state_np: (state_dim,)
    # actions_taken_np: (N, act_dim)
    obs_t = torch.from_numpy(obs_np).float().to(device)                           # (N, obs_dim)
    state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)          # (1, state_dim)

    a_taken = torch.from_numpy(actions_taken_np).float().to(device).unsqueeze(0)  # (1, N, act_dim)

    with torch.no_grad():
        a_pi = actor.act(obs_t).unsqueeze(0)                                      # (1, N, act_dim)
        a_pi = torch.clamp(a_pi, act_low_t, act_high_t)

        q_taken = critic(state_t, a_taken)                                        # (1, N) or (N,)
        q_pi = critic(state_t, a_pi)

        if q_taken.ndim == 1:
            q_taken = q_taken.unsqueeze(0)
        if q_pi.ndim == 1:
            q_pi = q_pi.unsqueeze(0)

        adv = (q_taken - q_pi).reshape(-1)                                        # (N,)
        abs_adv = torch.abs(adv).detach().cpu().numpy().astype(np.float32)

    return abs_adv


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


# -------------------------
# Main training loop
# -------------------------
if __name__ == "__main__":
    args = tyro.cli(Args)

    # Reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    # Env kwargs: VMAS wrapper semantics only used for interaction/reward
    kwargs = {}
    if args.env_type == "vmas":
        wrapper_semantic_enabled = args.semantic_enabled and (args.semantic_mode in ("interaction", "reward"))
        kwargs = {
            "device": args.device,
            "wrapper_semantic_enabled": wrapper_semantic_enabled,
            "semantic_mode": args.semantic_mode if wrapper_semantic_enabled else "interaction",
            "semantic_threshold": 0.0,
        }

    env = environment(args.env_type, args.env_name, args.env_family, args.agent_ids, kwargs)
    eval_env = environment(args.env_type, args.env_name, args.env_family, args.agent_ids, kwargs)

    # Determine action bounds
    if hasattr(env, "act_low") and hasattr(env, "act_high"):
        act_low_t = torch.from_numpy(np.asarray(env.act_low, dtype=np.float32)).to(device)
        act_high_t = torch.from_numpy(np.asarray(env.act_high, dtype=np.float32)).to(device)
    else:
        # fallback: assume [-1, 1] box
        act_dim = env.get_action_size()
        act_low_t = torch.full((act_dim,), -1.0, device=device)
        act_high_t = torch.full((act_dim,), 1.0, device=device)

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

    # Per-cluster alpha tracking
    cluster_score_windows: dict[tuple, deque] = {}
    cluster_alpha: dict[tuple, float] = {}

    # For continuous "policy distance": keep rolling action vectors per agent
    action_hist = [deque(maxlen=int(args.cluster_action_hist_window)) for _ in range(env.n_agents)]

    # Rollout metrics
    ep_rewards, ep_lengths, ep_stats = [], [], []
    num_episode, num_updates, step = 0, 0, 0

    # Semantic bookkeeping
    semantic_total_steps = 0
    semantic_kept_steps = 0
    semantic_kept_episodes = 0
    semantic_dropped_episodes = 0

    while step < args.total_timesteps:
        episode = {
            "obs": [],
            "actions": [],
            "reward": [],
            "states": [],
            "done": [],
            "semantic_score": [],
            "semantic_keep": [],
        }

        # reset
        try:
            obs, _ = env.reset(seed=args.seed)
        except TypeError:
            obs, _ = env.reset()

        ep_reward, ep_length = 0.0, 0
        done, truncated = False, False

        last_cluster_entropy = 0.0
        last_num_clusters = 1.0
        last_mean_policy_dist = 0.0

        while not done and not truncated:
            state = env.get_state()

            # action selection (continuous + exploration noise)
            with torch.no_grad():
                a = actor.act(torch.from_numpy(obs).float().to(device))  # (N, act_dim)
                if args.exploration_noise > 0:
                    a = a + float(args.exploration_noise) * torch.randn_like(a)
                a = torch.clamp(a, act_low_t, act_high_t)
                actions_np = a.cpu().numpy().astype(np.float32)

            for i in range(env.n_agents):
                action_hist[i].append(actions_np[i].copy())

            next_obs, reward, done, truncated, infos = env.step(actions_np)

            # --- semantic keep decision ---
            if args.semantic_enabled:
                if args.semantic_mode == "advantage":
                    obs_base = _strip_agent_ids(obs, env.n_agents, args.agent_ids)

                    # policy vectors for clustering:
                    # use rolling mean action per agent (more stable than single step)
                    policy_vecs = np.zeros((env.n_agents, env.get_action_size()), dtype=np.float32)
                    for i in range(env.n_agents):
                        if len(action_hist[i]) == 0:
                            policy_vecs[i] = 0.0
                        else:
                            policy_vecs[i] = np.mean(np.stack(list(action_hist[i])), axis=0).astype(np.float32)

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
                        obs_np=obs,
                        state_np=state,
                        actions_taken_np=actions_np,
                        device=device,
                        act_low_t=act_low_t,
                        act_high_t=act_high_t,
                    )

                    cluster_scores = {}
                    for comp in clusters:
                        cluster_key = tuple(sorted(comp))
                        cluster_scores[cluster_key] = float(np.mean(abs_adv[np.asarray(comp, dtype=np.int64)]))

                    # update per-cluster alpha
                    for cluster_key, score in cluster_scores.items():
                        if cluster_key not in cluster_score_windows:
                            cluster_score_windows[cluster_key] = deque(maxlen=int(args.adv_alpha_window))
                            cluster_alpha[cluster_key] = 0.0

                        cluster_score_windows[cluster_key].append(float(score))

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
                        semantic_keep = True
                    else:
                        passed = [
                            float(score) >= float(cluster_alpha.get(cluster_key, 0.0))
                            for cluster_key, score in cluster_scores.items()
                        ]
                        semantic_keep = bool(all(passed)) if args.cluster_keep_rule == "all" else bool(any(passed))

                    semantic_score = float(np.mean(abs_adv))
                    last_cluster_entropy = _shannon_entropy_of_cluster_sizes(clusters, env.n_agents)
                    last_num_clusters = float(len(clusters))
                    last_mean_policy_dist = float(cstats["mean_pairwise_policy_dist"])
                else:
                    # interaction/reward semantic score comes from wrapper
                    semantic_score = float(infos.get("semantic_score", 0.0))
                    semantic_keep = bool(infos.get("semantic_keep", True))
            else:
                semantic_score = 0.0
                semantic_keep = True

            # bookkeeping
            ep_reward += float(reward)
            ep_length += 1
            step += 1

            episode["obs"].append(obs.astype(np.float32))
            episode["actions"].append(actions_np.astype(np.float32))
            episode["reward"].append(np.float32(reward))
            episode["done"].append(np.float32(done))
            episode["states"].append(np.asarray(state, dtype=np.float32))
            episode["semantic_score"].append(np.float32(semantic_score))
            episode["semantic_keep"].append(bool(semantic_keep))

            semantic_total_steps += 1
            semantic_kept_steps += int(semantic_keep)

            obs = next_obs

        # end episode
        num_episode += 1
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)

        if args.env_type == "smaclite":
            ep_stats.append(infos)

        # Store episode with gating
        store_ok, keep_idx = _episode_semantic_gate(
            episode=episode,
            mode=args.semantic_discard_mode,
            threshold=0.0,
            min_keep_frac=args.semantic_min_keep_frac,
        )

        if store_ok:
            semantic_kept_episodes += 1
            if args.semantic_discard_mode == "step" and keep_idx is not None:
                filtered_episode = {k: [v[i] for i in keep_idx] for k, v in episode.items()}
                rb.store(filtered_episode)
            else:
                rb.store(episode)
        else:
            semantic_dropped_episodes += 1

        # logging
        if num_episode % args.log_every == 0:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/num_episodes", num_episode, step)
            if args.env_type == "smaclite":
                writer.add_scalar("rollout/battle_won", np.mean([info["battle_won"] for info in ep_stats]), step)
            ep_rewards, ep_lengths, ep_stats = [], [], []

        if args.semantic_enabled and num_episode % args.semantic_log_every == 0:
            writer.add_scalar("semantic/step_keep_rate", float(semantic_kept_steps) / max(1, semantic_total_steps), step)
            writer.add_scalar(
                "semantic/episode_keep_rate",
                float(semantic_kept_episodes) / max(1, semantic_kept_episodes + semantic_dropped_episodes),
                step,
            )
            writer.add_scalar("semantic/episodes_dropped", semantic_dropped_episodes, step)

            if args.semantic_mode == "advantage":
                writer.add_scalar("semantic/adv_keep_frac_target", float(args.adv_keep_frac), step)
                if len(cluster_alpha) > 0:
                    writer.add_scalar("semantic/adv_alpha_mean", float(np.mean(list(cluster_alpha.values()))), step)
                    writer.add_scalar("semantic/adv_alpha_max", float(np.max(list(cluster_alpha.values()))), step)

            if args.semantic_mode == "advantage" and args.cluster_enabled:
                writer.add_scalar("cluster/entropy", float(last_cluster_entropy), step)
                writer.add_scalar("cluster/num_clusters", float(last_num_clusters), step)
                writer.add_scalar("cluster/mean_pairwise_policy_dist", float(last_mean_policy_dist), step)

        # -------------------------
        # Training step
        # -------------------------
        if num_episode > args.batch_size and (num_episode % args.train_freq == 0):
            batch_obs, batch_action, batch_reward, batch_states, batch_done, batch_mask = rb.sample(args.batch_size)

            # critic update
            critic_loss = 0.0
            for t in range(batch_obs.size(1)):
                with torch.no_grad():
                    if t == batch_obs.size(1) - 1:
                        targets = batch_reward[:, t].unsqueeze(-1).expand(-1, env.n_agents)
                    else:
                        a_next = target_actor.act(batch_obs[:, t + 1])
                        a_next = torch.clamp(a_next, act_low_t, act_high_t)
                        q_next = target_critic(batch_states[:, t + 1], a_next)
                        q_next = torch.nan_to_num(q_next, nan=0.0)

                        targets = (
                            batch_reward[:, t].unsqueeze(-1).expand(-1, env.n_agents)
                            + args.gamma
                            * (1 - batch_done[:, t].unsqueeze(-1).expand(-1, env.n_agents))
                            * q_next
                        )

                q_values = critic(batch_states[:, t], batch_action[:, t])
                critic_loss = critic_loss + (
                    F.mse_loss(targets[batch_mask[:, t]], q_values[batch_mask[:, t]]) * batch_mask[:, t].sum()
                )

            critic_loss = critic_loss / batch_mask.sum()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_gradients = norm_d([p.grad for p in critic.parameters()], 2)
            if args.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.clip_gradients)
            critic_optimizer.step()

            # actor update
            actor_loss = 0.0
            for t in range(batch_obs.size(1)):
                a_pi = actor.act(batch_obs[:, t])
                a_pi = torch.clamp(a_pi, act_low_t, act_high_t)
                qvals = critic(batch_states[:, t], a_pi, grad_processing=True, batch_action=batch_action[:, t])
                actor_loss = actor_loss - qvals[batch_mask[:, t]].sum()

            actor_loss = actor_loss / batch_mask.sum()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_gradients = norm_d([p.grad for p in actor.parameters()], 2)
            if args.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.clip_gradients)
            actor_optimizer.step()

            num_updates += 1
            writer.add_scalar("train/critic_loss", float(critic_loss.detach().cpu().item()), step)
            writer.add_scalar("train/actor_loss", float(actor_loss.detach().cpu().item()), step)
            writer.add_scalar("train/actor_gradients", float(actor_gradients.detach().cpu().item()), step)
            writer.add_scalar("train/critic_gradients", float(critic_gradients.detach().cpu().item()), step)
            writer.add_scalar("train/num_updates", num_updates, step)

            if num_episode % args.target_network_update_freq == 0:
                soft_update(target_net=target_actor, utility_net=actor, polyak=args.polyak)
                soft_update(target_net=target_critic, utility_net=critic, polyak=args.polyak)

        # -------------------------
        # Evaluation loop + optional rendering/video
        # -------------------------
        if num_episode % args.eval_steps == 0:
            eval_obs, _ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward, eval_ep_length = [], []
            current_reward, current_ep_length = 0.0, 0

            video_root = Path(args.eval_video_dir) / run_name / f"step_{step}"
            frames = []

            while eval_ep < args.num_eval_ep:
                if args.eval_render or args.eval_save_video:
                    frame = None
                    if hasattr(eval_env, "render"):
                        frame = eval_env.render(mode="rgb_array")
                    if frame is not None and len(frames) < int(args.eval_video_max_frames):
                        frames.append(frame)

                with torch.no_grad():
                    eval_actions = actor.act(torch.from_numpy(eval_obs).float().to(device))
                    eval_actions = torch.clamp(eval_actions, act_low_t, act_high_t)

                next_obs_, reward, done, truncated, _ = eval_env.step(eval_actions.cpu().numpy())
                current_reward += float(reward)
                current_ep_length += 1
                eval_obs = next_obs_

                if done or truncated:
                    if args.eval_save_video:
                        out_path = video_root / f"eval_ep_{eval_ep}"
                        _maybe_write_video(frames, out_path, fps=int(args.eval_video_fps), fmt=args.eval_video_format)

                    frames = []
                    eval_obs, _ = eval_env.reset()
                    eval_ep_reward.append(current_reward)
                    eval_ep_length.append(current_ep_length)
                    current_reward, current_ep_length = 0.0, 0
                    eval_ep += 1

            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward", np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)

    writer.close()
    if args.use_wnb:
        import wandb
        wandb.finish()
    env.close()
    eval_env.close()