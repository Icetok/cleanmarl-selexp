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
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from env.vmas_wrapper import VMASWrapper

# Optional dependency for video writing.
try:
    import imageio.v2 as imageio

    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False

print("[boot] mappo_continuous.py imported")


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

    # PPO / MAPPO hyperparameters
    rollout_steps: int = 100
    ppo_epochs: int = 10

    actor_hidden_dim: int = 128
    actor_num_layers: int = 2
    critic_hidden_dim: int = 128
    critic_num_layers: int = 2

    optimizer: str = "Adam"
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4

    total_timesteps: int = 150000
    gamma: float = 0.99
    gae_lambda: float = 0.95

    normalize_reward: bool = True
    normalize_advantage: bool = True
    normalize_return: bool = False

    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    clip_gradients: float = 1.0

    log_every: int = 10
    eval_steps: int = 5000
    num_eval_ep: int = 5

    # Eval / video
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

    # -------------------------
    # Semantic selection
    # -------------------------
    semantic_enabled: bool = False
    semantic_mode: str = "advantage"  # "advantage" | "interaction" | "reward"
    semantic_log_every: int = 50

    # soft discard / second-stage sampling
    semantic_sampling_enabled: bool = True
    semantic_candidate_multiplier: int = 4
    semantic_priority_fraction: float = 0.5

    # advantage gating
    adv_keep_frac: float = 0.2
    adv_alpha_ema_beta: float = 0.9   # kept for interface parity, unused
    adv_alpha_window: int = 5000      # kept for interface parity, unused
    adv_warmup_steps: int = 5000

    # clustering
    cluster_enabled: bool = False
    cluster_mode: str = "proximity"
    cluster_proximity_radius: float = 0.75
    cluster_policy_dist_thresh: float = 0.25
    cluster_keep_rule: str = "any"


class RunningRewardStats:
    def __init__(self):
        self.reward_count = 0
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0

    def update(self, rewards: np.ndarray):
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if rewards.size == 0:
            return
        self.reward_count += int(rewards.size)
        self.reward_sum += float(np.sum(rewards))
        self.reward_sq_sum += float(np.sum(rewards ** 2))

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


class RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        act_dim: int,
        device: str = "cpu",
    ):
        self.rollout_steps = int(rollout_steps)
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.device = device
        self.reset()

    def reset(self):
        T, E, N, O, S, A = (
            self.rollout_steps,
            self.num_envs,
            self.num_agents,
            self.obs_dim,
            self.state_dim,
            self.act_dim,
        )

        self.obs = torch.zeros((T, E, N, O), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((T, E, N, A), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((T, E, N), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((T, E), dtype=torch.float32, device=self.device)
        self.states = torch.zeros((T, E, S), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((T, E), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((T, E), dtype=torch.float32, device=self.device)

        # semantic metadata per step/env
        self.keep_mask = torch.ones((T, E), dtype=torch.bool, device=self.device)
        self.semantic_score = torch.zeros((T, E), dtype=torch.float32, device=self.device)

        self.ptr = 0

    def add(
        self,
        obs,
        actions,
        log_probs,
        rewards,
        states,
        dones,
        values,
        keep_mask=None,
        semantic_score=None,
    ):
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.states[t] = states
        self.dones[t] = dones
        self.values[t] = values

        if keep_mask is None:
            self.keep_mask[t] = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        else:
            self.keep_mask[t] = keep_mask

        if semantic_score is None:
            self.semantic_score[t] = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        else:
            self.semantic_score[t] = semantic_score

        self.ptr += 1


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim) -> None:
        super().__init__()
        mean_layers = [nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())]
        for _ in range(num_layer):
            mean_layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        mean_layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))
        self.mean_layers = nn.ModuleList(mean_layers)

        self.logstd_layer = nn.Parameter(torch.zeros(output_dim) - 1.0, requires_grad=True)

    def mean(self, x):
        for layer in self.mean_layers:
            x = layer(x)
        return x

    def act(self, x, action_scale, action_bias, actions=None):
        mean = self.mean(x)
        std = self.logstd_layer.exp().expand_as(mean)
        dist = Normal(mean, std)

        if actions is None:
            u = dist.rsample()
            squashed = torch.tanh(u)
            env_actions = action_scale * squashed + action_bias
        else:
            env_actions = actions
            squashed = (env_actions - action_bias) / torch.clamp(action_scale, min=1e-6)
            squashed = torch.clamp(squashed, -0.999999, 0.999999)
            u = torch.atanh(squashed)

        log_prob_u = dist.log_prob(u).sum(dim=-1)
        jac = torch.log(torch.clamp(action_scale * (1.0 - squashed.pow(2)) + 1e-6, min=1e-6)).sum(dim=-1)
        log_probs = log_prob_u - jac
        entropy = dist.entropy().sum(dim=-1)

        return env_actions, log_probs, entropy


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layer):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, 1)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads if g is not None]
    if len(norms) == 0:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.tensor(norms), d)


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


def environment(env_type, env_name, env_family, agent_ids, kwargs):
    if env_type != "vmas":
        raise ValueError("This MAPPO script is currently set up for env_type='vmas' only.")

    vmas_kwargs = dict(kwargs)

    device = vmas_kwargs.pop("device", "cpu")
    n_agents = vmas_kwargs.pop("n_agents", 4)
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


if __name__ == "__main__":
    print("[boot] entering main")
    args = tyro.cli(Args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device)

    kwargs = _build_vmas_kwargs(args, num_envs=args.vmas_num_envs)
    eval_kwargs = _build_vmas_kwargs(args, num_envs=1)

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
        "act_size:",
        env.get_action_size(),
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

    actor = Actor(
        input_dim=env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=env.get_action_size(),
    ).to(device)

    critic = Critic(
        input_dim=env.get_state_size(),
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
    ).to(device)

    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor, eps=1e-5)
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic, eps=1e-5)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"

    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"MAPPO-Continuous-{run_name}-seed{args.seed}",
        )

    writer = SummaryWriter(f"runs/MAPPO-Continuous-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    rb = RolloutBuffer(
        rollout_steps=args.rollout_steps,
        num_envs=env.num_envs,
        num_agents=env.n_agents,
        obs_dim=env.get_obs_size(),
        state_dim=env.get_state_size(),
        act_dim=env.get_action_size(),
        device=device,
    )

    reward_stats = RunningRewardStats()

    # To match MADDPG semantics as closely as possible:
    # keep full-history cluster-score records (no alpha window truncation)
    cluster_score_windows: dict[tuple, list[float]] = {}
    cluster_alpha: dict[tuple, float] = {}

    ep_rewards, ep_lengths = [], []
    num_episode, training_step, step = 0, 0, 0

    semantic_total_steps = 0
    semantic_kept_steps = 0

    next_eval_step = int(args.eval_steps)

    obs, _ = env.reset(seed=args.seed)

    if step < 5:
        print("obs shape:", obs.shape, "min/max:", float(obs.min()), float(obs.max()))
        st = env.get_state()
        print("state shape:", st.shape, "min/max:", float(st.min()), float(st.max()))

    while step < args.total_timesteps:
        rb.reset()

        last_cluster_entropy = 0.0
        last_num_clusters = 1.0
        last_mean_policy_dist = 0.0

        # -------------------------
        # Collect rollout
        # -------------------------
        for _ in range(args.rollout_steps):
            state = env.get_state()

            obs_t = torch.from_numpy(obs).float().to(device)
            state_t = torch.from_numpy(state).float().to(device)

            with torch.no_grad():
                actions_t, log_probs_t, _ = actor.act(
                    obs_t,
                    action_scale=action_scale,
                    action_bias=action_bias,
                )
                values_t = critic(state_t)

            actions_np = actions_t.detach().cpu().numpy().astype(np.float32)
            next_obs, reward, done, truncated, infos = env.step(actions_np)

            reward_np = np.asarray(reward, dtype=np.float32).reshape(-1)
            done_np = np.asarray(done, dtype=bool).reshape(-1)
            trunc_np = np.asarray(truncated, dtype=bool).reshape(-1)
            terminal_np = np.logical_or(done_np, trunc_np).astype(np.float32)

            reward_stats.update(reward_np)

            # For wrapper semantic modes, store directly now.
            # For advantage mode, overwrite after GAE computation.
            step_keep_mask_t = None
            step_semantic_score_t = None
            if args.semantic_enabled and args.semantic_mode != "advantage":
                step_keep_mask_np = np.asarray(
                    infos.get("semantic_keep", np.ones((env.num_envs,), dtype=bool))
                ).reshape(-1)
                step_semantic_score_np = np.asarray(
                    infos.get("semantic_score", np.zeros((env.num_envs,), dtype=np.float32))
                ).reshape(-1)

                step_keep_mask_t = torch.from_numpy(step_keep_mask_np).bool().to(device)
                step_semantic_score_t = torch.from_numpy(step_semantic_score_np).float().to(device)

            rb.add(
                obs=obs_t,
                actions=actions_t,
                log_probs=log_probs_t,
                rewards=torch.from_numpy(reward_np).float().to(device),
                states=state_t,
                dones=torch.from_numpy(terminal_np).float().to(device),
                values=values_t,
                keep_mask=step_keep_mask_t,
                semantic_score=step_semantic_score_t,
            )

            if args.cluster_enabled:
                for e in range(env.num_envs):
                    obs_base = _strip_agent_ids(obs[e], env.n_agents, args.agent_ids)
                    policy_vecs = actions_np[e]
                    clusters, _, cstats = _build_clusters_continuous(
                        obs_base=obs_base,
                        policy_vecs=policy_vecs,
                        mode=args.cluster_mode,
                        prox_radius=args.cluster_proximity_radius,
                        policy_dist_thresh=args.cluster_policy_dist_thresh,
                    )
                    last_cluster_entropy = _shannon_entropy_of_cluster_sizes(clusters, env.n_agents)
                    last_num_clusters = float(len(clusters))
                    last_mean_policy_dist = float(cstats["mean_pairwise_policy_dist"])

            ep_rewards.extend(reward_np.tolist())
            step += env.num_envs

            ended = np.where(terminal_np > 0.5)[0]
            if len(ended) > 0:
                ep_lengths.extend([1] * len(ended))
                num_episode += len(ended)

                obs, _ = env.reset(seed=args.seed + training_step + int(step))
            else:
                obs = next_obs

            if step >= args.total_timesteps:
                break

        # -------------------------
        # Bootstrap
        # -------------------------
        with torch.no_grad():
            next_state = env.get_state()
            next_state_t = torch.from_numpy(next_state).float().to(device)
            next_value = critic(next_state_t)

        rewards = rb.rewards[:rb.ptr]
        dones = rb.dones[:rb.ptr]
        values = rb.values[:rb.ptr]

        if args.normalize_reward:
            rew_std = torch.clamp(rewards.std(), min=1e-6)
            rewards = rewards / rew_std

        advantages = torch.zeros_like(rewards, device=device)
        returns = torch.zeros_like(rewards, device=device)

        lastgaelam = torch.zeros((env.num_envs,), dtype=torch.float32, device=device)
        for t in reversed(range(rb.ptr)):
            if t == rb.ptr - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]

            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values

        if args.normalize_advantage:
            adv_mean = advantages.mean()
            adv_std = torch.clamp(advantages.std(), min=1e-6)
            advantages = (advantages - adv_mean) / adv_std

        if args.normalize_return:
            ret_mean = returns.mean()
            ret_std = torch.clamp(returns.std(), min=1e-6)
            returns = (returns - ret_mean) / ret_std

        # -------------------------
        # Semantic selection
        # -------------------------
        if args.semantic_enabled and args.semantic_mode == "advantage":
            for t in range(rb.ptr):
                for e in range(env.num_envs):
                    obs_np = rb.obs[t, e].detach().cpu().numpy()
                    act_np = rb.actions[t, e].detach().cpu().numpy()
                    obs_base = _strip_agent_ids(obs_np, env.n_agents, args.agent_ids)

                    if args.cluster_enabled:
                        clusters, _, cstats = _build_clusters_continuous(
                            obs_base=obs_base,
                            policy_vecs=act_np,
                            mode=args.cluster_mode,
                            prox_radius=args.cluster_proximity_radius,
                            policy_dist_thresh=args.cluster_policy_dist_thresh,
                        )
                    else:
                        clusters = [list(range(env.n_agents))]
                        cstats = {"mean_pairwise_policy_dist": 0.0, "num_clusters": 1.0}

                    # MAPPO has env-step scalar GAE advantage, not per-agent Q-advantage.
                    # To keep feature parity with MADDPG's selection interface, use
                    # |advantage| as the score and propagate it over agents for clustering.
                    abs_adv_scalar = float(torch.abs(advantages[t, e]).detach().cpu().item())
                    agent_scores = np.full((env.n_agents,), abs_adv_scalar, dtype=np.float32)

                    cluster_scores = {}
                    for comp in clusters:
                        cluster_key = tuple(sorted(comp))
                        score = float(np.mean(agent_scores[np.asarray(comp, dtype=np.int64)]))
                        cluster_scores[cluster_key] = score

                        if cluster_key not in cluster_score_windows:
                            cluster_score_windows[cluster_key] = []
                            cluster_alpha[cluster_key] = 0.0

                        cluster_score_windows[cluster_key].append(score)

                        if len(cluster_score_windows[cluster_key]) >= 100:
                            q = float(
                                np.quantile(
                                    np.asarray(cluster_score_windows[cluster_key], dtype=np.float32),
                                    1.0 - float(args.adv_keep_frac),
                                )
                            )
                            cluster_alpha[cluster_key] = q

                    if step < args.adv_warmup_steps:
                        keep = True
                    else:
                        passed = [
                            float(score) >= float(cluster_alpha.get(cluster_key, 0.0))
                            for cluster_key, score in cluster_scores.items()
                        ]
                        keep = bool(all(passed)) if args.cluster_keep_rule == "all" else bool(any(passed))

                    rb.keep_mask[t, e] = bool(keep)
                    rb.semantic_score[t, e] = float(np.mean(agent_scores))

                    last_cluster_entropy = _shannon_entropy_of_cluster_sizes(clusters, env.n_agents)
                    last_num_clusters = float(len(clusters))
                    last_mean_policy_dist = float(cstats["mean_pairwise_policy_dist"])

        semantic_total_steps += int(rb.ptr * env.num_envs)
        semantic_kept_steps += int(rb.keep_mask[:rb.ptr].sum().item())

        # -------------------------
        # Flatten for PPO update
        # -------------------------
        T, E, N = rb.ptr, env.num_envs, env.n_agents

        b_obs = rb.obs[:T].reshape(T * E * N, env.get_obs_size())
        b_actions = rb.actions[:T].reshape(T * E * N, env.get_action_size())
        b_old_log_probs = rb.log_probs[:T].reshape(T * E * N)
        b_advantages = advantages[:T].unsqueeze(-1).expand(T, E, N).reshape(T * E * N)

        b_states = rb.states[:T].reshape(T * E, env.get_state_size())
        b_returns = returns[:T].reshape(T * E)

        b_keep_step = rb.keep_mask[:T].reshape(T * E)
        b_keep_agent = rb.keep_mask[:T].unsqueeze(-1).expand(T, E, N).reshape(T * E * N)

        if b_keep_step.sum().item() == 0:
            b_keep_step = torch.ones_like(b_keep_step, dtype=torch.bool, device=device)
        if b_keep_agent.sum().item() == 0:
            b_keep_agent = torch.ones_like(b_keep_agent, dtype=torch.bool, device=device)

        actor_losses = []
        critic_losses = []
        entropies_bonuses = []
        kl_divergences = []
        actor_gradients = []
        critic_gradients = []
        clipped_ratios = []

        for _ in range(args.ppo_epochs):
            _, current_logprob, current_entropy = actor.act(
                x=b_obs,
                action_scale=action_scale,
                action_bias=action_bias,
                actions=b_actions,
            )

            log_ratio = current_logprob - b_old_log_probs
            ratio = torch.exp(log_ratio)

            pg_loss1 = b_advantages * ratio
            pg_loss2 = b_advantages * torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip)

            if args.semantic_enabled:
                actor_loss = -torch.min(pg_loss1[b_keep_agent], pg_loss2[b_keep_agent]).mean()
                entropy_bonus = current_entropy[b_keep_agent].mean()
            else:
                actor_loss = -torch.min(pg_loss1, pg_loss2).mean()
                entropy_bonus = current_entropy.mean()

            total_actor_loss = actor_loss - args.entropy_coef * entropy_bonus

            current_values = critic(b_states)
            if args.semantic_enabled:
                critic_loss = F.mse_loss(current_values[b_keep_step], b_returns[b_keep_step])
            else:
                critic_loss = F.mse_loss(current_values, b_returns)

            total_loss = total_actor_loss + args.value_coef * critic_loss

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            total_loss.backward()

            actor_gradient = norm_d([p.grad for p in actor.parameters()], 2)
            critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)

            if args.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.clip_gradients)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.clip_gradients)

            actor_optimizer.step()
            critic_optimizer.step()

            if args.semantic_enabled:
                approx_kl = ((ratio[b_keep_agent] - 1) - log_ratio[b_keep_agent]).mean()
                clipped_ratio = ((ratio[b_keep_agent] - 1.0).abs() > args.ppo_clip).float().mean()
            else:
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipped_ratio = ((ratio - 1.0).abs() > args.ppo_clip).float().mean()

            training_step += 1

            actor_losses.append(float(actor_loss.detach().cpu().item()))
            critic_losses.append(float(critic_loss.detach().cpu().item()))
            entropies_bonuses.append(float(entropy_bonus.detach().cpu().item()))
            kl_divergences.append(float(approx_kl.detach().cpu().item()))
            actor_gradients.append(float(actor_gradient.detach().cpu().item()))
            critic_gradients.append(float(critic_gradient.detach().cpu().item()))
            clipped_ratios.append(float(clipped_ratio.detach().cpu().item()))

        writer.add_scalar("train/critic_loss", np.mean(critic_losses), step)
        writer.add_scalar("train/actor_loss", np.mean(actor_losses), step)
        writer.add_scalar("train/entropy", np.mean(entropies_bonuses), step)
        writer.add_scalar("train/kl_divergence", np.mean(kl_divergences), step)
        writer.add_scalar("train/clipped_ratios", np.mean(clipped_ratios), step)
        writer.add_scalar("train/actor_gradients", np.mean(actor_gradients), step)
        writer.add_scalar("train/critic_gradients", np.mean(critic_gradients), step)
        writer.add_scalar("train/num_updates", training_step, step)
        writer.add_scalar("train/reward_std_running", float(reward_stats.reward_std), step)

        if num_episode % args.log_every == 0 and len(ep_rewards) > 0:
            writer.add_scalar("rollout/ep_reward", float(np.mean(ep_rewards)), step)
            writer.add_scalar("rollout/ep_length", float(np.mean(ep_lengths)) if len(ep_lengths) > 0 else 0.0, step)
            writer.add_scalar("rollout/num_episodes", num_episode, step)
            writer.add_scalar("rollout/reward_mean_running", float(reward_stats.reward_mean), step)
            writer.add_scalar("rollout/reward_std_running", float(reward_stats.reward_std), step)
            ep_rewards, ep_lengths = [], []

        if args.semantic_enabled and num_episode % args.semantic_log_every == 0:
            writer.add_scalar(
                "semantic/buffer_keep_rate",
                float(rb.keep_mask[:rb.ptr].float().mean().item()) if rb.ptr > 0 else 1.0,
                step,
            )
            writer.add_scalar(
                "semantic/step_keep_rate",
                float(semantic_kept_steps) / max(1, semantic_total_steps),
                step,
            )
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
            print(f"[eval] step={step} saving={args.eval_save_video}")

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
                    eval_actions, _, _ = actor.act(
                        x=eval_obs_t,
                        action_scale=action_scale,
                        action_bias=action_bias,
                    )

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

    writer.close()
    if args.use_wnb:
        import wandb
        wandb.finish()

    env.close()
    eval_env.close()