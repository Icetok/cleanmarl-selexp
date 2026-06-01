import torch
import tyro
import datetime
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import torch.nn.functional as F

from env.pettingzoo_wrapper import PettingZooWrapper
from env.smaclite_wrapper import SMACliteWrapper
from env.lbf import LBFWrapper

from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

try:
    import imageio.v2 as imageio
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False


@dataclass
class Args:
    env_type: str = "lbf"
    env_name: str = "Foraging-8x8-2p-1f-v3"
    env_family: str = "mpe"
    agent_ids: bool = True

    batch_size: int = 16

    actor_hidden_dim: int = 128
    actor_num_layers: int = 1
    critic_hidden_dim: int = 128
    critic_num_layers: int = 1
    activation: str = "relu"

    optimizer: str = "Adam"
    learning_rate_actor: float = 8e-4
    learning_rate_critic: float = 8e-4
    adam_eps: float = 1e-5

    total_timesteps: int = 1000000
    gamma: float = 0.99
    td_lambda: float = 0.95

    normalize_reward: bool = False
    normalize_advantage: bool = True
    normalize_return: bool = False

    epochs: int = 3
    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 1.0
    clip_gradients: float = 5.0

    log_every: int = 10
    eval_steps: int = 10
    num_eval_ep: int = 10

    # LBF-specific
    lbf_time_limit: int = 150
    lbf_reward_aggr: str = "sum"

    # Selection
    semantic_enabled: bool = False
    semantic_mode: str = "advantage"
    semantic_log_every: int = 10

    adv_keep_frac: float = 0.5
    adv_min_keep_frac: float = 0.1
    adv_warmup_steps: int = 0
    adv_positive_only: bool = False
    adv_use_abs: bool = False
    cf_advantage_enabled: bool = False
    
    # Eval / video
    eval_save_video: bool = True
    eval_video_dir: str = "eval_videos"
    eval_video_fps: int = 8
    eval_video_format: str = "mp4"
    eval_video_max_frames: int = 300
    eval_num_videos_to_save: int = 3

    use_wnb: bool = False
    wnb_project: str = ""
    wnb_entity: str = ""
    device: str = "cpu"
    seed: int = 1


def _make_activation(name: str) -> nn.Module:
    name = name.lower().strip()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class RunningRewardStats:
    def __init__(self):
        self.reward_count = 0
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0

    def update(self, rewards):
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if rewards.size == 0:
            return
        self.reward_count += int(rewards.size)
        self.reward_sum += float(np.sum(rewards))
        self.reward_sq_sum += float(np.sum(rewards ** 2))

    @property
    def reward_mean(self):
        if self.reward_count == 0:
            return 0.0
        return self.reward_sum / self.reward_count

    @property
    def reward_std(self):
        if self.reward_count <= 1:
            return 1.0
        mean = self.reward_mean
        var = max(self.reward_sq_sum / self.reward_count - mean * mean, 1e-8)
        return float(np.sqrt(var))


class RolloutBuffer:
    def __init__(
        self,
        buffer_size,
        num_agents,
        obs_space,
        state_space,
        action_space,
        device="cpu",
    ):
        self.buffer_size = int(buffer_size)
        self.num_agents = int(num_agents)
        self.obs_space = int(obs_space)
        self.state_space = int(state_space)
        self.action_space = int(action_space)
        self.device = device
        self.episodes = [None] * self.buffer_size
        self.pos = 0

    def add(self, episode):
        if len(episode["obs"]) == 0:
            return
        
        out = {}
        for key, values in episode.items():
            arr = np.stack(values)
            out[key] = torch.from_numpy(arr).to(self.device)

        out["obs"] = out["obs"].float()
        out["avail_actions"] = out["avail_actions"].bool()
        out["actions"] = out["actions"].long()
        out["log_prob"] = out["log_prob"].float()
        out["reward"] = out["reward"].float()
        out["reward_agents"] = out["reward_agents"].float()
        out["states"] = out["states"].float()
        out["done"] = out["done"].float()

        self.episodes[self.pos] = out
        self.pos += 1

    def get_batch(self, normalize_reward=False):
        valid_episodes = [ep for ep in self.episodes[:self.pos] if ep is not None]

        if len(valid_episodes) == 0:
            return None

        lengths = [len(episode["obs"]) for episode in valid_episodes]
        max_length = max(lengths)

        self.pos = 0

        B, T, N = len(valid_episodes), max_length, self.num_agents

        obs = torch.zeros((B, T, N, self.obs_space), device=self.device)
        avail_actions = torch.zeros((B, T, N, self.action_space), dtype=torch.bool, device=self.device)
        actions = torch.zeros((B, T, N), dtype=torch.long, device=self.device)
        log_probs = torch.zeros((B, T, N), device=self.device)
        reward = torch.zeros((B, T), device=self.device)
        reward_agents = torch.zeros((B, T, N), device=self.device)
        states = torch.zeros((B, T, self.state_space), device=self.device)
        done = torch.zeros((B, T), device=self.device)
        mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)

        for i, ep in enumerate(valid_episodes):
            length = lengths[i]

            obs[i, :length] = ep["obs"]
            avail_actions[i, :length] = ep["avail_actions"]
            actions[i, :length] = ep["actions"]
            log_probs[i, :length] = ep["log_prob"]
            reward[i, :length] = ep["reward"]
            reward_agents[i, :length] = ep["reward_agents"]
            states[i, :length] = ep["states"]
            done[i, :length] = ep["done"]
            mask[i, :length] = True

        reward_std_used = None
        if normalize_reward:
            reward_std_used = torch.clamp(reward[mask].std(), min=1e-6)
            reward[mask] = reward[mask] / reward_std_used
            reward_agents[mask] = reward_agents[mask] / reward_std_used

        self.episodes = [None] * self.buffer_size

        return (
            obs,
            actions,
            log_probs,
            reward,
            reward_agents,
            states,
            avail_actions,
            done,
            mask,
            reward_std_used,
        )


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim, activation_name="relu") -> None:
        super().__init__()
        self.output_dim = output_dim

        layers = [nn.Sequential(nn.Linear(input_dim, hidden_dim), _make_activation(activation_name))]
        for _ in range(num_layer):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), _make_activation(activation_name)))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(layers)

    def logits(self, x, avail_action=None):
        for layer in self.layers:
            x = layer(x)

        if avail_action is not None:
            x = x.masked_fill(~avail_action, -1e9)

        return x

    def act(self, x, avail_action=None, deterministic=False):
        logits = self.logits(x, avail_action)
        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy()
    

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, activation_name="relu") -> None:
        super().__init__()

        layers = [nn.Sequential(nn.Linear(input_dim, hidden_dim), _make_activation(activation_name))]
        for _ in range(num_layer):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), _make_activation(activation_name)))
        layers.append(nn.Linear(hidden_dim, 1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)
    
class QCritic(nn.Module):
    def __init__(self, state_dim, joint_action_dim, hidden_dim, num_layer, activation_name="relu"):
        super().__init__()
        input_dim = state_dim + joint_action_dim

        layers = [nn.Sequential(nn.Linear(input_dim, hidden_dim), _make_activation(activation_name))]
        for _ in range(num_layer):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), _make_activation(activation_name)))
        layers.append(nn.Linear(hidden_dim, 1))

        self.layers = nn.ModuleList(layers)

    def forward(self, states, joint_actions):
        x = torch.cat([states, joint_actions], dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)


def environment(env_type, env_name, env_family, agent_ids, kwargs):
    if env_type == "pz":
        return PettingZooWrapper(
            family=env_family,
            env_name=env_name,
            agent_ids=agent_ids,
            **kwargs,
        )

    if env_type == "smaclite":
        return SMACliteWrapper(
            map_name=env_name,
            agent_ids=agent_ids,
            **kwargs,
        )

    if env_type == "lbf":
        return LBFWrapper(
            map_name=env_name,
            agent_ids=agent_ids,
            **kwargs,
        )

    raise ValueError(f"Unsupported env_type: {env_type}")


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), ord=d) for g in grads if g is not None]
    if len(norms) == 0:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms), ord=d)


def build_selection_mask(
    scores,
    valid_mask,
    keep_frac,
    min_keep_frac,
    use_abs=False,
    positive_only=False,
):
    flat_scores = scores.reshape(-1)
    flat_valid = valid_mask.reshape(-1)

    if use_abs or positive_only:
        eligible_mask = flat_valid & (flat_scores > 0.0)
    else:
        eligible_mask = flat_valid & (flat_scores != 0.0)

    eligible_idx = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)

    keep_flat = torch.zeros_like(flat_scores, dtype=torch.float32)

    if eligible_idx.numel() == 0:
        keep_flat[flat_valid] = 1.0
        return keep_flat.reshape_as(scores), None

    eligible_scores = flat_scores[eligible_idx]

    min_keep = max(1, int(float(min_keep_frac) * eligible_idx.numel()))
    k_keep = int(float(keep_frac) * eligible_idx.numel())
    k_keep = max(min_keep, k_keep)
    k_keep = min(k_keep, eligible_idx.numel())

    top_local_idx = torch.topk(
        eligible_scores,
        k=k_keep,
        largest=True,
        sorted=False,
    ).indices

    selected_idx = eligible_idx[top_local_idx]
    keep_flat[selected_idx] = 1.0

    threshold = flat_scores[selected_idx].min().item()

    return keep_flat.reshape_as(scores), threshold

def _as_uint8_rgb(frame):
    if frame is None:
        return None
    f = np.asarray(frame)
    if f.dtype == np.uint8:
        return f
    f = f.astype(np.float32)
    if f.max() <= 1.5:
        f = f * 255.0
    return np.clip(f, 0, 255).astype(np.uint8)


def _maybe_write_video(frames, out_path: Path, fps: int, fmt: str):
    if not frames:
        print("[warn] no frames collected; skipping video")
        return
    if not _HAS_IMAGEIO:
        print("[warn] imageio unavailable; skipping video. Install with: pip install imageio imageio-ffmpeg")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned = []
    for frame in frames:
        frame = _as_uint8_rgb(frame)
        if frame is not None and frame.ndim >= 2:
            cleaned.append(frame)

    if not cleaned:
        print("[warn] no valid frames collected; skipping video")
        return

    fmt = fmt.lower().strip()
    try:
        if fmt == "mp4":
            imageio.mimsave(str(out_path.with_suffix(".mp4")), cleaned, fps=int(fps), macro_block_size=2)
        else:
            imageio.mimsave(str(out_path.with_suffix(".gif")), cleaned, fps=int(fps))
    except Exception as e:
        print(f"[warn] failed to save video: {type(e).__name__}: {e}")


def _record_video_episodes(
    actor,
    render_env,
    device,
    video_root: Path,
    num_videos: int,
    max_frames: int,
    fps: int,
    fmt: str,
):
    for ep_idx in range(num_videos):
        obs, _ = render_env.reset(seed=42 + ep_idx)
        frames = []
        done = False
        truncated = False

        while not done and not truncated and len(frames) < max_frames:
            frame = render_env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)

            with torch.no_grad():
                actions, _, _ = actor.act(
                    torch.from_numpy(obs).float().to(device),
                    avail_action=torch.from_numpy(render_env.get_avail_actions()).bool().to(device),
                    deterministic=True,
                )

            obs, _, done, truncated, _ = render_env.step(actions.cpu().numpy())

        if len(frames) > 0:
            target_frames = int(3 * fps)
            if len(frames) < target_frames:
                frames.extend([frames[-1]] * (target_frames - len(frames)))

        _maybe_write_video(
            frames,
            video_root / f"eval_ep_{ep_idx}",
            fps=fps,
            fmt=fmt,
        )

if __name__ == "__main__":
    print("[boot] entering main")

    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    kwargs = {}
    render_kwargs = {}

    if args.env_type == "lbf":
        kwargs = {
            "time_limit": args.lbf_time_limit,
            "reward_aggr": args.lbf_reward_aggr,
            "seed": args.seed,
        }

        render_kwargs = {
            "time_limit": args.lbf_time_limit,
            "reward_aggr": args.lbf_reward_aggr,
            "seed": args.seed,
        }

    env = environment(args.env_type, args.env_name, args.env_family, args.agent_ids, kwargs)
    eval_env = environment(args.env_type, args.env_name, args.env_family, args.agent_ids, kwargs)

    render_env = (
        environment(args.env_type, args.env_name, args.env_family, args.agent_ids, render_kwargs)
        if args.eval_save_video
        else None
    )

    print(
        "[sanity] n_agents:",
        env.n_agents,
        "obs_size:",
        env.get_obs_size(),
        "state_size:",
        env.get_state_size(),
        "act_size:",
        env.get_action_size(),
    )

    actor = Actor(
        input_dim=env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=env.get_action_size(),
        activation_name=args.activation,
    ).to(device)

    critic = Critic(
        input_dim=env.get_state_size(),
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        activation_name=args.activation,
    ).to(device)
    
    qcritic = QCritic(
        state_dim=env.get_state_size(),
        joint_action_dim=env.n_agents * env.get_action_size(),
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        activation_name=args.activation,
    ).to(device)

    Optimizer = getattr(optim, args.optimizer)

    if args.optimizer.lower() == "adam":
        actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor, eps=args.adam_eps)
        critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic, eps=args.adam_eps)
        qcritic_optimizer = Optimizer(qcritic.parameters(), lr=args.learning_rate_critic, eps=args.adam_eps)
    else:
        actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
        critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)
        qcritic_optimizer = Optimizer(qcritic.parameters(), lr=args.learning_rate_critic)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"

    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"MAPPO-{run_name}-seed{args.seed}",
        )

    writer = SummaryWriter(f"runs/MAPPO-{run_name}")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    rb = RolloutBuffer(
        buffer_size=args.batch_size,
        obs_space=env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        device=device,
    )

    reward_stats = RunningRewardStats()

    ep_rewards = []
    ep_lengths = []
    ep_stats = []

    training_step = 0
    num_episodes_total = 0
    step = 0

    semantic_total = 0
    semantic_kept = 0

    while step < args.total_timesteps:
        step_at_rollout_start = step

        # -------------------------
        # Collect batch of episodes
        # -------------------------
        for _ in range(args.batch_size):
            if step >= args.total_timesteps:
                break
            
            episode = {
                "obs": [],
                "actions": [],
                "log_prob": [],
                "reward": [],
                "reward_agents": [],
                "states": [],
                "done": [],
                "avail_actions": [],
            }

            obs, _ = env.reset()
            ep_reward = 0.0
            ep_length = 0
            done = False
            truncated = False
            infos = {}

            while not done and not truncated and step < args.total_timesteps:
                avail_action = env.get_avail_actions()
                state = env.get_state()

                obs_t = torch.from_numpy(obs).float().to(device)
                avail_t = torch.from_numpy(avail_action).bool().to(device)

                with torch.no_grad():
                    actions, log_probs, _ = actor.act(obs_t, avail_action=avail_t)

                next_obs, reward, done, truncated, infos = env.step(actions.cpu().numpy())

                reward_scalar = float(np.asarray(reward).reshape(-1)[0])
                reward_agents = np.asarray(
                    infos.get(
                        "reward_agents",
                        np.full((env.n_agents,), reward_scalar / max(1, env.n_agents), dtype=np.float32),
                    ),
                    dtype=np.float32,
                ).reshape(env.n_agents)

                episode["obs"].append(obs)
                episode["actions"].append(actions.cpu().numpy())
                episode["log_prob"].append(log_probs.cpu().numpy())
                episode["reward"].append(np.float32(reward_scalar))
                episode["reward_agents"].append(reward_agents)
                episode["done"].append(np.float32(done or truncated))
                episode["avail_actions"].append(avail_action)
                episode["states"].append(state)

                ep_reward += reward_scalar
                ep_length += 1
                step += 1
                obs = next_obs

            rb.add(episode)

            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            reward_stats.update([ep_reward])
            num_episodes_total += 1

            if args.env_type == "smaclite":
                ep_stats.append(infos)

        # -------------------------
        # Logging rollout
        # -------------------------
        if len(ep_rewards) >= args.log_every:
            writer.add_scalar("rollout/ep_reward", float(np.mean(ep_rewards)), step)
            writer.add_scalar("rollout/ep_length", float(np.mean(ep_lengths)), step)
            writer.add_scalar("rollout/num_episodes", num_episodes_total, step)
            writer.add_scalar("rollout/ep_reward_running_mean", float(reward_stats.reward_mean), step)
            writer.add_scalar("rollout/ep_reward_running_std", float(reward_stats.reward_std), step)

            if args.env_type == "smaclite" and len(ep_stats) > 0:
                writer.add_scalar(
                    "rollout/battle_won",
                    float(np.mean([info["battle_won"] for info in ep_stats])),
                    step,
                )

            ep_rewards.clear()
            ep_lengths.clear()
            ep_stats.clear()

        # -------------------------
        # Build batch
        # -------------------------
        batch = rb.get_batch(normalize_reward=args.normalize_reward)

        if batch is None:
            break

        (
            b_obs,
            b_actions,
            b_log_probs,
            b_reward,
            b_reward_agents,
            b_states,
            b_avail_actions,
            b_done,
            b_mask,
            reward_std_used,
        ) = batch

        B, T, N = b_actions.shape

        # -------------------------
        # TD(lambda) returns and advantages
        # Team critic, team advantage
        # -------------------------
        return_lambda = torch.zeros((B, T), dtype=torch.float32, device=device)
        advantages = torch.zeros((B, T), dtype=torch.float32, device=device)

        with torch.no_grad():
            values = critic(b_states)

            for ep_idx in range(B):
                ep_len = int(b_mask[ep_idx].sum().item())
                last_return = torch.tensor(0.0, device=device)

                for t in reversed(range(ep_len)):
                    if t == ep_len - 1:
                        next_value = torch.tensor(0.0, device=device)
                    else:
                        next_value = values[ep_idx, t + 1]

                    last_return = b_reward[ep_idx, t] + args.gamma * (
                        args.td_lambda * last_return
                        + (1.0 - args.td_lambda) * next_value
                    )

                    return_lambda[ep_idx, t] = last_return
                    advantages[ep_idx, t] = last_return - values[ep_idx, t]

        advantages_unnorm = advantages.clone()

        # -------------------------
        # Counterfactual per-agent score only
        # -------------------------
        cf_advantages_unnorm = None

        if args.cf_advantage_enabled:
            with torch.no_grad():
                actions_onehot = F.one_hot(b_actions, num_classes=env.get_action_size()).float()
                # (B, T, N, A)

                Bcf, Tcf, Ncf, Acf = actions_onehot.shape

                mean_action = actions_onehot.mean(dim=2, keepdim=True)
                avg_actions_all = mean_action.expand(Bcf, Tcf, Ncf, Acf)

                q_avg = qcritic(
                    b_states,
                    avg_actions_all.reshape(Bcf, Tcf, Ncf * Acf),
                )

                cf_scores = torch.zeros((Bcf, Tcf, Ncf), dtype=torch.float32, device=device)

                for agent_i in range(Ncf):
                    mixed_actions = avg_actions_all.clone()
                    mixed_actions[:, :, agent_i, :] = actions_onehot[:, :, agent_i, :]

                    q_i = qcritic(
                        b_states,
                        mixed_actions.reshape(Bcf, Tcf, Ncf * Acf),
                    )

                    cf_scores[:, :, agent_i] = q_i - q_avg

                cf_advantages_unnorm = cf_scores

                valid_cf_mask = b_mask.unsqueeze(-1).expand(Bcf, Tcf, Ncf)
                writer.add_scalar("cf_score/mean", cf_scores[valid_cf_mask].mean().item(), step)
                writer.add_scalar("cf_score/std", cf_scores[valid_cf_mask].std().item(), step)
                writer.add_scalar("cf_score/min", cf_scores[valid_cf_mask].min().item(), step)
                writer.add_scalar("cf_score/max", cf_scores[valid_cf_mask].max().item(), step)
                writer.add_scalar("cf_score/abs_mean", cf_scores[valid_cf_mask].abs().mean().item(), step)

        if args.normalize_advantage:
            valid_adv = advantages[b_mask]
            adv_mean = valid_adv.mean()
            adv_std = torch.clamp(valid_adv.std(), min=1e-6)
            advantages = (advantages - adv_mean) / adv_std

        if args.normalize_return:
            valid_ret = return_lambda[b_mask]
            ret_mean = valid_ret.mean()
            ret_std = torch.clamp(valid_ret.std(), min=1e-6)
            return_lambda = (return_lambda - ret_mean) / ret_std

        # -------------------------
        # Semantic selection
        # -------------------------
        keep_mask_agent = torch.ones((B, T, N), dtype=torch.float32, device=device)
        keep_mask_step = torch.ones((B, T), dtype=torch.float32, device=device)
        semantic_score = torch.zeros((B, T), dtype=torch.float32, device=device)
        score_threshold = None

        if args.semantic_enabled and args.semantic_mode == "advantage":

            if args.cf_advantage_enabled and cf_advantages_unnorm is not None:
                cf_raw = cf_advantages_unnorm.detach()  # (B, T, N)

                # Convert per-agent CF scores into one transition-level score.
                # This matches mappo_continuous: select the whole transition based on
                # the strongest agent contribution.
                if args.adv_use_abs:
                    score = cf_raw.abs().max(dim=-1).values       # (B, T)
                elif args.adv_positive_only:
                    score = torch.clamp(cf_raw, min=0.0).max(dim=-1).values
                else:
                    score = cf_raw.max(dim=-1).values

                valid_score_mask = b_mask  # (B, T)

            else:
                adv_raw = advantages_unnorm.detach()  # (B, T)

                if args.adv_use_abs:
                    score = torch.abs(adv_raw)
                elif args.adv_positive_only:
                    score = torch.clamp(adv_raw, min=0.0)
                else:
                    score = adv_raw

                valid_score_mask = b_mask  # (B, T)

            if step_at_rollout_start < args.adv_warmup_steps:
                # During warmup, keep everything.
                keep_mask_step = torch.ones((B, T), dtype=torch.float32, device=device)
                keep_mask_agent = keep_mask_step.unsqueeze(-1).expand(B, T, N)
                semantic_score = score.float()

            else:
                selected, score_threshold = build_selection_mask(
                    scores=score,
                    valid_mask=valid_score_mask,
                    keep_frac=args.adv_keep_frac,
                    min_keep_frac=args.adv_min_keep_frac,
                    use_abs=args.adv_use_abs,
                    positive_only=args.adv_positive_only,
                )

                # selected is now always step-level: (B, T)
                keep_mask_step = selected.float()
                keep_mask_agent = keep_mask_step.unsqueeze(-1).expand(B, T, N)
                semantic_score = score.float()

        semantic_total += int(b_mask.sum().item())
        semantic_kept += int(((keep_mask_step > 0.5) & b_mask).sum().item())

        # -------------------------
        # PPO update
        # -------------------------
        actor_losses = []
        critic_losses = []
        qcritic_losses = []
        entropies_bonuses = []
        kl_divergences = []
        actor_gradients = []
        critic_gradients = []
        clipped_ratios = []

        for _ in range(args.epochs):
            current_logits = actor.logits(
                x=b_obs,
                avail_action=b_avail_actions,
            )

            current_dist = Categorical(logits=current_logits)
            current_logprob = current_dist.log_prob(b_actions)

            log_ratio = current_logprob - b_log_probs
            ratio = torch.exp(log_ratio)

            adv_agent = advantages.unsqueeze(-1).expand(B, T, N)

            pg_loss1 = adv_agent * ratio
            pg_loss2 = adv_agent * torch.clamp(
                ratio,
                1.0 - args.ppo_clip,
                1.0 + args.ppo_clip,
            )

            pg_loss = -torch.min(pg_loss1, pg_loss2)

            valid_agent_mask = b_mask.unsqueeze(-1).expand(B, T, N)
            actor_weight_mask = valid_agent_mask & (keep_mask_agent > 0.5)

            valid_actor_samples = torch.clamp(actor_weight_mask.float().sum(), min=1.0)
            actor_loss = (pg_loss * actor_weight_mask.float()).sum() / valid_actor_samples

            entropy_bonus = current_dist.entropy()[valid_agent_mask].mean()
            total_actor_loss = actor_loss - args.entropy_coef * entropy_bonus

            current_values = critic(b_states)
            critic_loss_per_step = F.mse_loss(
                current_values,
                return_lambda,
                reduction="none",
            )

            critic_loss = critic_loss_per_step[b_mask].mean()
            
            joint_actions_onehot = F.one_hot(b_actions, num_classes=env.get_action_size()).float()
            joint_actions_onehot = joint_actions_onehot.reshape(B, T, N * env.get_action_size())

            current_q = qcritic(b_states, joint_actions_onehot)
            qcritic_loss_per_step = F.mse_loss(current_q, return_lambda, reduction="none")
            qcritic_loss = qcritic_loss_per_step[b_mask].mean()

            if args.cf_advantage_enabled or args.semantic_enabled:
                total_loss = total_actor_loss + args.value_coef * critic_loss + args.value_coef * qcritic_loss
            else:
                total_loss = total_actor_loss + args.value_coef * critic_loss

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            qcritic_optimizer.zero_grad()

            total_loss.backward()

            actor_gradient = norm_d([p.grad for p in actor.parameters()], 2)
            critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)

            if args.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.clip_gradients)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.clip_gradients)
                torch.nn.utils.clip_grad_norm_(qcritic.parameters(), max_norm=args.clip_gradients)

            actor_optimizer.step()
            critic_optimizer.step()
            qcritic_optimizer.step()

            approx_kl = ((ratio - 1.0) - log_ratio)[valid_agent_mask].mean()
            clipped_ratio = ((ratio - 1.0).abs() > args.ppo_clip)[valid_agent_mask].float().mean()

            training_step += 1

            actor_losses.append(float(actor_loss.detach().cpu().item()))
            critic_losses.append(float(critic_loss.detach().cpu().item()))
            qcritic_losses.append(float(qcritic_loss.detach().cpu().item()))
            entropies_bonuses.append(float(entropy_bonus.detach().cpu().item()))
            kl_divergences.append(float(approx_kl.detach().cpu().item()))
            actor_gradients.append(float(actor_gradient.detach().cpu().item()))
            critic_gradients.append(float(critic_gradient.detach().cpu().item()))
            clipped_ratios.append(float(clipped_ratio.detach().cpu().item()))

        writer.add_scalar("train/critic_loss", float(np.mean(critic_losses)), step)
        writer.add_scalar("train/actor_loss", float(np.mean(actor_losses)), step)
        writer.add_scalar("train/qcritic_loss", float(np.mean(qcritic_losses)), step)
        writer.add_scalar("train/entropy", float(np.mean(entropies_bonuses)), step)
        writer.add_scalar("train/kl_divergence", float(np.mean(kl_divergences)), step)
        writer.add_scalar("train/clipped_ratios", float(np.mean(clipped_ratios)), step)
        writer.add_scalar("train/actor_gradients", float(np.mean(actor_gradients)), step)
        writer.add_scalar("train/critic_gradients", float(np.mean(critic_gradients)), step)
        writer.add_scalar("train/num_updates", training_step, step)

        if reward_std_used is not None:
            writer.add_scalar(
                "train/reward_batch_std_used_for_normalization",
                float(reward_std_used.detach().cpu().item()),
                step,
            )

        if args.semantic_enabled:
            writer.add_scalar("semantic/current_keep_rate", float(keep_mask_step[b_mask].mean().item()), step)
            writer.add_scalar("semantic/buffer_keep_rate", float(keep_mask_step[b_mask].mean().item()), step)
            writer.add_scalar("semantic/step_keep_rate", float(semantic_kept) / max(1, semantic_total), step)
            writer.add_scalar("semantic/num_kept_current", float(((keep_mask_step > 0.5) & b_mask).sum().item()), step)
            writer.add_scalar("semantic/adv_keep_frac_target", float(args.adv_keep_frac), step)
            writer.add_scalar("semantic/adv_score_mean", float(semantic_score[b_mask].mean().item()), step)
            writer.add_scalar("semantic/adv_score_max", float(semantic_score[b_mask].max().item()), step)

            if score_threshold is not None:
                writer.add_scalar("semantic/current_score_threshold", float(score_threshold), step)

        # -------------------------
        # Eval
        # -------------------------
        if (training_step / max(1, args.epochs)) % args.eval_steps == 0:
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []

            for _ in range(args.num_eval_ep):
                eval_obs, _ = eval_env.reset()
                done = False
                truncated = False
                current_reward = 0.0
                current_length = 0
                infos = {}

                while not done and not truncated:
                    with torch.no_grad():
                        actions, _, _ = actor.act(
                            torch.from_numpy(eval_obs).float().to(device),
                            avail_action=torch.from_numpy(eval_env.get_avail_actions()).bool().to(device),
                            deterministic=True,
                        )

                    next_obs, reward, done, truncated, infos = eval_env.step(actions.cpu().numpy())

                    current_reward += float(np.asarray(reward).reshape(-1)[0])
                    current_length += 1
                    eval_obs = next_obs

                eval_ep_reward.append(current_reward)
                eval_ep_length.append(current_length)
                eval_ep_stats.append(infos)

            writer.add_scalar("eval/ep_reward", float(np.mean(eval_ep_reward)), step)
            writer.add_scalar("eval/std_ep_reward", float(np.std(eval_ep_reward)), step)
            writer.add_scalar("eval/ep_length", float(np.mean(eval_ep_length)), step)

            if args.env_type == "smaclite":
                writer.add_scalar(
                    "eval/battle_won",
                    float(np.mean([info["battle_won"] for info in eval_ep_stats])),
                    step,
                )
            
            if args.eval_save_video and render_env is not None:
                video_root = Path(args.eval_video_dir) / run_name / f"step_{step}"
                _record_video_episodes(
                    actor=actor,
                    render_env=render_env,
                    device=device,
                    video_root=video_root,
                    num_videos=args.eval_num_videos_to_save,
                    max_frames=args.eval_video_max_frames,
                    fps=args.eval_video_fps,
                    fmt=args.eval_video_format,
                )

    writer.close()

    if args.use_wnb:
        import wandb
        wandb.finish()
        
    env.close()
    eval_env.close()
    if render_env is not None:
        render_env.close()