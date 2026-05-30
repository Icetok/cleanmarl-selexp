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
    positive_only=False,
):
    flat_scores = scores.reshape(-1)
    flat_valid = valid_mask.reshape(-1)

    eligible_mask = flat_valid.clone()
    if positive_only:
        eligible_mask = eligible_mask & (flat_scores > 0.0)

    eligible_idx = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)

    total_valid = int(flat_valid.sum().item())
    min_keep = max(1, int(float(min_keep_frac) * total_valid))

    keep_flat = torch.zeros_like(flat_scores, dtype=torch.float32)

    if eligible_idx.numel() < min_keep:
        keep_flat[flat_valid] = 1.0
        return keep_flat.reshape_as(scores), None

    eligible_scores = flat_scores[eligible_idx]

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


if __name__ == "__main__":
    print("[boot] entering main")

    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    kwargs = {}
    if args.env_type == "lbf":
        kwargs = {
            "time_limit": args.lbf_time_limit,
            "reward_aggr": args.lbf_reward_aggr,
            "seed": args.seed,
        }

    env = environment(args.env_type, args.env_name, args.env_family, args.agent_ids, kwargs)
    eval_env = environment(args.env_type, args.env_name, args.env_family, args.agent_ids, kwargs)

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

    Optimizer = getattr(optim, args.optimizer)

    if args.optimizer.lower() == "adam":
        actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor, eps=args.adam_eps)
        critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic, eps=args.adam_eps)
    else:
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
            reward_sum = b_reward_agents.sum(dim=-1, keepdim=True)

            if env.n_agents > 1:
                other_mean = (reward_sum - b_reward_agents) / float(env.n_agents - 1)
            else:
                other_mean = torch.zeros_like(b_reward_agents)

            cf_rewards = b_reward_agents - other_mean

            cf_advantages = torch.zeros_like(cf_rewards, device=device)
            last_cf = torch.zeros((B, N), dtype=torch.float32, device=device)

            for t in reversed(range(T)):
                active = b_mask[:, t].float().unsqueeze(-1)
                last_cf = cf_rewards[:, t] + args.gamma * args.td_lambda * active * last_cf
                cf_advantages[:, t] = last_cf * active

            cf_advantages_unnorm = cf_advantages.clone()

        if args.normalize_advantage:
            valid_adv = advantages[b_mask]
            adv_mean = valid_adv.mean()
            adv_std = torch.clamp(valid_adv.std(), min=1e-6)
            advantages = (advantages - adv_mean) / adv_std
            
        cf_advantages_norm = None

        if args.cf_advantage_enabled and cf_advantages_unnorm is not None:
            valid_cf_mask = b_mask.unsqueeze(-1).expand(B, T, N)
            valid_cf = cf_advantages_unnorm[valid_cf_mask]

            cf_mean = valid_cf.mean()
            cf_std = torch.clamp(valid_cf.std(), min=1e-6)

            cf_advantages_norm = (cf_advantages_unnorm - cf_mean) / cf_std

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
                score_raw = cf_advantages_unnorm.detach()
                valid_score_mask = b_mask.unsqueeze(-1).expand(B, T, N)
            else:
                score_raw = advantages_unnorm.detach()
                valid_score_mask = b_mask

            if args.adv_use_abs:
                score = torch.abs(score_raw)
            elif args.adv_positive_only:
                score = torch.clamp(score_raw, min=0.0)
            else:
                score = score_raw

            if step_at_rollout_start < args.adv_warmup_steps:
                pass
            else:
                selected, score_threshold = build_selection_mask(
                    scores=score,
                    valid_mask=valid_score_mask,
                    keep_frac=args.adv_keep_frac,
                    min_keep_frac=args.adv_min_keep_frac,
                    positive_only=args.adv_positive_only,
                )

                if selected.ndim == 3:
                    keep_mask_agent = selected.float()
                    keep_mask_step = keep_mask_agent.mean(dim=-1)
                    semantic_score = score.mean(dim=-1)
                else:
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
            
            if args.cf_advantage_enabled and cf_advantages_norm is not None:
                adv_agent = cf_advantages_norm
            else:
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

            approx_kl = ((ratio - 1.0) - log_ratio)[valid_agent_mask].mean()
            clipped_ratio = ((ratio - 1.0).abs() > args.ppo_clip)[valid_agent_mask].float().mean()

            training_step += 1

            actor_losses.append(float(actor_loss.detach().cpu().item()))
            critic_losses.append(float(critic_loss.detach().cpu().item()))
            entropies_bonuses.append(float(entropy_bonus.detach().cpu().item()))
            kl_divergences.append(float(approx_kl.detach().cpu().item()))
            actor_gradients.append(float(actor_gradient.detach().cpu().item()))
            critic_gradients.append(float(critic_gradient.detach().cpu().item()))
            clipped_ratios.append(float(clipped_ratio.detach().cpu().item()))

        writer.add_scalar("train/critic_loss", float(np.mean(critic_losses)), step)
        writer.add_scalar("train/actor_loss", float(np.mean(actor_losses)), step)
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

    writer.close()

    if args.use_wnb:
        import wandb
        wandb.finish()

    env.close()
    eval_env.close()