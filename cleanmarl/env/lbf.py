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
    
    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            return self._render_rgb_array_manual()

        try:
            return self.env.render()
        except Exception as e:
            if not hasattr(self, "_render_warned"):
                self._render_warned = True
                print(f"[lbf_render] render failed: {type(e).__name__}: {e}")
            return None
        
    def _render_rgb_array_manual(self, cell_size=50):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from importlib import resources

        env = self.env.unwrapped
        rows, cols = env.field.shape

        grid_line = 1
        width = cols * (cell_size + grid_line)
        height = rows * (cell_size + grid_line)

        # make MP4/libx264-safe
        if width % 2 != 0:
            width += 1
        if height % 2 != 0:
            height += 1

        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # grid
        for r in range(rows + 1):
            y = r * (cell_size + grid_line)
            draw.line([(0, y), (width, y)], fill=(0, 0, 0), width=1)

        for c in range(cols + 1):
            x = c * (cell_size + grid_line)
            draw.line([(x, 0), (x, height)], fill=(0, 0, 0), width=1)

        # load real LBF icons
        try:
            icon_root = resources.files("lbforaging.foraging.icons")
            apple = Image.open(icon_root / "apple.png").convert("RGBA")
            agent_icon = Image.open(icon_root / "agent.png").convert("RGBA")
        except Exception as e:
            print(f"[lbf_render] could not load icons, using fallback shapes: {e}")
            apple = None
            agent_icon = None

        try:
            font = ImageFont.truetype("Times New Roman.ttf", 12)
        except Exception:
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
            except Exception:
                font = ImageFont.load_default()

        def paste_icon(icon, row, col):
            x = col * (cell_size + grid_line) + 1
            y = row * (cell_size + grid_line) + 1

            if icon is None:
                draw.rectangle(
                    [(x + 10, y + 10), (x + cell_size - 10, y + cell_size - 10)],
                    fill=(220, 60, 60),
                )
                return

            icon_resized = icon.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            img.paste(icon_resized, (x, y), icon_resized)

        def draw_badge(row, col, level):
            radius = cell_size / 5
            cx = col * (cell_size + grid_line) + 0.75 * (cell_size + grid_line)
            cy = row * (cell_size + grid_line) + 0.75 * (cell_size + grid_line)

            draw.ellipse(
                [(cx - radius, cy - radius), (cx + radius, cy + radius)],
                fill=(255, 255, 255),
                outline=(0, 0, 0),
                width=2,
            )

            text = str(int(level))
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((cx - tw / 2, cy - th / 2 - 1), text, fill=(0, 0, 0), font=font)

        # draw food from env.field, not env.food
        for row, col in zip(*env.field.nonzero()):
            level = env.field[row, col]
            paste_icon(apple, row, col)
            draw_badge(row, col, level)

        # draw agents
        for player in env.players:
            row, col = player.position
            paste_icon(agent_icon, row, col)
            draw_badge(row, col, player.level)

        return np.asarray(img, dtype=np.uint8)

    def close(self):
        self.env.close()