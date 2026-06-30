import os
import subprocess
import datetime
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from buffer import ReplayBuffer
from episode_buffer import EpisodeBuffer
from goal_geometry import (world_coords, spin_fraction, spin_thresholds,
                           SPIN_WINDOW, reach_reward, reach_radius_at, distance,
                           eval_step_budget, GOAL_RADIUS)
from motion import MotionState, motion_dim
from models.actor import GaussianActor
from models.critic import TwinQCritic
from task_chain import DEFAULT_CHAIN
from torch.utils.tensorboard.writer import SummaryWriter

# Continuous action dimension: [linear, angular] ∈ [-1, 1]^2
ACTION_DIM = 2
# goal_dim=6: [robot_x, robot_y, goal_x, goal_y, sin(angle), cos(angle)]
GOAL_DIM = 6
GOAL_SCALE = (864., 576., 864., 576., 1., 1.)

# Chain eval reach overrides (match env interaction radii)
from homebot.goals import GOAL_THRESHOLD
_DOOR_REACH  = 47.0
_TRASH_REACH = 31.0
_REACH_OVERRIDE = {"go_to_door": _DOOR_REACH, "collect_trash": _TRASH_REACH}


def process_observation(raw_obs):
    """HxWxC uint8 → CxHxW uint8 tensor (96×96)."""
    img = raw_obs if isinstance(raw_obs, np.ndarray) else raw_obs["observation"]
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(img).permute(2, 0, 1)


class Agent:

    def __init__(self, env: gym.Env,
                 max_buffer_size: int = 200000,
                 goal_layers: int = 2,
                 head_layers: int = 4,
                 use_motion: bool = True,
                 motion_window: int = 8,
                 random_goal_tiles: bool = True,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha_init: float = 0.2,
                 fixed_alpha: float = None) -> None:
        self.env = env
        self.random_goal_tiles = random_goal_tiles
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.gamma = gamma
        self.tau = tau
        self.use_motion = use_motion
        self.motion_window = motion_window

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        raw_obs, _ = self.env.reset()
        obs = process_observation(raw_obs["observation"])
        self.obs_shape = obs.shape   # (3, 96, 96)
        self.frame_skip = getattr(self.env, "_skip", 1)

        self.action_dim = ACTION_DIM
        self.goal_dim   = GOAL_DIM
        self.m_dim = motion_dim(ACTION_DIM, motion_window) if use_motion else 0

        # Actor + twin critics + target critics
        actor_kw = dict(
            action_dim=ACTION_DIM, input_shape=self.obs_shape,
            goal_dim=GOAL_DIM, goal_scale=GOAL_SCALE,
            goal_layers=goal_layers, head_layers=head_layers,
            use_motion=use_motion,
            motion_in_dim=self.m_dim if use_motion else None,
        )
        critic_kw = dict(
            action_dim=ACTION_DIM, input_shape=self.obs_shape,
            goal_dim=GOAL_DIM, goal_scale=GOAL_SCALE,
            goal_layers=goal_layers, head_layers=head_layers,
            use_motion=use_motion,
            motion_in_dim=self.m_dim if use_motion else None,
        )
        self.actor          = GaussianActor(**actor_kw).to(self.device)
        self.critic         = TwinQCritic(**critic_kw).to(self.device)
        self.target_critic  = TwinQCritic(**critic_kw).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self._fixed_alpha = fixed_alpha
        self.target_entropy = -float(ACTION_DIM)
        self.log_alpha = torch.tensor(math.log(alpha_init), dtype=torch.float32,
                                      device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.memory = ReplayBuffer(
            max_size=max_buffer_size,
            input_shape=self.obs_shape,
            action_dim=ACTION_DIM,
            input_device=self.device,
            output_device=self.device,
            goal_dim=GOAL_DIM,
            motion_dim=self.m_dim,
        )
        self.episode_buffer = EpisodeBuffer(action_dim=ACTION_DIM)

        self.total_env_steps = 0
        self.total_grad_steps = 0
        self.updates_per_step = 1
        self.best_chain_score = -1.0

        self._chain_env = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def alpha(self):
        if self._fixed_alpha is not None:
            return self._fixed_alpha
        return self.log_alpha.exp().item()

    def _reset_goal(self, base, desired_goal):
        if not self.random_goal_tiles:
            return desired_goal
        tiles = base._map.valid_floor_tiles()
        col, row = tiles[int(base.np_random.integers(0, len(tiles)))]
        gx, gy = base._map.tile_to_pixel(col, row)
        goal = np.array([float(gx), float(gy)], dtype=np.float32)
        base._desired_goal = goal
        return goal

    def _to_tensor(self, obs, goal, motion):
        obs_t = obs.unsqueeze(0).float().to(self.device) / 255.0
        goal_t = torch.as_tensor(goal, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        motion_t = None
        if self.use_motion:
            motion_t = torch.as_tensor(motion, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
        return obs_t, goal_t, motion_t

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs, goal, motion=None) -> np.ndarray:
        """Stochastic sample from the actor (reparameterized). Returns np.ndarray."""
        obs_t, goal_t, motion_t = self._to_tensor(obs, goal, motion)
        with torch.no_grad():
            action, _, _ = self.actor.sample(obs_t, goal_t, motion_t)
        return action.squeeze(0).cpu().numpy()

    def select_mean_action(self, obs, goal, motion=None) -> np.ndarray:
        """Deterministic mean action tanh(mu) for greedy eval."""
        obs_t, goal_t, motion_t = self._to_tensor(obs, goal, motion)
        with torch.no_grad():
            action = self.actor.mean_action(obs_t, goal_t, motion_t)
        return action.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _polyak_update(self):
        """Soft update target critics: θ_target ← τ*θ + (1-τ)*θ_target."""
        for p, pt in zip(self.critic.parameters(), self.target_critic.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def train_step(self, batch_size: int):
        (obs, actions, rewards, next_obs, dones,
         goals, next_goals, motions, next_motions) = self.memory.sample_buffer(batch_size)

        obs      = obs      / 255.0
        next_obs = next_obs / 255.0

        rewards = rewards.unsqueeze(1)  # (B, 1)
        dones   = dones.unsqueeze(1).float()

        # --- Critic update ---
        alpha = self.alpha  # float: fixed or exp(log_alpha)
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(
                next_obs, next_goals, next_motions)
            q1_target, q2_target = self.target_critic(
                next_obs, next_actions, next_goals, next_motions)
            min_q_target = torch.min(q1_target, q2_target)
            y = rewards + (1 - dones) * self.gamma * (min_q_target - alpha * next_log_pi)

        q1, q2 = self.critic(obs, actions, goals, motions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --- Actor update ---
        new_actions, log_pi, _ = self.actor.sample(obs, goals, motions)
        q1_pi = self.critic.Q1(obs, new_actions, goals, motions)
        actor_loss = (alpha * log_pi - q1_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # --- Alpha update (skipped when alpha is fixed) ---
        if self._fixed_alpha is None:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Polyak update target critics
        self._polyak_update()

        self.total_grad_steps += 1
        return critic_loss.item(), actor_loss.item(), alpha

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def greedy_eval(self, n_episodes: int = 20, stochastic: bool = True) -> float:
        """Single-goal reach rate.

        stochastic=True uses actor.sample() (the SAC deployment policy);
        stochastic=False uses tanh(mu) (deterministic mean, for comparison).
        """
        self.actor.eval()
        successes = 0

        for _ in range(n_episodes):
            raw_obs, _ = self.env.reset()
            obs          = process_observation(raw_obs["observation"])
            desired_goal = raw_obs["desired_goal"]
            base         = self.env.unwrapped
            r            = base._robot
            desired_goal = self._reset_goal(base, desired_goal)
            ms = MotionState(ACTION_DIM, self.motion_window)

            done = False
            ep_reward = 0.0
            while not done:
                goal_vec = world_coords(r.x, r.y, desired_goal[0], desired_goal[1],
                                        r.angle)
                motion = ms.vec(r.x, r.y)
                if stochastic:
                    action = self.select_action(obs, goal_vec, motion)
                else:
                    action = self.select_mean_action(obs, goal_vec, motion)
                prev_x, prev_y = r.x, r.y
                ms.commit(r.x, r.y, action)
                raw_next, reward, term, trunc, _ = self.env.step(action)
                obs = process_observation(raw_next["observation"])
                ep_reward += float(reward)
                done = term or trunc

            if ep_reward > 0.5:
                successes += 1

        self.actor.train()
        return successes / n_episodes

    def _run_chain_leg(self, obs, env, base, goal_xy, budget, ms):
        """Drive toward goal_xy. Returns (reached, steps, obs, positions)."""
        robot = base._robot
        positions = [(robot.x, robot.y)]
        steps = 0
        reach_r = _REACH_OVERRIDE.get("__this_leg__", GOAL_THRESHOLD)  # caller sets this
        while steps < budget:
            goal_vec = world_coords(robot.x, robot.y, goal_xy[0], goal_xy[1],
                                    robot.angle)
            motion = ms.vec(robot.x, robot.y)
            action = self.select_action(obs, goal_vec, motion)
            ms.commit(robot.x, robot.y, action)
            raw_next, _, term, trunc, _ = env.step(action)
            obs = process_observation(raw_next["observation"])
            positions.append((robot.x, robot.y))
            steps += 1
            if distance(robot.x, robot.y, goal_xy[0], goal_xy[1]) <= reach_r:
                return True, steps, obs, positions
            if term or trunc:
                break
        return False, steps, obs, positions

    def chain_eval(self, n_episodes: int = 10):
        """The real deployment metric: chained task score.

        Returns (mean_score, full_chain_rate, mean_spin).
        """
        from task_chain import resolve_goal, world_state, leg_succeeded

        if self._chain_env is None:
            self._chain_env = gym.make(
                "HomeBot2D-V1", render_mode="rgb_array",
                action_mode="continuous",
                obs_resolution=(96, 96), n_trash=2, max_steps=20000,
                map_name="default", random_start=True,
            )
            if self.frame_skip > 1:
                from env_wrappers import FrameSkipWrapper
                self._chain_env = FrameSkipWrapper(self._chain_env, skip=self.frame_skip)

        self.actor.eval()
        n_legs = len(DEFAULT_CHAIN)
        move_min, net_max = spin_thresholds(SPIN_WINDOW)
        total, full = 0, 0
        spins = []

        for seed in range(n_episodes):
            base = self._chain_env.unwrapped
            raw_obs, _ = self._chain_env.reset(seed=seed)
            obs = process_observation(raw_obs)
            ms = MotionState(ACTION_DIM, self.motion_window)

            targets = [(name, resolve_goal(base, name)) for name in DEFAULT_CHAIN]
            ep_legs = []
            for name, (gx, gy) in targets:
                robot = base._robot
                skip = getattr(self._chain_env, "_skip", 1)
                if name == "collect_trash":
                    budget = 600 // skip
                else:
                    budget = max(1, eval_step_budget(
                        distance(robot.x, robot.y, gx, gy))) // skip
                reach = _REACH_OVERRIDE.get(name, GOAL_THRESHOLD)

                # Temporarily override reach for the leg helper
                positions = [(robot.x, robot.y)]
                steps = 0
                reached = False
                before = world_state(base)
                while steps < budget:
                    goal_vec = world_coords(robot.x, robot.y, gx, gy, robot.angle)
                    motion = ms.vec(robot.x, robot.y)
                    action = self.select_action(obs, goal_vec, motion)
                    ms.commit(robot.x, robot.y, action)
                    raw_next, _, term, trunc, _ = self._chain_env.step(action)
                    obs = process_observation(raw_next)
                    positions.append((robot.x, robot.y))
                    steps += 1
                    if distance(robot.x, robot.y, gx, gy) <= reach:
                        reached = True
                        break
                    if term or trunc:
                        break
                after = world_state(base)
                leg_ok = leg_succeeded(name, before, after, reached)
                ep_legs.append((name, leg_ok, steps, positions))

            reached_count = sum(1 for _, r, *_ in ep_legs if r)
            total += reached_count
            if reached_count == n_legs:
                full += 1
            spins.extend(
                spin_fraction(pos, SPIN_WINDOW, move_min, net_max)
                for _, _, _, pos in ep_legs
            )

        self.actor.train()
        mean_spin = sum(spins) / len(spins) if spins else 0.0
        return total / n_episodes, full / n_episodes, mean_spin

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save(self):
        self.actor.save_the_model("actor", verbose=True)

    def load(self):
        self.actor.load_the_model("actor", device=self.device)

    def save_best(self, episode, chain_score):
        path = "checkpoints/best.pt"
        torch.save({
            "actor":       self.actor.state_dict(),
            "critic":      self.critic.state_dict(),
            "log_alpha":   self.log_alpha.item(),
            "episode":     int(episode),
            "chain_score": float(chain_score),
        }, path)
        print(f"  New best checkpoint saved (episode={episode}, chain_score={chain_score:.2f})")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, episodes=1000, batch_size=256, run_tag=None,
              eval_interval=50, eval_episodes=20, chain_eval_interval=10,
              her_anneal_start=None,
              reach_start=None, reach_end=None,
              reach_anneal_start=0, reach_anneal_end=None):

        use_reach_curriculum = reach_start is not None
        if use_reach_curriculum and reach_anneal_end is None:
            reach_anneal_end = episodes

        if run_tag is None:
            try:
                refs = subprocess.check_output(
                    ['git', 'for-each-ref', '--format=%(refname:short)',
                     '--points-at', 'HEAD', 'refs/remotes/origin/'],
                    stderr=subprocess.DEVNULL).decode().strip()
                if refs:
                    run_tag = refs.splitlines()[0].replace('origin/', '')
                if not run_tag:
                    run_tag = subprocess.check_output(
                        ['git', 'branch', '--show-current'],
                        stderr=subprocess.DEVNULL).decode().strip()
                if not run_tag:
                    run_tag = 'unknown'
            except Exception:
                run_tag = 'unknown'

        writer = SummaryWriter(
            f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}')

        for episode in range(episodes):
            raw_obs, _   = self.env.reset()
            obs          = process_observation(raw_obs["observation"])
            desired_goal = raw_obs["desired_goal"]
            base         = self.env.unwrapped
            r            = base._robot
            desired_goal = self._reset_goal(base, desired_goal)
            ms = MotionState(ACTION_DIM, self.motion_window)

            if use_reach_curriculum:
                reach_radius = reach_radius_at(episode, reach_start, reach_end,
                                               reach_anneal_start, reach_anneal_end)
                writer.add_scalar('Train/reach_radius', reach_radius, episode)

            done = False
            episode_reward = 0.0
            episode_critic_loss = 0.0
            episode_actor_loss  = 0.0
            episode_steps = 0

            while not done:
                angle_prev   = r.angle
                pos_prev     = np.array([r.x, r.y], dtype=np.float32)
                goal_vec     = world_coords(r.x, r.y, desired_goal[0], desired_goal[1],
                                            r.angle)
                motion_prev  = ms.vec(r.x, r.y)

                action = self.select_action(obs, goal_vec, motion_prev)
                ms.commit(r.x, r.y, action)

                raw_next, env_reward, env_term, trunc, _ = self.env.step(action)
                pos_next   = np.array([r.x, r.y], dtype=np.float32)
                angle_next = r.angle

                if use_reach_curriculum:
                    reward = float(reach_reward(pos_next, desired_goal, reach_radius))
                    term   = reward > 0.5
                else:
                    reward, term = float(env_reward), bool(env_term)

                next_obs    = process_observation(raw_next["observation"])
                motion_next = ms.vec(pos_next[0], pos_next[1])
                # Store term (not trunc): timeout is not a terminal state.
                self.episode_buffer.store(
                    obs, action, reward, next_obs, term,
                    achieved_prev=pos_prev,
                    achieved_next=pos_next,
                    angle_prev=angle_prev,
                    angle_next=angle_next,
                    motion_prev=motion_prev,
                    motion_next=motion_next,
                )

                episode_reward += reward
                episode_steps  += 1
                self.total_env_steps += 1
                obs = next_obs
                done = term or trunc

                # UTD = 1 update per env step
                for _ in range(self.updates_per_step):
                    if self.memory.can_sample(batch_size):
                        c_loss, a_loss, _ = self.train_step(batch_size)
                        episode_critic_loss += c_loss
                        episode_actor_loss  += a_loss

            # HER
            k_eff = self.episode_buffer.K
            if her_anneal_start is not None and episode >= her_anneal_start:
                span = max(1, episodes - her_anneal_start)
                frac = min(1.0, (episode - her_anneal_start) / span)
                k_eff = self.episode_buffer.K * (1.0 - frac)

            her_reward = (
                (lambda a, d, info: reach_reward(a, d, reach_radius))
                if use_reach_curriculum
                else self.env.unwrapped.compute_reward)  # type: ignore[attr-defined]
            self.episode_buffer.send_to(
                self.memory,
                desired_goal=desired_goal,
                compute_reward=her_reward,
                k=k_eff,
            )
            self.episode_buffer.clear()
            writer.add_scalar("Train/hindsight_k", k_eff, episode)

            avg_c = episode_critic_loss / max(1, episode_steps)
            avg_a = episode_actor_loss  / max(1, episode_steps)
            alpha  = self.alpha
            print(f"Episode {episode} | reward: {episode_reward:.1f} | "
                  f"steps: {episode_steps} | alpha: {alpha:.4f}")

            writer.add_scalar("Train/episode_reward",    episode_reward,  episode)
            writer.add_scalar("Train/avg_critic_loss",   avg_c,           episode)
            writer.add_scalar("Train/avg_actor_loss",    avg_a,           episode)
            writer.add_scalar("Train/alpha",             alpha,           episode)
            writer.add_scalar("Train/episode_steps",     episode_steps,   episode)
            writer.add_scalar("Train/total_env_steps",   self.total_env_steps, episode)
            writer.add_scalar("Buffer/fill",
                              min(self.memory.mem_ctr, self.memory.mem_size), episode)

            if episode % 10 == 0:
                self.save()

            if episode % eval_interval == 0:
                stoch_rate = self.greedy_eval(n_episodes=eval_episodes, stochastic=True)
                det_rate   = self.greedy_eval(n_episodes=eval_episodes, stochastic=False)
                writer.add_scalar("Eval/reach_stochastic",   stoch_rate, episode)
                writer.add_scalar("Eval/reach_deterministic", det_rate,   episode)
                print(f"  [Eval] episode {episode}: stoch_reach={stoch_rate:.3f}  "
                      f"det_reach={det_rate:.3f}")

            if episode % chain_eval_interval == 0:
                chain_score, chain_full, chain_spin = self.chain_eval()
                writer.add_scalar("Eval/chain_score",        chain_score, episode)
                writer.add_scalar("Eval/chain_full",         chain_full,  episode)
                writer.add_scalar("Eval/chain_spin_fraction", chain_spin, episode)
                print(f"  [Chain] episode {episode}: score={chain_score:.2f}/"
                      f"{len(DEFAULT_CHAIN)} | full_chain={chain_full:.2f} | "
                      f"spin={chain_spin:.3f}")

                if chain_score > self.best_chain_score:
                    self.best_chain_score = chain_score
                    self.save_best(episode, chain_score)
