import math
from dataclasses import dataclass
from typing import Callable
import random
import numpy as np
import torch

from goal_geometry import world_coords

BLOCKED_EPS = 0.5        # px; below this the step made no progress (true pin)
BLOCKED_PENALTY = -0.10
SPIN_PENALTY = -0.10


def _blocked_penalty(t) -> float:
    moved = float(np.linalg.norm(t.achieved_next - t.achieved_prev))
    return BLOCKED_PENALTY if moved < BLOCKED_EPS else 0.0


def _spin_penalty(t, step_idx, action_dim=2, window=8) -> float:
    """SPIN_PENALTY if the robot is moving but has very low net displacement
    over the window. action_dim is the continuous action dimension (default 2)."""
    if t.motion_prev is None or len(t.motion_prev) != action_dim + 4:
        return 0.0
    moved = float(np.linalg.norm(t.achieved_next - t.achieved_prev))
    if moved < BLOCKED_EPS:
        return 0.0

    # Net displacement stored at [action_dim+2, action_dim+3], normalized.
    ndx = t.motion_prev[action_dim + 2]
    ndy = t.motion_prev[action_dim + 3]
    net_disp = float(math.sqrt(ndx**2 + ndy**2))

    if step_idx >= window and net_disp < 0.25:
        return SPIN_PENALTY
    return 0.0


@dataclass
class Transition:
    obs:           torch.Tensor
    action:        np.ndarray   # continuous action [linear, angular]
    reward:        float
    next_obs:      torch.Tensor
    done:          bool
    achieved_prev: np.ndarray
    achieved_next: np.ndarray
    angle_prev:    float = 0.0  # robot heading at obs (for goal encoding)
    angle_next:    float = 0.0  # robot heading at next_obs
    motion_prev:   np.ndarray | None = None
    motion_next:   np.ndarray | None = None


class EpisodeBuffer:
    """Caches one episode's transitions for HER relabeling.

    Goal encoding includes heading: [robot_x, robot_y, goal_x, goal_y,
    sin(angle), cos(angle)]. The heading is part of the goal vector because the
    continuous action (linear, angular) is in the robot frame — without heading
    the Q(s,a) mapping is non-Markovian from image+coords alone.
    """

    K = 2  # hindsight goals per transition (future strategy)

    def __init__(self, action_dim: int = 2):
        self._transitions: list[Transition] = []
        self._action_dim = action_dim

    def store(self, obs, action, reward, next_obs, done,
              achieved_prev, achieved_next,
              angle_prev=0.0, angle_next=0.0,
              motion_prev=None, motion_next=None):
        self._transitions.append(Transition(
            obs=obs,
            action=np.asarray(action, dtype=np.float32),
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
            achieved_prev=achieved_prev,
            achieved_next=achieved_next,
            angle_prev=float(angle_prev),
            angle_next=float(angle_next),
            motion_prev=motion_prev,
            motion_next=motion_next,
        ))

    def __len__(self):
        return len(self._transitions)

    def clear(self):
        self._transitions.clear()

    def send_to(
        self,
        replay_buffer,
        desired_goal: np.ndarray,
        compute_reward: Callable,
        k: float | None = None,
        n_step: int = 1,
        gamma: float = 0.99,
    ) -> None:
        """Write original transitions then k hindsight-relabeled copies.

        `k` is the (possibly fractional) hindsight count; defaults to class K.
        Stochastically rounded so a fractional k hits the target ratio in expectation.

        `n_step` windows up to n consecutive transitions into one multi-step return,
        stopping early at episode end (pass 1) or at a positive relabeled reward
        (pass 2 — the HER done-fix: a window must stop the instant it hits the
        relabeled goal, or it would bootstrap past it). The per-transition bootstrap
        discount (gamma**n_eff) is stored alongside the transition since n_eff varies.
        n_step=1 reduces exactly to the original single-step behavior.
        """
        dg = desired_goal
        k = self.K if k is None else k
        adim = self._action_dim
        T = self._transitions

        # Pass 1: original transitions, windowed against the true desired goal
        for i in range(len(T)):
            t = T[i]
            R, discount, n_eff, done_flag = 0.0, 1.0, 0, False
            for j in range(i, min(i + n_step, len(T))):
                tj = T[j]
                r_j = tj.reward + _blocked_penalty(tj) + _spin_penalty(tj, j, adim)
                R += discount * r_j
                discount *= gamma
                n_eff += 1
                if tj.done:
                    done_flag = True
                    break
            last = T[i + n_eff - 1]
            goal_at_s  = world_coords(t.achieved_prev[0], t.achieved_prev[1],
                                      dg[0], dg[1], t.angle_prev)
            goal_at_sp = world_coords(last.achieved_next[0], last.achieved_next[1],
                                      dg[0], dg[1], last.angle_next)
            replay_buffer.store_transition(
                t.obs, t.action, R, last.next_obs, done_flag,
                goal_at_s, goal_at_sp,
                motion=t.motion_prev, next_motion=last.motion_next,
                discount=gamma ** n_eff,
            )

        # Pass 2: hindsight transitions, windowed against a fixed relabeled goal
        for i in range(len(T)):
            t = T[i]
            future = T[i + 1:]
            if not future:
                continue
            kk = int(k) + (1 if random.random() < (k - int(k)) else 0)
            kk = min(kk, len(future))
            if kk <= 0:
                continue
            for hg_t in random.sample(future, kk):
                hindsight_goal = hg_t.achieved_next

                R, discount, n_eff, done_flag = 0.0, 1.0, 0, False
                for j in range(i, min(i + n_step, len(T))):
                    tj = T[j]
                    hindsight_reward = float(compute_reward(
                        tj.achieved_next[np.newaxis],
                        hindsight_goal[np.newaxis],
                        {},
                    )[0])
                    r_j = hindsight_reward + _blocked_penalty(tj) + _spin_penalty(tj, j, adim)
                    R += discount * r_j
                    discount *= gamma
                    n_eff += 1
                    # HER done-fix: stop the instant the relabeled goal is hit so
                    # the target doesn't bootstrap past it (the Q-DQN regression bug).
                    if hindsight_reward > 0.5:
                        done_flag = True
                        break
                last = T[i + n_eff - 1]
                hs_goal_at_s  = world_coords(t.achieved_prev[0], t.achieved_prev[1],
                                             hindsight_goal[0], hindsight_goal[1],
                                             t.angle_prev)
                hs_goal_at_sp = world_coords(last.achieved_next[0], last.achieved_next[1],
                                             hindsight_goal[0], hindsight_goal[1],
                                             last.angle_next)
                replay_buffer.store_transition(
                    t.obs, t.action, R, last.next_obs, done_flag,
                    hs_goal_at_s, hs_goal_at_sp,
                    motion=t.motion_prev, next_motion=last.motion_next,
                    discount=gamma ** n_eff,
                )
