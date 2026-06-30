"""Previous-motion features for the SAC actor/critic (anti-oscillation / spin detection).

Continuous SAC variant: the last action is a 2D float vector [linear, angular]
instead of a discrete one-hot. The rest — per-step velocity and windowed net
displacement — carries over directly from the Q-DQN motion module.

motion_dim(2, window=8) = 2 + 2 + 2 = 6:
  [last_linear | last_angular | dx/step | dy/step | net_dx/(W*step) | net_dy/(W*step)]

State-intrinsic, so HER never relabels it.
"""
from collections import deque

import numpy as np

from goal_geometry import ROBOT_STEP_PX

MOTION_WINDOW = 8


def motion_dim(action_dim: int, window: int = 1) -> int:
    """Total motion vector width. action_dim = size of continuous action (2 for SAC)."""
    return action_dim + 2 + (2 if window > 1 else 0)


def make_motion(action_dim, last_action, dx, dy, net_dx=0.0, net_dy=0.0,
                step=ROBOT_STEP_PX, window=1):
    """[ last_action | dx/step | dy/step | net_dx/(W*step) | net_dy/(W*step) ]

    last_action is a float array of shape (action_dim,); None -> zeros.
    Velocity normalized by the max step so it sits in ~[-1, 1] at full speed.
    Net displacement normalized by window*step so a straight run reads ~1,
    a closed cycle ~0.
    """
    m = np.zeros(motion_dim(action_dim, window), dtype=np.float32)
    if last_action is not None:
        m[:action_dim] = np.asarray(last_action, dtype=np.float32)
    m[action_dim]     = dx / step
    m[action_dim + 1] = dy / step
    if window > 1:
        m[action_dim + 2] = net_dx / (window * step)
        m[action_dim + 3] = net_dy / (window * step)
    return m


class MotionState:
    """Per-rollout tracker for continuous actions.

    Usage each step:
        motion = ms.vec(x, y)           # feature at current state
        action = actor.sample(obs, goal, motion)
        ms.commit(x, y, action)         # record pose + action
        raw_next, ... = env.step(action)
    """

    def __init__(self, action_dim: int, window: int = 1):
        self.adim = action_dim
        self.window = window
        self.reset()

    def reset(self):
        self.last_action = None
        self.prev = None
        self.history = deque(maxlen=max(1, self.window))

    def vec(self, x, y):
        if self.prev is None:
            dx = dy = 0.0
        else:
            dx, dy = x - self.prev[0], y - self.prev[1]
        if self.window > 1 and self.history:
            ox, oy = self.history[0]
            net_dx, net_dy = x - ox, y - oy
        else:
            net_dx = net_dy = 0.0
        return make_motion(self.adim, self.last_action, dx, dy,
                           net_dx, net_dy, window=self.window)

    def commit(self, x, y, action):
        """action: np.ndarray of shape (action_dim,)"""
        self.history.append((x, y))
        self.prev = (x, y)
        self.last_action = np.asarray(action, dtype=np.float32)
