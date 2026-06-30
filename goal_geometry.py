"""Geometry helpers for the SAC reacher eval.

Stateless and unit-testable without a gym import.
"""
import math
import numpy as np

# Env trash pickup: robot.RADIUS(15) + tile_size(32) * _TRASH_RANGE(0.5) = 31 px.
GOAL_RADIUS = 31.0
ROBOT_STEP_PX = 4.0      # homebot DISCRETE_SPEED (also max continuous displacement per step)
EVAL_BUDGET_MULT = 3

# Spinning / limit-cycle detection
SPIN_WINDOW = 8


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(bx - ax, by - ay)


def reach_reward(achieved, desired, radius):
    """Sparse 0/1 reach reward at a parametric radius."""
    diff = np.asarray(achieved, dtype=np.float32) - np.asarray(desired, dtype=np.float32)
    dist = np.linalg.norm(diff, axis=-1)
    return (dist <= radius).astype(np.float32)


def reach_radius_at(episode, start, end, anneal_start, anneal_end):
    if episode <= anneal_start:
        return start
    if episode >= anneal_end:
        return end
    frac = (episode - anneal_start) / max(1, anneal_end - anneal_start)
    return start + (end - start) * frac


def spin_thresholds(window: int = SPIN_WINDOW):
    return 0.5 * window * ROBOT_STEP_PX, 2.0 * ROBOT_STEP_PX


def spin_fraction(positions, window, move_min, net_max):
    """Fraction of steps inside a 'moving but not progressing' loop."""
    n = len(positions)
    if n <= window:
        return 0.0

    max_history = 3 * window
    min_history = max(3, window // 2)

    spin = 0
    for t in range(window, n):
        current_path = 0.0
        is_spinning = False
        start_idx = max(0, t - max_history)
        end_idx = t - min_history

        for i in range(t - 1, start_idx - 1, -1):
            step_dist = distance(positions[i][0], positions[i][1],
                                 positions[i + 1][0], positions[i + 1][1])
            current_path += step_dist

            if i <= end_idx:
                net = distance(positions[i][0], positions[i][1],
                               positions[t][0], positions[t][1])
                if current_path >= move_min and net <= net_max:
                    is_spinning = True
                    break

        if is_spinning:
            spin += 1

    return spin / (n - window)


def world_coords(rx: float, ry: float, gx: float, gy: float,
                 angle: float = 0.0) -> np.ndarray:
    """Goal state: [robot_x, robot_y, goal_x, goal_y, sin(angle), cos(angle)].

    Continuous SAC needs explicit heading because the action is (linear, angular)
    in the robot frame — without heading the Q(s,a)/policy mapping is non-Markovian
    from (image, goal_xy) alone. Sin/cos avoids angle wraparound.
    """
    return np.array([rx, ry, gx, gy, math.sin(angle), math.cos(angle)],
                    dtype=np.float32)


def eval_step_budget(init_dist: float) -> int:
    return EVAL_BUDGET_MULT * max(1, math.ceil(max(init_dist, GOAL_RADIUS) / ROBOT_STEP_PX))
