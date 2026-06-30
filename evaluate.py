"""Standalone greedy eval for a saved SAC checkpoint.

Usage:
    python3 evaluate.py
    python3 evaluate.py --checkpoint checkpoints/best.pt --episodes 50 --stochastic
"""
import argparse
import cv2
import numpy as np
import torch
import gymnasium as gym
import homebot  # noqa: F401

from goal_geometry import world_coords
from models.actor import GaussianActor
from motion import MotionState
from agent import ACTION_DIM, GOAL_DIM, GOAL_SCALE

MOTION_DIM = 6  # motion_dim(ACTION_DIM=2, window=8)


def process_observation(raw):
    img = raw if isinstance(raw, np.ndarray) else raw["observation"]
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(img).permute(2, 0, 1)


def load_actor(path, device, goal_layers=2, head_layers=4, use_motion=True):
    actor = GaussianActor(
        action_dim=ACTION_DIM, input_shape=(3, 96, 96),
        goal_dim=GOAL_DIM, goal_scale=GOAL_SCALE,
        goal_layers=goal_layers, head_layers=head_layers,
        use_motion=use_motion, motion_in_dim=MOTION_DIM if use_motion else None,
    ).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state_dict = ckpt.get("actor", ckpt)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--goal-layers", type=int, default=2)
    parser.add_argument("--head-layers", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample stochastically (default: deterministic mean)")
    parser.add_argument("--frame-skip", type=int, default=2)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    actor = load_actor(args.checkpoint, device, args.goal_layers, args.head_layers)

    env = gym.make(
        "HomeBot2D-Goal-V1", render_mode="rgb_array", action_mode="continuous",
        obs_resolution=(96, 96), n_trash=1, max_steps=1000,
        map_name="default", goals=["collect_trash"], random_start=True,
    )
    if args.frame_skip > 1:
        from env_wrappers import FrameSkipWrapper
        env = FrameSkipWrapper(env, skip=args.frame_skip)

    successes = 0
    for ep in range(args.episodes):
        raw_obs, _ = env.reset()
        obs = process_observation(raw_obs["observation"])
        desired_goal = raw_obs["desired_goal"]
        base = env.unwrapped
        r = base._robot
        ms = MotionState(ACTION_DIM, 8)
        done = False
        ep_reward = 0.0
        while not done:
            goal_vec = world_coords(r.x, r.y, desired_goal[0], desired_goal[1], r.angle)
            motion = ms.vec(r.x, r.y)
            obs_t = obs.unsqueeze(0).float().to(device) / 255.0
            goal_t = torch.as_tensor(goal_vec, dtype=torch.float32, device=device).unsqueeze(0)
            motion_t = torch.as_tensor(motion, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if args.stochastic:
                    action, _, _ = actor.sample(obs_t, goal_t, motion_t)
                else:
                    action = actor.mean_action(obs_t, goal_t, motion_t)
            action_np = action.squeeze(0).cpu().numpy()
            ms.commit(r.x, r.y, action_np)
            raw_next, reward, term, trunc, _ = env.step(action_np)
            obs = process_observation(raw_next["observation"])
            ep_reward += float(reward)
            done = term or trunc
        if ep_reward > 0.5:
            successes += 1
        print(f"Episode {ep}: reward={ep_reward:.1f}")

    rate = successes / args.episodes
    print(f"\nReach rate: {successes}/{args.episodes} = {rate:.3f}")
    env.close()


if __name__ == "__main__":
    main()
