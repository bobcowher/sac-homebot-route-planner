"""Local smoke test for the chain-training loop (CPU-safe, ~1 min).

Exercises: base-env multi-goal train loop + per-leg HER flush, greedy_eval,
chain_eval with seed_offset, both checkpoint paths, and the final-confirm
load path — without the expensive full-size evals.
"""
import torch
import gymnasium as gym
import homebot  # noqa: F401

from agent import Agent

FRAME_SKIP = 2

env = gym.make(
    "HomeBot2D-V1", render_mode="rgb_array", action_mode="continuous",
    obs_resolution=(96, 96), n_trash=2, max_steps=20000,
    map_name="default", random_start=True,
)
from env_wrappers import FrameSkipWrapper
env = FrameSkipWrapper(env, skip=FRAME_SKIP)

agent = Agent(env=env, max_buffer_size=5000, use_motion=True, motion_window=8)

# 1. Two training episodes, two legs each; batch_size huge so no grad steps
#    (keeps CPU time down; train_step itself is unchanged from prior runs).
#    confirm_bar=0 + confirm_interval=1 force the [Confirm] path, and the
#    FINAL_POLICY/FINAL_CONFIRM n=40 evals run at the end.
agent.train(episodes=2, batch_size=999999, run_tag="smoke",
            eval_interval=10000, chain_eval_interval=1,
            goals_per_episode=2,
            confirm_bar=0.0, confirm_episodes=1,
            confirm_interval=1, confirm_start=1)
print(f"SMOKE train loop OK | buffer fill={agent.memory.mem_ctr} "
      f"env_steps={agent.total_env_steps}")
assert agent.memory.mem_ctr > 0, "HER flush stored nothing"

# 2. Eval paths.
rate = agent.greedy_eval(n_episodes=1, stochastic=True)
print(f"SMOKE greedy_eval OK | rate={rate}")
score, full, spin = agent.chain_eval(n_episodes=1, seed_offset=1000)
print(f"SMOKE chain_eval OK | score={score} full={full} spin={spin:.3f}")

# 3. Checkpoint paths, including the final-confirm load.
agent.save_best(1, 3.3)
agent.save_best(1, 3.3, chain_full=0.4, path="checkpoints/best_confirmed.pt")
ckpt = torch.load("checkpoints/best_confirmed.pt", map_location=agent.device,
                  weights_only=True)
agent.actor.load_state_dict(ckpt["actor"])
assert ckpt["chain_full"] == 0.4 and ckpt["episode"] == 1
print("SMOKE checkpoints OK")
print("SMOKE ALL OK")
