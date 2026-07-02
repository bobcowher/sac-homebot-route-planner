from agent import Agent
import gymnasium as gym
import homebot  # noqa: F401  (side-effect env registration)

FRAME_SKIP = 2

# Chain-style training: the base env (same one chain_eval deploys on), not the
# single-goal Goal env. Reward/termination live in the training loop now
# (distance <= GOAL_THRESHOLD, identical to the HER relabel rule). max_steps
# is a generous ceiling; per-leg budgets in agent.train() do the real limiting.
env = gym.make(
    "HomeBot2D-V1",
    render_mode="rgb_array",
    action_mode="continuous",
    obs_resolution=(96, 96),
    n_trash=2,
    max_steps=20000,
    map_name="default",
    random_start=True,
)

if FRAME_SKIP > 1:
    from env_wrappers import FrameSkipWrapper
    env = FrameSkipWrapper(env, skip=FRAME_SKIP)

agent = Agent(
    env=env,
    max_buffer_size=200000,
    goal_layers=2,
    head_layers=4,
    use_motion=True,
    motion_window=8,
)

# Chain training at Q-DQN's real budget. Run 418 (same config, 1800 eps)
# ended still climbing: sustained ep1560-1790 = 4.27/5, 62.9% full-chain
# (ties run 409's lifetime numbers at half the budget), last 10 windows
# 4.52/5, 82%, spin 0.031-0.048. Q-DQN run 411 needed ~3880 of 4500 eps to
# peak (4.83/5, 83.3% sustained, pinned at 5.0 after ~ep2500) — this run
# gives chain training the same 4500 episodes to consolidate. Also new:
# periodic held-out confirms (raw-best confirms saturate at 5.0 — run 418's
# ep700 confirm blocked all later ones) and FINAL_POLICY + FINAL_CONFIRM
# both measured at n=40.
agent.train(
    episodes=4500,
    batch_size=256,
    eval_interval=50,
    eval_episodes=20,
    chain_eval_interval=10,
    goals_per_episode=5,
    her_anneal_start=None,
    confirm_interval=250,
    confirm_start=500,
)
