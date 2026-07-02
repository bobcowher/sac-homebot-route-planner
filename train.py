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

# Chain training, first run: 5 sequential random-tile goals per episode with
# heading + motion history carrying across goal switches. Attacks the root
# cause identified in the entropy arc (runs 412-417): the old single-goal loop
# never sampled the mid-chain state distribution chain_eval deploys on, so
# post-saturation training drifted chain behavior while single-goal reach
# stayed pinned at 90-100%. Baseline to beat: run 409 sustained 4.29/5, 60%
# full-chain; Q-DQN benchmark 4.8/5, 80%.
agent.train(
    episodes=1800,
    batch_size=256,
    eval_interval=50,
    eval_episodes=20,
    chain_eval_interval=10,
    goals_per_episode=5,
    her_anneal_start=None,
)
