from agent import Agent
import gymnasium as gym
import homebot  # noqa: F401  (side-effect env registration)

FRAME_SKIP = 2

env = gym.make(
    "HomeBot2D-Goal-V1",
    render_mode="rgb_array",
    action_mode="continuous",
    obs_resolution=(96, 96),
    n_trash=1,           # single trash == conditioned goal == completion event
    max_steps=1000,
    map_name="default",
    goals=["collect_trash"],
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
    random_goal_tiles=True,
    target_entropy=-1.0,      # isolate this alone vs 409's -2.0; no alpha floor this time
)

# 1000 eps, not 3500: testing convergence SPEED against Q-DQN's 500-600
# episode budget, and against 409's 1000-episode checkpoint (already perfect
# 5.0/5 chain_score, 100% chain_full by then).
agent.train(
    episodes=1000,
    batch_size=256,
    eval_interval=50,
    eval_episodes=20,
    chain_eval_interval=10,
    her_anneal_start=None,
)
