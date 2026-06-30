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
)

agent.train(
    episodes=2500,
    batch_size=256,
    eval_interval=50,
    eval_episodes=20,
    chain_eval_interval=10,
    her_anneal_start=None,
)
