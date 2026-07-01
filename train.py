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

# 3500 eps: matches run 409's exact budget for a fair comparison. Run 415
# (same config, 1800 eps) sustained-window average was 3.68/5, 30.4% full-
# chain -- worse than 409's real sustained number (4.29/5, 60%, corrected
# from an earlier cherry-picked "final raw" reading). Does the faster ramp
# to peak capability (ep860-1370 vs 409's slower climb) convert into a
# BETTER sustained result once given the same total training time, or is
# 409's more conservative default target_entropy actually more stable long
# run despite starting slower?
agent.train(
    episodes=3500,
    batch_size=256,
    eval_interval=50,
    eval_episodes=20,
    chain_eval_interval=10,
    her_anneal_start=None,
)
