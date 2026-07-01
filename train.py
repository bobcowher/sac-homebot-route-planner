from agent import Agent
import gymnasium as gym
import homebot  # noqa: F401  (side-effect env registration)
import torch

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
    target_entropy=-1.0,
)

# NOT a training run. Run 415 (this same config) peaked chain_score=4.70/5,
# chain_full=90% at ep1370 (best.pt), but the sustained 24-window average
# over the last 240 episodes was only 3.68/5, 30.4% full-chain -- a huge gap
# between peak and sustained performance. Before concluding the peak was
# real vs a lucky 10-episode draw, reload best.pt (still sitting in the
# shared checkpoints/ dir from run 415, since no run has started since) and
# run a much larger n=40 chain_eval to get a statistically confident read.
ckpt = torch.load("checkpoints/best.pt", map_location=agent.device, weights_only=True)
print(f"Loaded checkpoint: episode={ckpt['episode']}  "
      f"logged_chain_score={ckpt['chain_score']:.2f}")
agent.actor.load_state_dict(ckpt["actor"])

score, full, spin = agent.chain_eval(n_episodes=40)
print(f"CONFIRM_EVAL n=40: chain_score={score:.3f}/5  "
      f"chain_full={full:.3f}  spin={spin:.3f}")
