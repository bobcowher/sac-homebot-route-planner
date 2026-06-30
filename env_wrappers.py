import gymnasium as gym


class FrameSkipWrapper(gym.Wrapper):
    """Repeats the chosen action for `skip` steps, accumulating reward."""
    def __init__(self, env, skip=2):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
