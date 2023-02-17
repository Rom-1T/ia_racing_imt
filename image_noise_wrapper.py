import albumentations as A
class NoisyImage(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env=env)
    def reset(self):
        return self.env.reset()
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        x = np.random.randint(50,100)
        obs = A.RandomSunFlare(flare_roi=(0,0,1,1), num_flare_circles_lower=1, num_flare_circles_upper= 3,src_color=(255, 255, 255), src_radius=x, always_apply=False)(image=obs)['image']
        return obs, reward, done, info
