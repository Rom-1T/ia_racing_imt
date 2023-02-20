import albumentations as A
class NoisyImage(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env=env)
    def reset(self):
        return self.env.reset()
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        x = np.random.randint(50,100)
        y = np.random.randint(0,5)
        t1 = A.RandomSunFlare(flare_roi=(0, 0, 1, 1), num_flare_circles_lower=1, num_flare_circles_upper=3,
                              src_color=(255, 255, 255), src_radius=x, always_apply=True)
        t2 = A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1,
                          drop_color=(255, 255, 255), blur_value=1, brightness_coefficient=1, rain_type=None,
                          always_apply=True)
        t3 = A.CoarseDropout(max_holes=10, max_height=10, max_width=15, min_holes=1, min_height=1, min_width=1,
                             fill_value=(255,255,255), mask_fill_value=None, always_apply=True)
        t4_ = A.MotionBlur(blur_limit=(7, 7), always_apply=True)
        t4 = A.Compose([t4_,t4_])
        transfos = A.SomeOf([t1, t2,t3, t4], n=y)
        obs = transfos(image=obs)['image']
        return obs, reward, done, info
