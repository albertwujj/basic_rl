import gym
import numpy as np

# FrozenLake-v0
class EnvWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.desc = env.desc

    def one_hot(self, integer):
        ret = np.zeros((self.observation_space.n))
        ret[integer] = 1
        return ret

    def step(self, action):
        obs, rew, done, _ = self.env.step(action)
        return self.one_hot(obs), rew, done, None

    def reset(self):
        obs = self.env.reset()
        return self.one_hot(obs)

    def viz_actions(self, model):
        for i in range(len(self.desc)):
            for j in range(len(self.desc[0])):
                a = (model.choose_act([self.one_hot(i * 4 + j)])[1][0] * 100).astype(np.int32)
                print(a, end='')
            print()
        print()