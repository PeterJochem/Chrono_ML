import gym
from gym import error, spaces, utils
from gym.utils import seeding

class CassieEnv(gym.Env):       
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.myNumber = 0

    def step(self, action):
        self.myNumber = self.myNumber + 1
        return self.myNumber
      
    def reset(self):  
        self.myNumber = 0
        return self.myNumber

    def render(self, mode='human', close=False):
        pass
