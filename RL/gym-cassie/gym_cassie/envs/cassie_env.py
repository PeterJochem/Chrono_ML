import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import time
import pybullet_data

absolute_path_urdf = "/home/peter/Desktop/HoppingRobot_Fall/RL/gym-cassie/gym_cassie/envs/cassie/urdf/cassie_collide.urdf"

class CassieEnv(gym.Env):       
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        
        self.physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version

        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0, 0, -10)

        p.loadURDF("plane.urdf")
        self.humanoid = p.loadURDF(absolute_path_urdf, [0, 0, 0.8], useFixedBase = False)

        self.cubeStartPos = [0, 0, 1]
        self.cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])


    def step(self, action):
        
        # Step forward some finite number of seconds or milliseconds
        for i in range (10):
            p.stepSimulation()
            time.sleep(1.0/240.0)
            self.cubePos, self.cubeOrn = p.getBasePositionAndOrientation(self.humanoid)
            print(self.cubePos, self.cubeOrn)
        
        # return observation, reward, done, info
      
    def reset(self):  
        # Implement me!
        return self.myNumber

    
      
    def render(self, mode='human', close=False):
        pass




