""" Run this just to make sure gym has our environment installed """
import gym
import numpy as np
import gym_hopping_robot
env = gym.make('hopping_robot-v0')

for outerLoop in range(100):
    for i in range(50):
        print("i is " + str(i))

        #action = neuralNetwork.forwardProp(observation) 

        # This is the home position
        #action = [0, 0, 0]
        action = (np.random.rand(1, 1)[0]) * 3.14
        observation, reward, done, info = env.step(action)
        

        if (done):
            env.reset()


