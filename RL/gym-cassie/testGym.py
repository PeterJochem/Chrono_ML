""" Run this just to make sure gym has our environment installed """
import gym
import gym_cassie
import numpy as np
env = gym.make('cassie-v0')

for outerLoop in range(100):
    i = 1
    while i in range(50):
        #print("i is " + str(i))
        #action = neuralNetwork.forwardProp(observation) 
        # This is the home position
        #action = [0, 0, 1.0204, -1.97, 0.084, 2.06, -1.9, 0, 0, 1.0204, -1.97, -0.084, 2.06, -1.9, 0]
        action = (np.random.rand(1, 14)[0] - 0.5)
        observation, reward, done, info = env.step(action)
        i = i + 1

        if (done):
            env.reset()

    env.reset()

