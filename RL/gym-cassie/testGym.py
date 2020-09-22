""" Run this just to make sure gym has our environment installed """
import gym
import gym_cassie
env = gym.make('cassie-v0')

while True:
    env.step(1)

