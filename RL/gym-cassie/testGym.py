""" Run this just to make sure gym has our environment installed """
import gym
import gym_cassie
env = gym.make('cassie-v0')

i = 1
while i in range(10):
    print("i is " + str(i) )
    env.step(1)
    i = i + 1

env.reset()

while True:
    env.step(1)

