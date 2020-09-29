""" Run this just to make sure gym has our environment installed """
import gym
import gym_laikago
env = gym.make('laikago-v0')

print("\n Letting robot fall down")
while i in range(10):
    
    # Let the robot fall down
    # Fix me
    env.step(1)
    
print("Resetting robot to start state")
env.reset()

print("checking that start state has robot idle before first step call")
while True:
    env.step(1)

