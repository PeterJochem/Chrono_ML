# This Week
Last Friday, Matt and I talked about what the next steps are for the project. We are interested in seeing if a reinforcement learning algorithm could learn a control policy for walking on deformable surfaces. We decided to implement an OpenAI gym environment for the Cassie robot as well as create a URDF and gym for a hopper robot similiar to Dan's robot. <br />

# Hopper Robot Gym and URDF
Dan and I also talked about how we could avoid having the ML algorithm do direct torque control of the joints. Dan's open-loop optimal controller is a wrench controller. This means that there is some lower level controller which must convert the optimal wrench down to a set of joint torques to apply. In that same way, it makes sense to have the reinforcement agent learn what wrench to apply rather than what torques to apply. We already have an analytical function that maps a desired end effector wrench of an open chain to the set of joint torques to apply. We could use the functions of the Modern Robotics library to take desired wrenches and map them to joint torques. This would significantly reduce the time needed to train the agent because it would not need to learn a representation of the underlying dynamics captured by the Newton-Euler equations. <br />


# Results
I set up the gym environment for the Cassie robot, created an open chain URDF for the hopper robot, and started an OpenAI gym for the hopper. More details are availble [here](https://github.com/PeterJochem/Chrono_ML/tree/master/RL).
I also did more reading on reinforcement learning and implemented Double Deep Q Learning, [more details here](https://github.com/PeterJochem/Deep_RL/tree/master/DDQN/cart_pole). Specificaly, I read about the REINFORCE algorithm and got started on actor critic methods. I think the first algorithm to use would be DDPG (Deep Deterministic Policy Gradients). I have done RL under continous state spaces but this is my first time doing RL with a continous action space. DDPG was created to do just that, learn control policies for continous action spaces. I started setting up a DDPG algorithm with Keras. I struggled a lot with getting Keras to compute the gradients I needed.  

# Next Steps
Process the Chrono dataset with the added velocity information. <br /> 

Setup DDPG and validate it with the OpenAI Pendulum gym environment. This should be able to train in 1-10 minutes. <br />

Setup DDPG for either the Hopper Robot or Cassie for hard ground. This should not be very hard if the above works but could require long training sessions and or labor intensive hyper-parameter tuning. <br />  

Alter the Cassie URDF so that both the Hopper and Cassie have the same flat plate type of feet. <br />

Setup DDPG for the Hopper and or Cassie with the deformable surfaces module.                  
