# Description
Agility Robotics released its meshes and URDF of the Cassie robot. Erwin Coumans, the creator of Bullet, also has a [repo](https://github.com/erwincoumans/pybullet_robots) describing how to use PyBullet with 10 or so common robots. He includes an example for Cassie.

# cassie_env.py
This implements the class that represents the gym. It implements the step, reset, and render methods that are the interface to the OpenAI gym. It also handles the tracking of the robot's state. 

# cassie.py
This is an example of how to interact and use the PyBullet simulation.

# cassie 
This directory has the files for Cassie's URDF and meshes 

# plane.mtl, plane.urdf, plane.obj
These are required to implement the plane in PyBullet  
