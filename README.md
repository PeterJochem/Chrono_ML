# Hopping Robot - Machine Learning
We want to learn a function describing how a robot's foot interacts with granular material. Ultimatelty, the goal is to help build a monoped robot that can hop on deformable surfaces. I am working with (link) Dan Lynch, (link) and Juntao He to build such a robot. My job is to take a dataset gathered from a physics simualtion (Kronos) (link) and use machine learning to summarize the relationships between the data. We would like to know what the force vector is when the robot's foot makes contact with the ground. Further iterations could involve significantly increasing the size of the input space. Currently, the robot's foot is simply a flat plate. We could also learn a mapping that takes the foot's shape or mass properties as a variable parameter. 

# Terradynamics Paper
Describe the original Terradynamics paper

# Project Chrono
Project Chrono is an open source phyics simulator. It has the ability to simulate granular materials which is why we choose it over some of the other options in this area. Juntao has generated a dataset of a little more than 2,000,000 data points of how the plate (foot) interacts with a bed of granular material.   

# Machine Learning Problem
We want to know what the force vector is for a robot's foot making contact with the ground. If the ground were solid, we might approach this differently, but we would like the robot to be able to hop on deformable surfaces. The dynamic properties of these materials are far more complicated. One approach is to simulate the materials and try to learn the function, rather than modeling it analytically. We opted for the machine learning approach. So,we used Chrono to simulate a flat plate interacting with a bed of granular material. The input is a a pair of angles, gamma () and beta (). These describe the foot's orientation and direction of motion as it travels through the granular material. The label is a 2 vector of the stress per unit depth (into the granular material) experienced by the plate in both the x and z directions.  
