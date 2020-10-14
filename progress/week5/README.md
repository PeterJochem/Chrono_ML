# This Week

## DDPG on Custom Robot
I am working on getting my DDPG implementation to work with my Hopper Robot in PyBullet. I looked into MuJoCo to see if it could simulate granular materials and it does. So, I tried to see if I could make an OpenAI gym environment in MuJoCo where we simulate the granular materials. I could not get MuJoCo to simulate anywhere near enough particles in order to run our experiment. <br />

I worked on creating the reward function for the custom hopper robot in PyBullet. It is much trickier than I would have expected. The robot always seems to find a way to exploit the reward function and get a high score but not really accomplish the desired behavior of hopping. Sometimes it gets stuck in the local optimum of launching itself forward to get a high score as fast as possible. If I reward staying alive more, the robot learns to stand still and collect an indefinite reward. I need to find a good reward function. 

## Dataset Automation 
I forked Juntao's code and made changes to the simulation code in order to create an automatic way to generate the dataset. Before, we had to modify the code itself, compile it, run it, and repeat manually for each pair of variable parameters in the dataset. This made it hard to generate datasets and required a TON of work. We also had a constant speed which was much slower than what Dan's robot will be moving at. I modified the code where needed and created bash scripts to do all this work automatically. Now, I can specify over what interval to vary the parameters in our input space and also how precisely to discretize the interval and the bash scripts generate the data. 

## Next Steps
Use the neural network from dataset3 to calculate the GRF forces and apply them in the PyBullet environment. 
         
