# This Week
I got DDPG up and running with a few environments. I have been using the penduluum environment as a way to test my implementation. This allows me to see in a few minutes whether my implementation of DDPG is correct or not. Other environments can require much more training time. <br />

I also setup MuJoCo's Hopper environment for a more robust DDPG test. I have the agent learning in the penduluum environment but my implementation seems to fail in the MuJoCo Hopper environment. Hmmmm, more debugging. <br />

I also setup the machine learning pipeline for the second dataset. This dataset includes the velocity information. The RFT model does not use the velocity of the foot to estimate the GRF. Because of this, the original simulation uses a much lower speed since the RFT does not require the speed information. I need to work with Juntao to generate a much larger dataset over which we vary the speed that the plate intrudes through the granular material. The current dataset's speeds are not at a speed which is realistic for Dan's robot.

# Next Steps 
The next steps are to continue debugging the MuJoCo hopper in order to validate my implementation of DDPG. I also need to work with Juntao to generate a new dataset where we vary the speed at which the robot's foot intrudes into the granular material. This new data should also increase the speed at which the foot traverses through the material. It's speeds should be near what Dan's robot moves at.      
