import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import time
import pybullet_data
import keras

absolute_path_urdf = "/home/peter/Desktop/HoppingRobot_Fall/RL/gym-cassie/gym_cassie/envs/cassie/urdf/cassie_collide.urdf"

class CassieEnv(gym.Env):       
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        
        # Two environments? One for graphics, one w/o graphics?
        self.physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
        
        self.jointIds=[]
        self.paramIds=[]

        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0, 0, -10)

        p.loadURDF("plane.urdf")
        self.humanoid = p.loadURDF(absolute_path_urdf, [0, 0, 0.8], useFixedBase = False)

        self.cubeStartPos = [0, 0, 1]
        self.cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        
        print("The number of joints on the Cassie robot is " + str(p.getNumJoints(self.humanoid)) )
        
        # Setup the debugParam sliders
        # Why -10, 10, -10
        self.gravId = p.addUserDebugParameter("gravity", -10, 10, -10) 
        self.homePositionAngles = [0, 0, 1.0204, -1.97, -0.084, 2.06, -1.9, 0, 0, 1.0204, -1.97, -0.084, 2.06, -1.9, 0]
        
        activeJoint = 0
        for j in range (p.getNumJoints(self.humanoid)):
            
            # Why set the damping factors to 0?
            p.changeDynamics(self.humanoid, j, linearDamping = 0, angularDamping = 0)
            info = p.getJointInfo(self.humanoid, j)
            jointName = info[1]
            jointType = info[2]

            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                
                self.jointIds.append(j)
                self.paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, self.homePositionAngles[activeJoint]))
                p.resetJointState(self.humanoid, j, self.homePositionAngles[activeJoint])
                activeJoint = activeJoint + 1

        # What exactly does this do?
        p.setRealTimeSimulation(1)
        
        self.stateId = p.saveState() # Stores state in memory rather than on disk
                    

    def step(self, action):
        
        # Forward prop neural network to get GRF, use that to change the gravity
        p.getCameraImage(320, 200)
        p.setGravity(0, 0, p.readUserDebugParameter(self.gravId))

         
        # Step forward some finite number of seconds or milliseconds
        for i in range (10):
            p.stepSimulation()
            time.sleep(1.0/240.0)
            self.cubePos, self.cubeOrn = p.getBasePositionAndOrientation(self.humanoid)
            #print(self.cubePos, self.cubeOrn)
            time.sleep(0.001)
            
            for i in range(len(self.paramIds)):
                nextJointId = self.paramIds[i]
                targetPos = p.readUserDebugParameter(nextJointId)
                p.setJointMotorControl2(self.humanoid, self.jointIds[i], p.POSITION_CONTROL, targetPos, force = 140.0)

         
        # return observation, reward, done, info
      
    def reset(self):  
            
        p.restoreState(self.stateId, "cassie_start.bullet")
              
      
    def render(self, mode='human', close = False):
        pass
    
    """Read the state of the simulation to compute 
    and return the reward scalar for the agent"""
    def computeReward(self):
        pass 
            




