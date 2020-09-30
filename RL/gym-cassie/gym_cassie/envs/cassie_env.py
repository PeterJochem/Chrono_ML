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
        
        self.isOver = False

        # Two environments? One for graphics, one w/o graphics?
        #self.physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
        
        self.physicsClient = p.connect(p.GUI, options="--width=1920 --height=1080")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Removes the slider panel, looks very clean

        self.jointIds=[]
        self.paramIds=[]

        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0, 0, -10)

        p.loadURDF("plane.urdf")
        self.humanoid = p.loadURDF(absolute_path_urdf, [0, 0, 0.8], useFixedBase = False)

        #self.cubeStartPos = [0, 0, 1]
        #self.cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        
        print("The number of joints on the Cassie robot is " + str(p.getNumJoints(self.humanoid)))
        
        self.defineHomePosition()
        self.goToHomePosition()

        p.setRealTimeSimulation(0) # 0 means you must manually call p.stepSimulation()
        
        self.stateId = p.saveState() # Stores state in memory rather than on disk

    """Reset the robot to the home position"""
    def defineHomePosition(self):
        
        self.gravId = p.addUserDebugParameter("gravity", -10, 10, -10)

        # Why -10, 10, -10
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


    def goToHomePosition(self):
            
        activeJoint = 0
        for j in range (p.getNumJoints(self.humanoid)):

            # Why set the damping factors to 0?
            p.changeDynamics(self.humanoid, j, linearDamping = 0, angularDamping = 0)
            info = p.getJointInfo(self.humanoid, j)
            jointName = info[1]
            jointType = info[2]

            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                p.resetJointState(self.humanoid, j, self.homePositionAngles[activeJoint])
                activeJoint = activeJoint + 1
            
    """Action is a vector of joint angles for the PID controller"""
    def step(self, action):
        
        # Forward prop neural network to get GRF, use that to change the gravity
        # FIX ME 
        p.getCameraImage(320, 200)
        p.setGravity(0, 0, p.readUserDebugParameter(self.gravId))
        
        # Step forward some finite number of seconds or milliseconds
        self.controller(action)
        for i in range (10):
                p.stepSimulation()
                
                #time.sleep(1.0/240.0)
                # self.cubePos, self.cubeOrn = p.getBasePositionAndOrientation(self.humanoid)
                #time.sleep(0.001)
       
         
        # observation = list of joint angles

        return [], self.computeReward(), self.checkForEnd(), None
        # what is info?
        #return observation, reward, done, info

    """Check for controller signals. Allows user to use the GUI sliders
    Do I need this to programatically control robot????"""
    # controlSignal is the list of 12 joint angles 
    def controller(self, controlSignal):

        for i in range(len(self.paramIds)): 
            #nextJointId = self.paramIds[i]
            #targetPos = p.readUserDebugParameter(nextJointId) # This reads from the sliders
            targetPos = controlSignal[i]   
            p.setJointMotorControl2(self.humanoid, self.jointIds[i], p.POSITION_CONTROL, targetPos, force = 140.0)

    """Return the robot to its initial state""" 
    def reset(self):  
            
        robot_position, robot_orientation = p.getBasePositionAndOrientation(self.humanoid)
        self.jointIds=[]
        self.paramIds=[]
        p.removeAllUserParameters() # Must remove and replace
        self.defineHomePosition()
        p.restoreState(self.stateId)
        self.goToHomePosition()

    def render(self, mode='human', close = False):
        pass
    

    def checkForEnd(self): 
        
        self.robot_position, self.robot_orientation = p.getBasePositionAndOrientation(self.humanoid)
        roll, pitch, yaw = p.getEulerFromQuaternion(self.robot_orientation)
        
        # could also check the z coordinate of the robot?
        if (abs(roll) > (1.1) or abs(pitch) > (1.1)):
            self.isOver = True
            return True
        
        return False


    """Read the state of the simulation to compute 
    and return the reward scalar for the agent"""
    def computeReward(self):
        
        self.robot_position, self.robot_orientation = p.getBasePositionAndOrientation(self.humanoid)
        x, y, z = self.robot_position

        # Convert quaternion to Euler angles
        roll, pitch, yaw = p.getEulerFromQuaternion(self.robot_orientation)
    
        # Reward is inversely proportional to the rotation about the y axis (pitch), and x angle (roll)
        # Reward is proportional to the forward motion in the +x direction 
        return (-1.0/roll) + (-1.0/pitch) + (x * 10.0)




