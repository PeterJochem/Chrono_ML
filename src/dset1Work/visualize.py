import numpy as np
import random
import matplotlib.pyplot as plt


"""Wrapper class for a training instance. Makes it easier to wrangle data later"""
class trainInstance:

    def __init__(self, inputVector, label):

        self.inputVector = inputVector # [gamma, beta]
        self.label = label # [x-stress/depth, y-stress/depth]


"""Set of data and methods for creating a dataset which can be processed by Tensorflow or Keras"""
class dataSet:

    def __init__(self, fileName):

        myFile = open(fileName, "r")
        self.allTrainingInstances = []
        self.allData = []

        self.minDepth = 1000000
        self.maxDepth = -1000000
        
        for line in myFile:

            x = line.split(",") # # [gamma, beta, depth, x-stress, y-stress] - the csv file format

            gamma = float(x[0]) # The angles are in degrees
            beta = float(x[1]) 

            depth = float(x[2]) # This is in cm - Chen Li uses cm^3
            x_stress = float(x[3]) #/ 100**2
            z_stress = float(x[4]) #/ 100**2
            
            self.allData.append([gamma, beta, depth, x_stress, z_stress])
        
            # Record the min and max depth for debugging and log it to console
            if (depth < self.minDepth):
                self.minDepth = depth
            if (depth > self.maxDepth):
                self.maxDepth = depth
        
        self.sortByDepth()

        for vector in self.allData:
            gamma, beta, depth, x_stress, z_stress = vector
            newTrainInstance = trainInstance([gamma, beta], [(x_stress/depth), (z_stress/depth)])
            self.allTrainingInstances.append(newTrainInstance)
            
        self.logStats()
        self.displayData(5)


    def logStats(self):
        print("The dataset was created succesfully")
        numInstances = len(self.allData)
        print("The dataset has " + f"{numInstances:,}" + " training instances")

    """Sort the dataset by depth"""
    def sortByDepth(self):
        
        # depth is at index 2 of the training instance's input vector
        self.allData = sorted(self.allData, key=lambda x:x[2])


    """"""
    def displayData(self, numDepthGroups = 1):
        angle_resolution = 8
        angle_increment = 180.0 / angle_resolution # Remember we are using DEGREES
        

        for outerLoop in range(numDepthGroups):
            # Create fixed size 2d arrays
            #F_X = [[0]*angle_resolution for i in range(angle_resolution)]
            #F_Z = [[0]*angle_resolution for i in range(angle_resolution)]
            F_X = [[0 for i in range(angle_resolution)] for j in range(angle_resolution)]
            F_Z = [[0 for i in range(angle_resolution)] for j in range(angle_resolution)]

            for i in range(angle_resolution):
                for j in range(angle_resolution):    
                    F_X[i][j] = []
                    F_Z[i][j] = []

            lowerIndex = int(len(self.allTrainingInstances)/float(numDepthGroups) * outerLoop)
            upperIndex = int(len(self.allTrainingInstances)/float(numDepthGroups) * (outerLoop + 1) - 2)
             
            nextBatch = self.allTrainingInstances[lowerIndex:upperIndex]
            
            print("The length of the next batch is " + str(len(nextBatch)))

            for i in range(angle_resolution):
                gamma_min = -1 * (90.0) + (i * angle_increment)
                gamma_max = -1 * (90.0) + ((i + 1) * angle_increment)

                for j in range(angle_resolution):
                    
                    beta_min = (90.0) - ((j + 1) * angle_increment)
                    beta_max = (90.0) - (j * angle_increment) 
                    
                    for k in range(len(nextBatch)):
                
                        nextGamma = nextBatch[k].inputVector[0]
                        nextBeta = nextBatch[k].inputVector[1] 
                        
                        #gamma_check = (nextGamma >= gamma_min) and (nextGamma <= gamma_max)
                        #beta_check = (nextBeta >= beta_min) and (nextBeta <= beta_max)
                        if ((nextGamma >= gamma_min) and (nextGamma <= gamma_max) and (nextBeta >= beta_min) and (nextBeta <= beta_max)):
                
                            # Append the stresses/depth calculations
                            F_X[j][i].append(nextBatch[k].label[0]) 
                            F_Z[j][i].append(nextBatch[k].label[1])

            self.createPlots(F_Z, F_X, True, True)
    

    def createPlots(self, F_Z, F_X, useDefaultMap = False, customLimits = False):

        if (customLimits == True):
            plt.rcParams['image.cmap'] = 'jet' # Similiar color scale as Juntao's data visualizations

        for i in range(len(F_X)):
            for j in range(len(F_X[i])):
                F_X_sum = 0
                F_Z_sum = 0
                for k in range(len(F_X[i][j])):
                    F_X_sum = F_X_sum + F_X[i][j][k]
                    F_Z_sum = F_Z_sum + F_Z[i][j][k]
    
                if (len(F_X[i][j]) <= 0):
                    print("No data in group")
                    F_X[i][j] = 1.0 # FIX ME!!
                else:
                    F_X[i][j] = float(F_X_sum)/len(F_X[i][j])
                
                if (len(F_Z[i][j]) <= 0):
                    F_Z[i][j] = 1.0 # FIX ME!!!!
                else:
                    F_Z[i][j] = float(F_Z_sum)/len(F_Z[i][j])
    


        fig = plt.figure(figsize = (4, 4))
        columns = 2
        rows = 1
        ax1 = fig.add_subplot(rows, columns, 1)
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        plt.imshow(F_Z)
        ax1.set_ylabel('β')
        ax1.set_xlabel('γ')
        ax1.title.set_text('α_Z')

        if (customLimits == True):
            plt.clim(-0.25, 0.25); # This conrols the scale of the color map - I set it to better match Junatos visualization

        plt.colorbar()

        ax2 = fig.add_subplot(rows, columns, 2)
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        plt.imshow(F_X)
        ax2.set_ylabel('β')
        ax2.set_xlabel('γ')
        ax2.title.set_text('α_X')

        if (customLimits == True):
            plt.clim(-0.1, 0.1); # This conrols the scale of the color map - I set it to better match Junatos visualization

        plt.colorbar()
        plt.show()

    


stressDataFilePath = "../../dataSets/dset1/allData/compiledSet.csv"
myDataSet = dataSet(stressDataFilePath)
