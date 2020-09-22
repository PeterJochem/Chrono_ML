import numpy as np
import time
import random
import matplotlib.pyplot as plt
import warnings

import keras 
#from keras import models
#from keras import layers

warnings.filterwarnings('ignore') # Gets rid of future warnings
with warnings.catch_warnings():
    import tensorflow as tf
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class trainInstance:

    def __init__(self, inputVector, label):

        self.inputVector = inputVector # [gamma, beta]
        self.label = label # [x-stress, y-stress]

class dataSet:
    
    def __init__(self, fileName):
        
        myFile = open(fileName, "r")
        self.allData = []
        
        self.minDepth = 1000000 
        self.maxDepth = -1000000

        for line in myFile: 
            
            # [gamma, beta, depth, x-stress, y-stress]
            x = line.split(",")
            gamma = float(x[0]) / 90.0 # The angles are in degrees
            beta = float(x[1]) / 90.0

            depth = float(x[2]) # This is in cm - Chen Li uses cm^3
            x_stress = float(x[3]) #/ 100**2
            z_stress = float(x[4]) #/ 100**2

            newTrainInstance = trainInstance([gamma, beta], [(x_stress/depth), (z_stress/depth)])
            self.allData.append(newTrainInstance)

            #print(x_stress/depth)

            if (depth < self.minDepth):
                self.minDepth = depth
            if (depth > self.maxDepth):
                self.maxDepth = depth
    

        self.logStats()
        
        
    def logStats(self):
        print("The dataset was created succesfully")
        print("The dataset has " + str(len(self.allData)) + " training instances") 
        
    """ Tensorflow reformatting
    Split the lists of data into their input and outputs """
    def reformat(self, trainSet, testSet):
            
        train_inputVectors = []
        train_labels = []
        for i in range(len(trainSet)):
            train_inputVectors.append(trainSet[i].inputVector)
            train_labels.append(trainSet[i].label)
        

        test_inputVectors = []
        test_labels = []
        for i in range(len(testSet)):
            test_inputVectors.append(testSet[i].inputVector)
            test_labels.append(testSet[i].label)

         
        return train_inputVectors, train_labels, test_inputVectors, test_labels


"""Wrapper class for a tensorflow neural network"""
class NeuralNetwork:

    def __init__(self, dataset):
            
        self.useKeras = False
        self.useTf = False

        self.dataset = dataset
        self.inputShape = len(dataset.allData[0].inputVector)
        self.outputShape = len(dataset.allData[0].label)
        
        # Split data set into train and test
        np.random.shuffle(self.dataset.allData)

        self.trainSet = self.dataset.allData[ :int(0.80 * len(self.dataset.allData))]

        self.testSet = self.dataset.allData[int(0.80 * len(self.dataset.allData)): ]

        self.train_inputVectors, self.train_labels, self.test_inputVectors, self.test_labels = myDataSet.reformat(self.trainSet, self.testSet)
        
        # Setup the saving of the network
        #saver = tf.train.Saver()
        
        # Hyper parameters 
        self.epochs = 10000
        # numLayers ...
        # FIX ME
    
    
    def defineGraph_tf(self):
        
        self.useTf = True
        self.x = tf.placeholder(tf.float32, shape=(None, self.inputShape), name = 'x')

        # Outputs are [stress X, stress Z]
        self.y = tf.placeholder(tf.float32, shape=(None, self.outputShape), name = 'y')

        self.W1 = tf.Variable(tf.random_normal([self.inputShape, 10], stddev = 0.03), name = 'W1')
        self.b1 = tf.Variable(tf.random_normal([10]), name = 'b1')

        self.W2 = tf.Variable(tf.random_normal([10, 6], stddev = 0.03), name = 'W2')
        self.b2 = tf.Variable(tf.random_normal([6]), name = 'b2')

        self.W3 = tf.Variable(tf.random_normal([6, 2], stddev = 0.03), name = 'W3')
        self.b3 = tf.Variable(tf.random_normal([2]), name = 'b3')

        #self.W4 = tf.Variable(tf.random_normal([12, 5], stddev = 0.03), name = 'W4')
        #self.b4 = tf.Variable(tf.random_normal([5]), name = 'b4')

        #self.W5 = tf.Variable(tf.random_normal([5, self.outputShape], stddev = 0.03), name = 'W5')
        #self.b5 = tf.Variable(tf.random_normal([self.outputShape]), name = 'b5')

        self.hidden_out1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.W1), self.b1))
        self.hidden_out2 = tf.nn.relu(tf.add(tf.matmul(self.hidden_out1, self.W2), self.b2))
        #self.hidden_out3 = tf.nn.relu(tf.add(tf.matmul(self.hidden_out2, self.W3), self.b3))
        #self.hidden_out4 = tf.nn.relu(tf.add(tf.matmul(self.hidden_out3, self.W4), self.b4))

        self.y_pred = tf.add(tf.matmul(self.hidden_out2, self.W3), self.b3)

        self.loss = tf.nn.l2_loss(tf.subtract(self.y, self.y_pred)) #/ (len(myDataSet.allData))
        #self.loss = tf.reduce_mean(tf.square(self.y_pred - self.y))
        
        # Derivatives of the graph
        self.d_loss_dx = tf.gradients(self.loss, self.x)[0]

        # 0.00000001
        self.optimizer = tf.train.GradientDescentOptimizer(0.0000001)
        self.train_op = self.optimizer.minimize(self.loss)


    def defineGraph_keras(self):
            
        self.useKeras = True 
        self.network = keras.Sequential([
            keras.layers.Dense(20, input_dim = 2),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(28),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(14),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(2)
        ])

        # Set more of the model's parameters
        self.optimizer = keras.optimizers.Adam(learning_rate = 0.01)

        self.network.compile(loss='mse',
                optimizer = self.optimizer,
                metrics=['mse']) 
        


    """ hi """
    def train_tf(self):
        
        with tf.Session() as sess:
                
            # Initialize the variables
            sess.run(tf.global_variables_initializer())
        
            #save_path = saver.save(sess, "../myNetworks/myNet.ckpt")
            
            feed_dict_train = {self.x: self.train_inputVectors, self.y: self.train_labels}
            feed_dict_test = {self.x: self.test_inputVectors, self.y: self.test_labels}

            loss_train_array = np.zeros(self.epochs)
            loss_test_array = np.zeros(self.epochs)  

            for i in range(self.epochs):
                sess.run(self.train_op, feed_dict_train)
                loss_train_array[i] = self.loss.eval(feed_dict_train) 
                loss_test_array[i] = self.loss.eval(feed_dict_test)

            # Feed in a real pair of data we trained on
            # grad_numerical = sess.run(self.d_loss_dx, {self.x: [self.test_inputVectors[0] ] , self.y: [self.test_labels[0] ] } )  
            prediction = sess.run(self.y_pred, {self.x: [self.test_inputVectors[0]], self.y: [self.test_labels[0]] } )
            print("")
            print("The prediction is ")
            print(prediction)

            plt.plot(np.linspace(0, self.epochs, self.epochs), loss_test_array, color = "blue")
            plt.title('Loss Function vs Epoch - Hopping Foot')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.show()

        #"""Describe"""
        #def viewLearnedFunction(self):
        
            angle_resolution = 20
            # Remember we are using DEGREES
            angle_increment = 180.0 / angle_resolution
            
            # Vary this
            # depth_increment =  
            # For experimentation - set this to the average value
            #depth = 0.5 * (self.dataset.minDepth + self.dataset.maxDepth)
        
            num_depth_intervals = 20
            depth_increment = self.dataset.maxDepth - self.dataset.minDepth
            
            # inputData = []
    
            F_X = np.zeros((angle_resolution + 1, angle_resolution + 1)) 
            F_Z = np.zeros((angle_resolution + 1, angle_resolution + 1))

            #count = 0
            for i in range(angle_resolution):
                for j in range(angle_resolution):
                    
                    gamma = -1 * (180.0 / 2.0) + (j * angle_increment)
                    beta = (180.0 / 2.0) - (i * angle_increment)

                    nextEntry = [gamma, beta]
            
                    # The y-label is irrelevant to predicting the y-label, here, after training
                    prediction = self.network.predict([nextEntry]) 
                    #sess.run(self.y_pred, {self.x: [nextEntry] , self.y: [[ 1.0, 1.0 ]] } )
                        
                        
                    #if (newX < 0):
                    #    newX = 0
                    #if (newZ < 0):
                    #    newZ = 0

                    F_X[i][j] = prediction[0][0]
                    #F_Z[i][j] = newZ
                    

            w = 10
            h = 10
            fig = plt.figure(figsize = (8, 8))
            columns = 2
            rows = 1
            ax1 = fig.add_subplot(rows, columns, 1)
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])
            plt.imshow(F_Z)
            ax1.set_ylabel('β')
            ax1.set_xlabel('γ')
            ax1.title.set_text('α_Z')
            plt.colorbar()

            ax2 = fig.add_subplot(rows, columns, 3)
            ax2.set_yticklabels([])
            ax2.set_xticklabels([])
            plt.imshow(F_X)
            ax2.set_ylabel('β')
            ax2.set_xlabel('γ')
            ax2.title.set_text('α_X')
    
            plt.colorbar()
            plt.show()

        
    def train_keras(self):
        
        # self.train_inputVectors, self.y: self.train_labels
        print("\n\n\n\n\nThe length of __ is " + str(len(self.train_inputVectors)) )
        #batch_size = 1000
        self.network.fit([self.train_inputVectors], [self.train_labels], batch_size = len(self.train_inputVectors), epochs = 4000)
            
        angle_resolution = 40
        # Remember we are using DEGREES
        angle_increment = 180.0 / angle_resolution

        # Vary this
        # depth_increment =  
        # For experimentation - set this to the average value
        #depth = 0.5 * (self.dataset.minDepth + self.dataset.maxDepth)

        #num_depth_intervals = 20
        #depth_increment = self.dataset.maxDepth - self.dataset.minDepth

        F_X = np.zeros((angle_resolution, angle_resolution))
        F_Z = np.zeros((angle_resolution, angle_resolution))
            
        for i in range(angle_resolution):
            for j in range(angle_resolution):
                
                gamma = (-1*(180.0 / 2.0) + (j * angle_increment)) / 90.0 
                beta = ((180.0 / 2.0) - (i * angle_increment)) / 90.0

                nextEntry = [gamma, beta]

                prediction = self.network.predict([[nextEntry]])

                #if (newX < 0):
                #newX = 0
                #if (newZ < 0):
                #newZ = 0

                F_X[i][j] = prediction[0][0] # 0.15
                F_Z[i][j] = prediction[0][1]


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
        plt.colorbar()
        
        
        ax2 = fig.add_subplot(rows, columns, 2)
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        plt.imshow(F_X)
        ax2.set_ylabel('β')
        ax2.set_xlabel('γ')
        ax2.title.set_text('α_X')
        print("The (0, 0) prediction is " + str(self.network.predict([[[0, 0]]])))
        plt.colorbar()
        plt.show()
            


# Create a datset object with all our data
myDataSet = dataSet("../dataSets/allData/compiledSet.csv")

myNetwork = NeuralNetwork(myDataSet)

#myNetwork.defineGraph_tf()
#myNetwork.train_tf()
myNetwork.defineGraph_keras()
myNetwork.train_keras()



"""
    # Run the predictions with diffrent gamma, and betas
    # Warmup: Predict the original data/test data

    resolution = 20
    # Remember we are using DEGREES
    increment = 180.0 / resolution
    
    inputData = []
    
    F_X = np.zeros((resolution, resolution)) 
    F_Z = np.zeros((resolution, resolution))

    count = 0
    for i in range(resolution):
        for j in range(resolution):
            
            gamma = -1 * (180.0 / 2.0) + (j * increment) 
            beta = (180.0 / 2.0) - (i * increment)
            
            # depth = 1.0
            nextEntry = [gamma, beta]
            
            inputData.extend( [nextEntry] )
            count = count + 1
                
            # The y-label is irrelevant to predicting the y-label, here, after training
            prediction = sess.run(y_pred, {x: [nextEntry] , y: [[ 1.0, 1.0 ]] } )
            
            # F_X[i][j] = ( (prediction[0][0] + 0.1) / 0.2) * 255
            # F_Z[i][j] = ( (prediction[0][1] + 0.3) / 0.6) * 255

            F_X[i][j] = prediction[0][0] 
            F_Z[i][j] = prediction[0][1]


    w = 10
    h = 10
    fig = plt.figure(figsize = (8, 8))
    columns = 3
    rows = 1
    ax1 = fig.add_subplot(rows, columns, 1)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    plt.imshow(F_Z)
    ax1.set_ylabel('β')
    ax1.set_xlabel('γ')
    ax1.title.set_text('α_Z')
    plt.colorbar()

    ax2 = fig.add_subplot(rows, columns, 3)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    plt.imshow(F_X)
    ax2.set_ylabel('β')
    ax2.set_xlabel('γ')
    ax2.title.set_text('α_X')
    
    plt.colorbar()
    plt.show()
"""
