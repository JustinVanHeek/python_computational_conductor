import numpy as np
import tensorflow as tf
import os
import math
import random
import _thread as thread
import time
import glob
import re
import matplotlib.pyplot as plt
from bluetooth import *
#from sklearn.model_selection import train_test_split
#from scipy.spatial import ConvexHull

################################## Settings ##################################

#   Recording Settings
windowSize      =   200     # The max number of lines of motion data that is kept (must not be shorter than the longest motion)
ignoreWindow    =   25      # The time after the start of a beat is detected that new detections should be ignored (used to reduce two detections of the same beat within a short time frame)

#   Neural Network Settings
size            =   50      # Length of each dimension of the image, larger = more detailed gestures
brushRadius     =   4       # Radius that values are applied to the image from source point, larger = more generalized data
nHiddenNeurons  =   300     # Number of hidden neurons in the neural network, larger = usually better at recognizing more details
nEpochs         =   30      # Number of training epochs for the neural network, larger = usually better accuracy until overfitting occurs
labels          =   ["2beatNormal","2beatStaccato","2beatLegato",
                     "3beatNormal","3beatStaccato","3beatLegato",
                     "4beatNormal","4beatStaccato","4beatLegato"]         # Labels of the gestures to recognize
nFolds          =   3       # Number of cross validation folds
detailedEpochs  =   True    # More detail on each Epoch will be assessed and logged (will run slightly slower)
shouldLog       =   True    # If there should be any print statements (This really should only be False if the program is being run in a different manner than usual such as being imported into anothe rprogram)

# Firm Settings (That should probably not be changed unless you know what you are doing)
scaleAccel      =   1/20    # Scale factor on acceleration data
scaleRot        =   1/8     # Scale factor on gyroscope data
scaleRotV       =   1/180   # Scale factor on rotation vector data
# Note: The scales should be set to scale the data to around a range of -1 to 1
#       So long as the data being recieved is in the same units as the output of Android motion sensors, these should not be adjusted

##############################################################################



########################### Global Variables #################################
runProgram = True           # Flag used to determine if a program's thread should continue running
beatName = "beat#"          # The user's inputted label for the files to be generated (should be one of the labels found in the labels list)
userName = "user"           # The user's inputted label of the person who generated the files (for the purpose if one user's gestures are poor quality they can easily be identified and removed)
accelX = [0]*windowSize     # The list of acceleration data along the x-axis being recorded from the bluetooth device            
accelY = [0]*windowSize     # The list of acceleration data along the y-axis being recorded from the bluetooth device            
accelZ = [0]*windowSize     # The list of acceleration data along the z-axis being recorded from the bluetooth device
rotX = [0]*windowSize       # The list of gyroscope data around the x-axis being recorded from the bluetooth device
rotY = [0]*windowSize       # The list of gyroscope data around the y-axis being recorded from the bluetooth device
rotZ = [0]*windowSize       # The list of gyroscope data around the z-axis being recorded from the bluetooth device
rotVX = [0]*windowSize      # The list of rotation vector data around the x-axis being recorded from the bluetooth device
rotVY = [0]*windowSize      # The list of rotation vector data around the y-axis being recorded from the bluetooth device
rotVZ = [0,1]*(int(windowSize/2)) # The list of rotation vector data around the z-axis being recorded from the bluetooth device
buffer = ""                 # Contains the excess of a bluetooth message that was not finished/fully recieved
fig = None                  # The matplotlib figure
ax = None                   # The matplotlib axis
lineAccelX = None           # The matplotlib x acceleration line
lineAccelY = None           # The matplotlib y acceleration line
lineAccelZ = None           # The matplotlib z acceleration line
lineRotX = None             # The matplotlib x rotation (gyroscope) line
lineRotY = None             # The matplotlib y rotation (gyroscope) line
lineRotZ = None             # The matplotlib z rotation (gyroscope) line
lineRotVX = None            # The matplotlib x rotation vector line
lineRotVY = None            # The matplotlib y rotation vector line
lineRotVZ = None            # The matplotlib z rotation vector line
axisX = np.linspace(0, windowSize, windowSize, endpoint=False) # The x-axis from 0 to windowSize
beatStart = [-10]*windowSize # The matplotlib list of values that shows the search area for the start of a beat
predBeatStart = [-10]*windowSize # The matplotlib list of values that shows the predicted start location of a beat
lineBeatStart = None        # The matplotlib line that shows the search area for the start of a beat
linePredBeatStart = None    # The matplotlib line that shows the predicted start location of a beat
recordData = False
gatherRawData = False
beatIdx = 0
newBeat = False
currentBeat = None
testData = False
roundNum = 0
ignoringDetection = 0

##############################################################################



############################### Functions ####################################

# Utility Functions

def Log(string, newline = True):
    """ Simple print logging function that only prints output when the shouldLog flag is True. """

    if shouldLog:
        if newline:
            print(string)
        else:
            print(string, end = '')

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def GraphFile(file):
    """ Generates a graph of the recorded sensor data from the given file. """
    
    global fig, ax, lineAccelX, lineAccelY, lineAccelZ, lineRotX, lineRotY, lineRotZ, lineRotVX, lineRotVY, lineRotVZ
    data = LoadFile(file)
    dataLine = []
    for line in data:
        dataLine.extend(line)
    aX = dataLine[0::9]
    aY = dataLine[1::9]
    aZ = dataLine[2::9]
    rX = dataLine[3::9]
    rY = dataLine[4::9]
    rZ = dataLine[5::9]
    rVX = dataLine[6::9]
    rVY = dataLine[7::9]
    rVZ = dataLine[8::9]
    axisX = np.linspace(0, len(dataLine)/9, len(dataLine)/9, endpoint=False)

    
    Log(aX)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lineAccelX, lineAccelY, lineAccelZ, lineRotX, lineRotY, lineRotZ, lineRotVX, lineRotVY, lineRotVZ = ax.plot(axisX,np.array(aX),axisX,np.array(aY),axisX,np.array(aZ),axisX,np.array(rX),axisX,np.array(rY),axisX,np.array(rZ),axisX,np.array(rVX),axisX,np.array(rVY),axisX,np.array(rVZ))
    lineAccelX.set_label('Accel X')
    lineAccelY.set_label('Accel Y')
    lineAccelZ.set_label('Accel Z')
    lineRotX.set_label('Rotation X')
    lineRotY.set_label('Rotation Y')
    lineRotZ.set_label('Rotation Z')
    lineRotVX.set_label('Rotation Vector X')
    lineRotVY.set_label('Rotation Vector Y')
    lineRotVZ.set_label('Rotation Vector Z')
    ax.legend()

    fig.canvas.draw()
    fig.canvas.flush_events()
    ax.draw()

def GetMinMax(listOfNumbers):
    """ Finds the min and max values of a given list of numbers. """
    
    minValue = listOfNumbers[0]
    maxValue = listOfNumbers[0]
    for num in listOfNumbers:
        if num < minValue:
            minValue = num
        if num > maxValue:
            maxValue = num
    return [minValue,maxValue]


def GetAverage(listOfNumbers):
    """ Finds the average from a given list of numbers. """
    
    total = 0
    for num in listOfNumbers:
        total = total + num
    return total/len(listOfNumbers)

def PlotData():
    """ Updates the graph to shopw the most recent data that is being recieved from the bluetooth device. """
    
    global fig, ax, lineAccelX, lineAccelY, lineAccelZ, lineRotX, lineRotY, lineRotZ, lineBeatStart, linePredBeatStart, lineRotVX, lineRotVY, lineRotVZ

    oldRotVZ = rotVZ.copy()
    newRotVZ = []
    minRotVZ, maxRotVZ = GetMinMax(oldRotVZ)
    disp = 0.5-maxRotVZ
    for x in range(len(oldRotVZ)):
        value = oldRotVZ[x]+disp
        while value > 1:
            value = value - 1
        while value < -1:
            value = value + 1
        newRotVZ.append(value)


    if len(accelX) > 0:
        lineAccelX.set_ydata(np.array(accelX))
        lineAccelY.set_ydata(np.array(accelY))
        lineAccelZ.set_ydata(np.array(accelZ))
        lineRotX.set_ydata(np.array(rotX))
        lineRotY.set_ydata(np.array(rotY))
        lineRotZ.set_ydata(np.array(rotZ))
        lineRotVX.set_ydata(np.array(rotVX))
        lineRotVY.set_ydata(np.array(rotVY))
        lineRotVZ.set_ydata(np.array(rotVZ))
        lineBeatStart.set_ydata(np.array(beatStart))
        linePredBeatStart.set_ydata(np.array(predBeatStart))

        # Auto scale axis
        #ax.relim()
        #ax.autoscale_view()        

        fig.canvas.draw()
        fig.canvas.flush_events()
        #print("Updating Graph...")
        #print(accelX)


def InitializeGraph():
    """ Initializes the graph to display the recording sensor data from the bluetooth device. """
    
    global fig, ax, lineAccelX, lineAccelY, lineAccelZ, lineRotX, lineRotY, lineRotZ, lineBeatStart, linePredBeatStart, lineRotVX, lineRotVY, lineRotVZ
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lineAccelX, lineAccelY, lineAccelZ, lineRotX, lineRotY, lineRotZ, lineBeatStart, linePredBeatStart, lineRotVX, lineRotVY, lineRotVZ = ax.plot(axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.random.rand(windowSize))
    lineAccelX.set_label('Accel X')
    lineAccelY.set_label('Accel Y')
    lineAccelZ.set_label('Accel Z')
    lineRotX.set_label('Rotation X')
    lineRotY.set_label('Rotation Y')
    lineRotZ.set_label('Rotation Z')
    lineRotVX.set_label('Rot Vector X')
    lineRotVY.set_label('Rot Vector Y')
    lineRotVZ.set_label('Rot Vector Z')
    lineBeatStart.set_label('New Beat')
    linePredBeatStart.set_label('Predicted New Beat')
    plt.ylim(-1,1)
    ax.legend()

def ShowImage(fileName):
    """ Generates and displays the image that is used as an input for the neural network. """
    
    f = LoadFile(fileName)
    images = Generate2DImageListForm(f)
    
    for i in range(len(images)):
        image = images[i]
        x = []
        y = []
        color = (random.uniform(0,1),random.uniform(0,1),random.uniform(0,1))
        colors = []
        area = []
        for xPos in range(size):
            for yPos in range(size):
                value = image[xPos][yPos]
                if value > 0:
                    radius = value*10
                    area.append(radius)
                    colors.append(color)
                    x.append(xPos)
                    y.append(yPos)
        plt.scatter(x, y, s=area, c=colors, alpha=0.2)
    plt.ylim(-10,60)
    plt.xlim(-10,60)
    plt.show()

def Generate2DImageListForm(data):
    """ Generates the 2D image as a list with each element as one of the features. (Used in plotting the graph) """
    
    images = []
    images.append(Generate2DImageFeatureListForm(data, 0, scaleAccel))
    images.append(Generate2DImageFeatureListForm(data, 1, scaleAccel))
    images.append(Generate2DImageFeatureListForm(data, 2, scaleAccel))
    images.append(Generate2DImageFeatureListForm(data, 3, scaleRot))
    images.append(Generate2DImageFeatureListForm(data, 4, scaleRot))
    images.append(Generate2DImageFeatureListForm(data, 5, scaleRot))
    return images

def Generate2DImageFeatureListForm(data, col, scale):
    """ Generates the 2D image of the given feature. (Used in plotting the graph) """
    image = np.zeros((size,size))

    x = 0
    xIncrease = 50/len(data)

    for line in data:
        y = line[col]
        if y > 1:
            y = 1
        if y < -1:
            y = -1
        y = y + 1
        y = y*size/2
        y = int(y)

        print(str(col)+" at "+str(x)+" "+str(y))
        
        brushDist = 0
        for xBrush in range(-brushRadius,brushRadius+1):
            for yBrush in range(-brushRadius,brushRadius+1):
                if(int(x)+xBrush < size and y+yBrush < size):
                    brushDist = abs(xBrush)
                    if abs(yBrush) > brushDist:
                        brushDist = abs(yBrush)
                    if(int(x)+xBrush) > 0 and (y+yBrush) > 0:
                        if(image[int(x)+xBrush][y+yBrush] < (1-(float(brushDist)/brushRadius))):
                            image[int(x)+xBrush][y+yBrush] = (1-(float(brushDist)/brushRadius))
                
        x = x + xIncrease
        
    return image

def GetFiles(directory):
    files = os.listdir(directory)
    files.sort(key=natural_keys)
    fullFiles = []
    for file in files:
        fullFiles.append(directory + "/" + file)
    return fullFiles


# Core Program Functions

def Main():
    """ Main function that should begin the program in normal circumstances. """
    
    run = True
    while run:
        command = input("Enter command: ")
        run = Command(command)
        

def Command(command):
    """ Executes the given command. """
    
    global recordData, beatName, userName, testData, roundNum, gatherRawData
    if command == "bluetooth":
        thread.start_new_thread(BluetoothMethod, ())
        InitializeGraph()
        while runProgram:
            PlotData()
    elif command == "generate":
        beatName = input("Enter Beat Label: ")
        userName = input("Enter Your Name: ")
        roundNum = input("Enter Round of Recording: ")
        recordData = True
        thread.start_new_thread(BluetoothMethod, ())
        InitializeGraph()
        while runProgram:
            PlotData()
    elif command == "generateFromFile":
        beatName = input("Enter Beat Label: ")
        userName = input("Enter Your Name: ")
        roundNum = input("Enter Round of Recording: ")
        recordData = True
        thread.start_new_thread(StreamFromFile, ())
        InitializeGraph()
        while runProgram:
            PlotData()
    elif command == "test":
        test()
    elif command == "train":
        CreateModel()
    elif command == "run":
        testData = True
        thread.start_new_thread(BluetoothMethod, ())
        thread.start_new_thread(RunProgram, ())
        InitializeGraph()
        while runProgram:
            PlotData()
    elif command == "runFromFile":
        testData = True
        beatName = input("Enter Beat Label: ")
        userName = input("Enter Your Name: ")
        roundNum = input("Enter Round of Recording: ")
        thread.start_new_thread(StreamFromFile, ())
        thread.start_new_thread(RunProgram, ())
        InitializeGraph()
        while runProgram:
            PlotData()
    elif command == "viewGraph":
        file = input("Enter Path to File: ")
        GraphFile(file)
    elif command == "viewImage":
        file = input("Enter Path to File: ")
        ShowImage(file)
    elif command == "gatherRaw":
        gatherRawData = True
        beatName = input("Enter Beat Label: ")
        userName = input("Enter Your Name: ")
        roundNum = input("Enter Round of Recording: ")
        thread.start_new_thread(BluetoothMethod, ())
        InitializeGraph()
        i = windowSize
        while runProgram:
            PlotData()
    elif command == "exit":
        return False
    return True

def GatherRawData():
    SaveData(GetData(0,windowSize-1))
    

def RunProgram():
    """ Runs the classifier program by loading the trained model and predicting the given gesture made by the device connected via bluetooth. """
    
    global newBeat

    # Load Model
    tf.reset_default_graph()
    nInputNeurons = size*size*6
    nOutputNeurons = len(labels)
    
    # Preparing training data (inputs-outputs)  
    inputs = tf.placeholder(shape=[None, nInputNeurons], dtype=tf.float32)  
    outputs = tf.placeholder(shape=[None, nOutputNeurons], dtype=tf.float32) #Desired outputs for each input  
        
    # Weight initializations
    w_1 = init_weights((nInputNeurons, nHiddenNeurons))
    w_2 = init_weights((nHiddenNeurons, nOutputNeurons))
        
    # Forward propagation
    yhat    = forwardprop(inputs, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)
        
    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=outputs, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    saver = tf.train.Saver()
    
    sess = tf.Session()
    saver.restore(sess, "./model.ckpt")

    lastThreePredictions = [0,0,0]
    while runProgram:
        if newBeat:
            newBeat = False
            d = currentBeat.copy()
            data = Generate2DImage(d)
            out = sess.run(predict, feed_dict={inputs: [data], outputs: [[1,0,0,0,0,0,0,0,0]]})
            lastThreePredictions.append(out[0])
            del lastThreePredictions[0]

            classes = []
            for i in range(len(labels)):
                classes.append(0)
            classes[lastThreePredictions[0]] = classes[lastThreePredictions[0]] + 1
            classes[lastThreePredictions[1]] = classes[lastThreePredictions[1]] + 1
            classes[lastThreePredictions[2]] = classes[lastThreePredictions[2]] + 1
            mostLikely = 0
            mostLikelyValue = 0
            for i in range(len(classes)):
                if classes[i] > mostLikelyValue:
                    mostLikelyValue = classes[i]
                    mostLikely = i
            Log("Predicted gesture is " + labels[mostLikely])
    sess.close()

def CreateModel():
    """ Generates a trained neural network model for the classifier to use. """
    
    global loaded
    nInputNeurons = size*size*6
    nOutputNeurons = len(labels)
    files = os.listdir("Training Data")
    files.sort()
    trainingFiles = []
    for f in files:
        trainingFiles.append("Training Data/"+f)
  


    Log("# of Training Files = "+str(len(trainingFiles)))

        
    # Preparing training data (inputs-outputs)  
    inputs = tf.placeholder(shape=[None, nInputNeurons], dtype=tf.float32)  
    outputs = tf.placeholder(shape=[None, nOutputNeurons], dtype=tf.float32) #Desired outputs for each input  
        
    # Weight initializations
    w_1 = init_weights((nInputNeurons, nHiddenNeurons))
    w_2 = init_weights((nHiddenNeurons, nOutputNeurons))
        
    # Forward propagation
    yhat    = forwardprop(inputs, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)
        
    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=outputs, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    saver = tf.train.Saver()

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    Log("Initial Results:")
    Log("Should have been:")
    start = time.time()
    timeRecorded = False
    for epoch in range(nEpochs):
        # Train with each example
        first = True
        d = []
        Log("Epoch Progress: 0___________________100")
        Log("                ", False)
        l = len(trainingFiles)
        for i in range(l):
            if(i%int(l*0.05)) == 0:
                Log("#", False)
            
            startT = time.time()
            d = LoadData(trainingFiles[i])
            #if first:
            #    d = LoadData(trainingFiles[i])
            #    first = False
            #else:
            #    while True:
            #        if loaded:
            #            d = loadedFile.copy()
            #            loaded = False
            #            break
            #thread.start_new_thread(LoadDataThread, (trainingFiles[i+1],))
            sess.run(updates, feed_dict={inputs: [d], outputs: [LoadResult(trainingFiles[i])]})
            endT = time.time()

            #Log(str(endT-startT) + " seconds to load and train")
        Log(" Done epoch.")

        train_accuracy = 0
        for i in range(len(trainingFiles)):
            r = sess.run(predict, feed_dict={inputs: [LoadData(trainingFiles[i])], outputs: [LoadResult(trainingFiles[i])]})
            if r[0] == GetIdx(trainingFiles[i]):
                train_accuracy = train_accuracy + 1
        train_accuracy = train_accuracy / len(trainingFiles)
        #train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
        #                            sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
            
        Log("Epoch = %d, train accuracy = %.2f%%"
                % (epoch + 1, 100. * train_accuracy))
        if not timeRecorded:
            timeRecorded = True
            end = time.time()
        remainingTime = (end - start)*(nEpochs-epoch)/60
        Log(str(remainingTime) + " minutes remaining")
        

    #Log("Training Results:")
    #Log(sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
    #Log("Should have been:")
    #Log(train_y)

    save_path = saver.save(sess, "./model.ckpt")
    Log("Model saved in path: %s" % save_path)
    sess.close()

def test():
    """ Runs a cross-validation testing process on the dataset and configuration of the model parameters. """
    
    nInputNeurons = size*size*6
    nOutputNeurons = len(labels)

    # Basic Version
    #trainingFiles = loadDirectory("train")    
    #testingFiles = loadDirectory("test")

    # Random Train and Test Split Version
    #datasets = splitDataset("allData",0.4)
    #trainingFiles = loadDataset(datasets[0])
    #testingFiles = loadDataset(datasets[1])

    # Cross-Validation Version
    datasets = splitDatasetFolds("Training Data")
    Log(len(datasets))
    results = []

    for k in range(0,nFolds):

        Log("Loading Testing Files... (Fold #" + str(k)+")")
        testingFiles = datasets[k]
        trainingFiles = []
        Log("Loading Training Files...")
        for j in range(0,k):
            Log("Loading Fold #"+str(j))
            trainingFiles.extend(datasets[j])
            Log("# of Training Files = "+str(len(trainingFiles)))
        for j in range(k+1,nFolds):
            Log("Loading Fold #"+str(j))
            trainingFiles.extend(datasets[j])
            Log("# of Training Files = "+str(len(trainingFiles)))

        #trainingFiles = shuffle(trainingFiles)
        #testingFiles = shuffle(testingFiles)

        Log("# of Training Files = "+str(len(trainingFiles)))
        Log("# of Testing Files = "+str(len(testingFiles)))

        # Preparing training data (inputs-outputs)  
        inputs = tf.placeholder(shape=[None, nInputNeurons], dtype=tf.float32)  
        outputs = tf.placeholder(shape=[None, nOutputNeurons], dtype=tf.float32) #Desired outputs for each input  
        
        # Weight initializations
        w_1 = init_weights((nInputNeurons, nHiddenNeurons))
        w_2 = init_weights((nHiddenNeurons, nOutputNeurons))
        
        # Forward propagation
        yhat    = forwardprop(inputs, w_1, w_2)
        predict = tf.argmax(yhat, axis=1)
        
        # Backward propagation
        cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=outputs, logits=yhat))
        updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
        
        # Run SGD
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        test_accuracy = 0
        test_results = []
        test_correct = []
        train_results = []
        train_correct = []
        for epoch in range(nEpochs):
            # Train with each example
            Log("Epoch Progress: 0___________________100")
            Log("                ", False)
            l = len(trainingFiles)
            for i in range(l):
                if(i%int(l*0.05)) == 0:
                    Log("#", False)
                sess.run(updates, feed_dict={inputs: [LoadData(trainingFiles[i])], outputs: [LoadResult(trainingFiles[i])]})
                

            Log(" Done epoch.")

            if detailedEpochs:
                #train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                #                         sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
                train_accuracy = 0
                for i in range(len(trainingFiles)):
                    r = sess.run(predict, feed_dict={inputs: [LoadData(trainingFiles[i])], outputs: [LoadResult(trainingFiles[i])]})
                    train_results.append(r[0])
                    c = GetIdx(trainingFiles[i])
                    train_correct.append(c)
                    if r[0] == c:
                        train_accuracy = train_accuracy + 1
                train_accuracy = train_accuracy / len(trainingFiles)

                test_accuracy = 0
                for i in range(len(testingFiles)):
                    r = sess.run(predict, feed_dict={inputs: [LoadData(testingFiles[i])], outputs: [LoadResult(testingFiles[i])]})
                    test_results.append(r[0])
                    c = GetIdx(testingFiles[i])
                    test_correct.append(c)
                    if r[0] == c:
                        test_accuracy = test_accuracy + 1
                test_accuracy = test_accuracy / len(testingFiles)    
                        
                #test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                #                         sess.run(predict, feed_dict={inputs: test_X, outputs: test_y}))

                Log("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                      % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

        train_accuracy = 0
        for i in range(len(trainingFiles)):
            r = sess.run(predict, feed_dict={inputs: [LoadData(trainingFiles[i])], outputs: [LoadResult(trainingFiles[i])]})
            train_results.append(r[0])
            c = GetIdx(trainingFiles[i])
            train_correct.append(c)
            if r[0] == c:
                train_accuracy = train_accuracy + 1
        train_accuracy = train_accuracy / len(trainingFiles)

        test_accuracy = 0
        for i in range(len(testingFiles)):
            r = sess.run(predict, feed_dict={inputs: [LoadData(testingFiles[i])], outputs: [LoadResult(testingFiles[i])]})
            test_results.append(r[0])
            c = GetIdx(testingFiles[i])
            test_correct.append(c)
            if r[0] == c:
                test_accuracy = test_accuracy + 1
        test_accuracy = test_accuracy / len(testingFiles)   
        #Log("Training Results: " + str(train_accuracy))
        #Log(train_results)
        #Log("Should have been:")
        #Log(train_correct)
        #Log("Testing Results: "+ str(test_accuracy))
        #Log(test_results)
        #Log("Should have been:")
        #Log(test_correct)
        sess.close()

        results.append(test_accuracy)

    Log("Results:")
    total = 0
    for r in results:
        Log(str(r*100)+"%")
        total = total + r
    Log("Avr = " + str(total/nFolds*100) + "%")
    return total/nFolds

def StreamFromFile():
    global accelX, accelY, accelZ, rotX, rotY, rotZ, beatStart, rotVX, rotVY, rotVZ, beatIdx, newBeatEndPos, newBeatStartPos, newBeat, currentBeat, ignoringDetection
    beat = False
    files = GetFiles("Recorded Data Archive/Gathered Data Split By Window Size/Collected Stream of Sensor Data 2.0")
    for filename in files:
        if (beatName+"_sensor_"+userName+"_round_"+roundNum+"_") in filename:
            file = open(filename, "r")
            print("Reading file " + filename)
            for line in file:
                #print("Reading line " + str(line))
                time.sleep(0.1)

                messageSplit = line.split(",")
                accelX.append(float(messageSplit[0]))
                accelY.append(float(messageSplit[1]))
                accelZ.append(float(messageSplit[2]))
                rotX.append(float(messageSplit[3]))
                rotY.append(float(messageSplit[4]))
                rotZ.append(float(messageSplit[5]))
                rotVX.append(float(messageSplit[6]))
                rotVY.append(float(messageSplit[7]))
                rotVZ.append(float(messageSplit[8]))
                del accelX[0]
                del accelY[0]
                del accelZ[0]
                del rotX[0]
                del rotY[0]
                del rotZ[0]
                del rotVX[0]
                del rotVY[0]
                del rotVZ[0]
                predBeat = False
                if IsNewBeat():
                    newIdx = GetBeatIdx(newBeatStartPos,newBeatEndPos)
                    for i in range(newBeatStartPos,newBeatEndPos+1):
                        predBeatStart[i]=-0.9
                    newBeatStartPos = -1
                    newBeatEndPos = -1
                    beatStart[newIdx]=10
                    if recordData:
                        d = GetData(beatIdx, newIdx)
                        SaveData(d)
                    if testData:
                        currentBeat = GetData(beatIdx, newIdx)
                        newBeat = True
                    beatIdx = newIdx
                if gatherRawData:
                    gatherDelay = gatherDelay - 1
                    if gatherDelay < 0:
                        gatherDelay = windowSize
                        GatherRawData()
                predBeatStart.append(-10)
                del predBeatStart[0]
                if beat:
                    beatStart.append(10)
                    beat = False
                else:
                    beatStart.append(-10)
                del beatStart[0]
                beatIdx = beatIdx - 1
                if beatIdx < 0:
                    beatIdx = 0
                if newBeatStartPos > -1:
                    newBeatStartPos = newBeatStartPos - 1
                    if newBeatStartPos == -1:
                        newBeatStartPos = 0
                if newBeatEndPos > -1:
                    newBeatEndPos = newBeatEndPos - 1
                    if newBeatEndPos == -1:
                        newBeatEndPos = 0
                ignoringDetection = ignoringDetection - 1
                if ignoringDetection < 0:
                    ignoringDetection = 0 

def BluetoothMethod():
    """ Starts running the bluetooth portion of the program by waiting for an incoming connection and then updating the data as it recieves messages from the connected device. """

    global buffer, accelX, accelY, accelZ, rotX, rotY, rotZ, beatStart, rotVX, rotVY, rotVZ, beatIdx, newBeatEndPos, newBeatStartPos, newBeat, currentBeat, ignoringDetection
    server_sock = BluetoothSocket( RFCOMM )
    server_sock.bind(("",PORT_ANY))
    server_sock.listen(1)

    port = server_sock.getsockname()[1]

    uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"

    advertise_service( server_sock, "TestServer",
                       service_id = uuid,
                       service_classes = [ uuid, SERIAL_PORT_CLASS ],
                       profiles = [ SERIAL_PORT_PROFILE ], 
    #                   protocols = [ OBEX_UUID ] 
                        )

    Log("Waiting for connection on RFCOMM channel %d" % port)
    client_sock, client_info = server_sock.accept()
    Log("Accepted connection from ", client_info)

    gatherDelay = windowSize

    while runProgram:          

        try:
            req = client_sock.recv(1024)
            if len(req) == 0:
                break
            #Log("received [%s]" % req)
            messages = req.decode("utf-8")
            #Log("RECIEVED: " + messages)
            messages = buffer + messages
            buffer = ""
            messages = messages.split("BEAT")
            beat = False
            if len(messages) > 1:
                beat = True
            for message in messages:
                message = message.split(",END")
                for m in message[0:-1]:
                    messageSplit = m.split(",")
                    if len(messageSplit) == 9 and messageSplit[-1] != "":
                        accelX.append(float(messageSplit[0])*scaleAccel)
                        accelY.append(float(messageSplit[1])*scaleAccel)
                        accelZ.append(float(messageSplit[2])*scaleAccel)
                        rotX.append(float(messageSplit[3])*scaleRot)
                        rotY.append(float(messageSplit[4])*scaleRot)
                        rotZ.append(float(messageSplit[5])*scaleRot)
                        rotVX.append(float(messageSplit[6])*scaleRotV)
                        rotVY.append(float(messageSplit[7])*scaleRotV)
                        rotVZ.append(float(messageSplit[8])*scaleRotV)
                        del accelX[0]
                        del accelY[0]
                        del accelZ[0]
                        del rotX[0]
                        del rotY[0]
                        del rotZ[0]
                        del rotVX[0]
                        del rotVY[0]
                        del rotVZ[0]
                        predBeat = False
                        if IsNewBeat():
                            newIdx = GetBeatIdx(newBeatStartPos,newBeatEndPos)
                            for i in range(newBeatStartPos,newBeatEndPos+1):
                                predBeatStart[i]=-0.9
                            newBeatStartPos = -1
                            newBeatEndPos = -1
                            beatStart[newIdx]=10
                            if recordData:
                                d = GetData(beatIdx, newIdx)
                                SaveData(d)
                            if testData:
                                currentBeat = GetData(beatIdx, newIdx)
                                newBeat = True
                            beatIdx = newIdx
                        if gatherRawData:
                            gatherDelay = gatherDelay - 1
                            if gatherDelay < 0:
                                gatherDelay = windowSize
                                GatherRawData()
                        predBeatStart.append(-10)
                        del predBeatStart[0]
                        if beat:
                            beatStart.append(10)
                            beat = False
                        else:
                            beatStart.append(-10)
                        del beatStart[0]
                        beatIdx = beatIdx - 1
                        if beatIdx < 0:
                            beatIdx = 0
                        if newBeatStartPos > -1:
                            newBeatStartPos = newBeatStartPos - 1
                            if newBeatStartPos == -1:
                                newBeatStartPos = 0
                        if newBeatEndPos > -1:
                            newBeatEndPos = newBeatEndPos - 1
                            if newBeatEndPos == -1:
                                newBeatEndPos = 0
                        ignoringDetection = ignoringDetection - 1
                        if ignoringDetection < 0:
                            ignoringDetection = 0
                        
                    else:
                        Log("Error recieving message " + m)
                buffer = message[-1]


            data = None
            if req in ('temp', '*temp'):
                data = str(random.random())+'!'
            else:
                pass

            if data:
                Log("sending [%s]" % data)
                client_sock.send()

        except IOError:
            pass

        except KeyboardInterrupt:

            Log("disconnected")

            client_sock.close()
            server_sock.close()
            Log("all done")

            break


# Other Program Functions

def splitDatasetFolds(directory):
    """ Splits the dataset of the given directory into a list that contains a list for each of the labels.
        Note: Currently the directory MUST contain the same number of file for each label. """

    files = os.listdir(directory)
    files.sort()
    beats = []
    count = int(len(files)/len(labels))
    for i in range(0,len(labels)):
        beats.append(files[i*count:(i+1)*count])
    splitUpTo = count/nFolds
    folds = []
    numUnsorted = count
    for i in range(0,nFolds):
        folds.append([[],[],[],[],[],[],[],[],[]])
        c = 0
        while numUnsorted > 0 and c < splitUpTo:
            idx = random.randint(0,numUnsorted-1)
            folds[i][0].append(directory + "/" + beats[0][idx])
            del beats[0][idx]
            idx = random.randint(0,numUnsorted-1)
            folds[i][1].append(directory + "/" + beats[1][idx])
            del beats[1][idx]
            idx = random.randint(0,numUnsorted-1)
            folds[i][2].append(directory + "/" + beats[2][idx])
            del beats[2][idx]
            idx = random.randint(0,numUnsorted-1)
            folds[i][3].append(directory + "/" + beats[3][idx])
            del beats[3][idx]
            idx = random.randint(0,numUnsorted-1)
            folds[i][4].append(directory + "/" + beats[4][idx])
            del beats[4][idx]
            idx = random.randint(0,numUnsorted-1)
            folds[i][5].append(directory + "/" + beats[5][idx])
            del beats[5][idx]
            idx = random.randint(0,numUnsorted-1)
            folds[i][6].append(directory + "/" + beats[6][idx])
            del beats[6][idx]
            idx = random.randint(0,numUnsorted-1)
            folds[i][7].append(directory + "/" + beats[7][idx])
            del beats[7][idx]
            idx = random.randint(0,numUnsorted-1)
            folds[i][8].append(directory + "/" + beats[8][idx])
            del beats[8][idx]
            numUnsorted = numUnsorted - 1
            c = c + 1
    if(numUnsorted > 0):
        for b in beats:
            for f in b:
                folds[0].append(f)

    compiledFolds = []
    for i in range(0,nFolds):
        compiledFolds.append([])
        for f in folds[i]:
            compiledFolds[i].extend(f)

    # Prints
    Log("Folds:")
    for i in range(0,nFolds):
        Log("Fold #"+str(i)+":")
        for f in compiledFolds[i]:
            Log(f)
    
    return compiledFolds

def LoadData(fileName):
    """ Loads the data from the given file as a 2D image. """
    
    f = LoadFile(fileName)
    return Generate2DImage(f)

def GetIdx(fileName):
    """ Gets the index of the correct label for the given file. """
    
    label = fileName.split("/")[1].split("_")[0]
    idx = labels.index(label)
    return idx

def LoadResult(fileName):
    """ Generates the correct output list that the neural network shoudl produce. """
    
    idx = GetIdx(fileName)
    output = []
    for i in range(len(labels)):
        if i == idx:
            output.append(1)
        else:
            output.append(0)
    return output

def LoadFile(fileName):
    """ Loads the direct data from the given sensor recording data file. """
    
    #Log("Loading file " + fileName)
    lines = [line.rstrip('\n') for line in open(fileName)]
    data = []
    for line in lines:
        items = line.split(",")
        pos = []
        for item in items:
            if len(item) > 0:
                pos.append(float(item))
        data.append(pos)
    return data




def Generate2DImage(data):
    """ Generates the 2D image of the input file. (Used in the neural network) """
    
    image = []
    image.extend(Generate2DImageFeature(data, 0, scaleAccel))
    image.extend(Generate2DImageFeature(data, 1, scaleAccel))
    image.extend(Generate2DImageFeature(data, 2, scaleAccel))
    image.extend(Generate2DImageFeature(data, 3, scaleRot))
    image.extend(Generate2DImageFeature(data, 4, scaleRot))
    image.extend(Generate2DImageFeature(data, 5, scaleRot))
    return image

def Generate2DImageFeature(data, col, scale):
    """ Generates the 2D of the given feature. (Used in the neural network) """
    
    image = np.zeros((size,size))

    x = 0
    xIncrease = 50/len(data)

    for line in data:
        y = line[col]
        if y > 1:
            y = 1
        if y < -1:
            y = -1
        y = y + 1
        y = y*size/2
        y = int(y)
        
        brushDist = 0
        for xBrush in range(-brushRadius,brushRadius+1):
            for yBrush in range(-brushRadius,brushRadius+1):
                if(int(x)+xBrush < size and y+yBrush < size):
                    brushDist = abs(xBrush)
                    if abs(yBrush) > brushDist:
                        brushDist = abs(yBrush)
                    if(int(x)+xBrush) > 0 and (y+yBrush) > 0:
                        if(image[int(x)+xBrush][y+yBrush] < (1-(float(brushDist)/brushRadius))):
                            image[int(x)+xBrush][y+yBrush] = (1-(float(brushDist)/brushRadius))
                
        x = x + xIncrease
        
    oneLineImage = []
    for i in np.nditer(image):
        oneLineImage.append(i)
    return oneLineImage
        

def GetBeatIdx(start,end):
    """ Returns the predicted start/end of the current and next beat gesture.
        This is determined by the location with the lowest rotation vector around the Z-axis (the device's furthest rotation to the right). """
    
    highestValue = rotVX[start]
    highestIdx = start
    for i in range (start,end+1):
        if rotVX[i] > highestValue:
            highestValue = rotVX[i]
            highestIdx = i
    #print(highestValue)
    return highestIdx

def GetData(start,end):
    """ Returns a list of the currently recorded data from the given start and end points. """
    
    data = []
    for i in range(start,end+1):
        data.append([accelX[i],accelY[i],accelZ[i],rotX[i],rotY[i],rotZ[i],rotVX[i],rotVY[i],rotVZ[i]])
    return data


fileNum = 0
def SaveData(data):
    """ Saves the given data into the Generated Data directory. """
    
    global fileNum
    with open("Generated Data/"+beatName+"_sensor_"+userName+"_round_"+roundNum+"_"+str(fileNum), "w") as savefile:
        for line in data:
            for item in line:
                savefile.write(str(item)+",")
            savefile.write("\n")
    fileNum = fileNum + 1



newBeatStartPos = -1
newBeatEndPos = -1
ignoringDetection = 0

# Rotation Vector X Based Detection
rotVXInRange = False
def IsNewBeat():
    """ Determines if the program should decide that the current beat gesture has been completed and a new one is beginning. """
    
    global newBeatStartPos, newBeatEndPos, ignoringDetection, rotVXInRange
    if ignoringDetection < 1:
        minRotVX, maxRotVX = GetMinMax(rotVX)
        half = (maxRotVX-minRotVX)/2
        level = maxRotVX - half/2
        if rotVX[-1] > level and not rotVXInRange:
            rotVXInRange = True
            newBeatStartPos = windowSize - 1
        elif rotVX[-1] < level and rotVXInRange:
            rotVXInRange = False
            newBeatEndPos = windowSize - 1
            ignoringDetection = ignoreWindow
            return True
    return False

# Rotation Vector Z Based Detection
#rotVZInRange = False
#def IsNewBeat():
#    """ Determines if the program should decide that the current beat gesture has been completed and a new one is beginning. """
#    
#    global newBeatStartPos, newBeatEndPos, ignoringDetection, rotVZInRange
#    if ignoringDetection < 1:
#        minRotVZ, maxRotVZ = GetMinMax(rotVZ)
#        half = (maxRotVZ-minRotVZ)/2
#        level = minRotVZ + half/2
#        if rotVZ[-1] < level and not rotVZInRange:
#            rotVZInRange = True
#            newBeatStartPos = windowSize - 1
#        elif rotVZ[-1] > level and rotVZInRange:
#            rotVZInRange = False
#            newBeatEndPos = windowSize - 1
#            ignoringDetection = ignoreWindow
#            return True
#    return False


# Neural Network Functions

def init_weights(shape):
    """ Weights initialization. """
    
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """ Forward-propagation. """
    
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))
    yhat = tf.matmul(h, w_2)
    return yhat

##############################################################################


# Start the program normally
Main()
