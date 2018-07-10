import numpy as np
import tensorflow as tf
import os
import math
import random
import _thread as thread
import time
from bluetooth import *
from sklearn.model_selection import train_test_split

# Main Settings
size            =   50      # Length of each dimension of the 3D image, larger = more detailed gestures
brushRadius     =   4       # Radius that values are applied to the 3D image from source point, larger = more generalized
nHiddenNeurons  =   400     # Number of hidden neurons in the neural network, larger = usually better at recognizing more details
nEpochs         =   20      # Number of training epochs for the neural network, larger = usually better accuracy
labels          =   ["beat2","beat3","beat4"]        # Labels of the gestures to recognize (Note: training files should have the naming convention of [labelname]_##.csv
iterPerSecond   =   0.025   # Speed at which the data is being recorded
nFolds          =   3       # Number of cross validation folds
windowSize      =   100     # Number of lines of positional data that are kept recorded

# Global Variables

listOfPositions = []        # The recorded positional data
runProgram = True           # Flag to run the main bluetooth data recording & prediction loops


def loadPositionFile(fileName):
    #print("Loading file " + fileName)
    lines = [line.rstrip('\n') for line in open(fileName)]
    data = []
    for line in lines:
        items = line.split(",")
        pos = []
        for item in items:
            pos.append(float(item))
        data.append(pos)
    return data

def getMaxSpeed(positionData, scale, minX, minY, minZ):
    maxSpeed = 0
    prevScaledPos = [0,0,0]
    for line in positionData:
        unroundedX = (line[0]-minX)*scale
        unroundedY = (line[1]-minY)*scale
        unroundedZ = (line[2]-minZ)*scale
        unroundedXDif = unroundedX-prevScaledPos[0]
        unroundedYDif = unroundedY-prevScaledPos[1]
        unroundedZDif = unroundedZ-prevScaledPos[2]
        speed = unroundedXDif*unroundedXDif+unroundedYDif*unroundedYDif+unroundedZDif*unroundedZDif
        speed = math.sqrt(speed)
        speed = speed/iterPerSecond
        prevScaledPos = [unroundedX,unroundedY,unroundedZ]
        if speed > maxSpeed:
            maxSpeed = speed
    return maxSpeed

def Convert(positionData):
    imagePosition = np.zeros((size,size,size))
    imageUp = np.zeros((size,size,size))
    imageDown = np.zeros((size,size,size))
    imageLeft = np.zeros((size,size,size))
    imageRight = np.zeros((size,size,size))
    imageForward = np.zeros((size,size,size))
    imageBack = np.zeros((size,size,size))
    imageSpeed = np.zeros((size,size,size))

    maxX = 0
    minX = 0
    maxY = 0
    minY = 0
    maxZ = 0
    minZ = 0

    for line in positionData:
        if line[0] > maxX:
            maxX = line[0]
        if line[0] < minX:
            minX = line[0]
        if line[1] > maxY:
            maxY = line[1]
        if line[1] < minY:
            minY = line[1]
        if line[2] > maxZ:
            maxZ = line[2]
        if line[2] < minZ:
            minZ = line[2]

    width = maxX-minX
    height = maxY-minY
    depth = maxZ-minZ

    largestDim = width
    if height > largestDim:
        largestDim = height
    if depth > largestDim:
        largestDim = depth

    if largestDim == 0:
        largestDim = 1

    scale = (size-1)/largestDim
    offsetX = int((size - width*scale)/2.0)
    offsetY = int((size - height*scale)/2.0)
    offsetZ = int((size - depth*scale)/2.0)

    prevPos = [0,0,0]
    prevScaledPos = [0,0,0]
    maxSpeed = getMaxSpeed(positionData, scale, minX, minY, minZ)
    if maxSpeed == 0:
        maxSpeed = 1
    for line in positionData:
        xDif = line[0]-prevPos[0]
        yDif = line[1]-prevPos[1]
        zDif = line[2]-prevPos[2]
        prevPos = line
        totalDif = abs(xDif)+abs(yDif)+abs(zDif)

        upValue = 0
        downValue = 0
        leftValue = 0
        rightValue = 0
        forwardValue = 0
        backValue = 0
        
        unroundedX = (line[0]-minX)*scale
        unroundedY = (line[1]-minY)*scale
        unroundedZ = (line[2]-minZ)*scale
        unroundedXDif = unroundedX-prevScaledPos[0]
        unroundedYDif = unroundedY-prevScaledPos[1]
        unroundedZDif = unroundedZ-prevScaledPos[2]
        speed = unroundedXDif*unroundedXDif+unroundedYDif*unroundedYDif+unroundedZDif*unroundedZDif
        speed = math.sqrt(speed)
        speed = speed/iterPerSecond
        speedValue = speed/maxSpeed
        prevScaledPos = [unroundedX,unroundedY,unroundedZ]
        if speedValue > 1:
            speedValue = 1

        if totalDif > 0:
            if xDif > 0:
                rightValue = xDif/totalDif
            else:
                leftValue = abs(xDif)/totalDif
            if yDif > 0:
                upValue = yDif/totalDif
            else:
                downValue = abs(yDif)/totalDif
            if zDif > 0:
                forwardValue = zDif/totalDif
            else:
                backValue = abs(zDif)/totalDif

        #print("Speed = "+str(speed)+" at up:"+str(upValue)+" down: "+str(downValue)+" left: "+str(leftValue)+" right: "+str(rightValue)+" forward: "+str(forwardValue)+" back: "+str(backValue))

        x = round((line[0]-minX)*scale)
        y = round((line[1]-minY)*scale)
        z = round((line[2]-minZ)*scale)
        brushDist = 0
        for xBrush in range(-brushRadius,brushRadius+1):
            for yBrush in range(-brushRadius,brushRadius+1):
                for zBrush in range(-brushRadius,brushRadius+1):
                    if(x+offsetX+xBrush < size and y+offsetY+yBrush < size and z+offsetZ+zBrush < size):
                        brushDist = abs(xBrush)
                        if abs(yBrush) > brushDist:
                            brushDist = abs(yBrush)
                        if abs(zBrush) > brushDist:
                            brushDist = abs(zBrush)
                        if(imagePosition[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] < 1-(float(brushDist)/brushRadius)):
                            imagePosition[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] = 1-(float(brushDist)/brushRadius)

                        if(imageUp[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] < upValue*(1-(float(brushDist)/brushRadius))):
                            imageUp[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] = upValue*(1-(float(brushDist)/brushRadius))
                        if(imageDown[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] < downValue*(1-(float(brushDist)/brushRadius))):
                            imageDown[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] = downValue*(1-(float(brushDist)/brushRadius))
                        if(imageLeft[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] < leftValue*(1-(float(brushDist)/brushRadius))):
                            imageLeft[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] = leftValue*(1-(float(brushDist)/brushRadius))
                        if(imageRight[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] < rightValue*(1-(float(brushDist)/brushRadius))):
                            imageRight[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] = rightValue*(1-(float(brushDist)/brushRadius))
                        if(imageForward[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] < forwardValue*(1-(float(brushDist)/brushRadius))):
                            imageForward[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] = forwardValue*(1-(float(brushDist)/brushRadius))
                        if(imageBack[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] < backValue*(1-(float(brushDist)/brushRadius))):
                            imageBack[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] = backValue*(1-(float(brushDist)/brushRadius))

                        if(imageSpeed[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] < speedValue*(1-(float(brushDist)/brushRadius))):
                            imageSpeed[x+offsetX+xBrush][y+offsetY+yBrush][z+offsetZ+zBrush] = speedValue*(1-(float(brushDist)/brushRadius))

    # Skip saving a file and convert into the one line format
    oneLineImage = []
    for x in np.nditer(imageUp):
        oneLineImage.append(x)
    for x in np.nditer(imageDown):
        oneLineImage.append(x)
    for x in np.nditer(imageLeft):
        oneLineImage.append(x)
    for x in np.nditer(imageRight):
        oneLineImage.append(x)
    for x in np.nditer(imageForward):
        oneLineImage.append(x)
    for x in np.nditer(imageBack):
        oneLineImage.append(x)
    for x in np.nditer(imageSpeed):
        oneLineImage.append(x)

    return oneLineImage 
def convertFile(inputFile):
    print('Converting file '+inputFile)
    positionData = loadPositionFile(inputFile)
    return Convert(positionData)



def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


def loadFile(fileName):
    print("Loading file " + fileName)
    lines = [line.rstrip('\n') for line in open(fileName)]
    data = []
    for line in lines:
        items = line.split(",")
        for item in items:
            data.append(float(item))
    return data

def loadDirectory(path):
    print("Loading files from " + path)
    files = os.listdir(path)
    trainingFiles = []
    trainingLabels = []
    for file in files:
        trainingFiles.append(convertFile(path+"/"+file))
        #trainingFiles.append(loadFile(path+"/"+file))
        
        label = file.split("_")[0]
        idx = labels.index(label)
        output = []
        for i in range(len(labels)):
            if i == idx:
                output.append(1)
            else:
                output.append(0)
        trainingLabels.append(output)
    return (trainingFiles, trainingLabels)

def convertDirectory(pathFrom, pathTo):
    print("Converting files from " + pathFrom)
    files = os.listdir(pathFrom)
    for fileName in files:
        data = convertFile(pathFrom+"/"+fileName)
        with open(pathTo+"/converted_"+fileName, 'w+') as file:
            for item in data[:-1]:
                file.write(str(item)+",")
            file.write(str(data[-1])+"\n")

def splitDataset(directory, percent):
    files = os.listdir(directory)
    beats = []
    count = int(len(files)/len(labels))
    for i in range(0,len(labels)):
        beats.append(files[i*count:(i+1)*count])
    splitUpTo = count*percent
    trainBeats = []
    testBeats = []
    numUnsorted = count
    while numUnsorted > splitUpTo:
        idx = random.randint(0,numUnsorted-1)
        trainBeats.append(directory + "/" + beats[0][idx])
        del beats[0][idx]
        idx = random.randint(0,numUnsorted-1)
        trainBeats.append(directory + "/" + beats[1][idx])
        del beats[1][idx]
        idx = random.randint(0,numUnsorted-1)
        trainBeats.append(directory + "/" + beats[2][idx])
        del beats[2][idx]
        numUnsorted = numUnsorted - 1
    for beat in beats:
        for file in beat:
            testBeats.append(directory + "/" + file)

    return(trainBeats,testBeats)

def splitDatasetFolds(directory):
    files = os.listdir(directory)
    beats = []
    count = int(len(files)/len(labels))
    for i in range(0,len(labels)):
        beats.append(files[i*count:(i+1)*count])
    splitUpTo = count/nFolds
    folds = []
    numUnsorted = count
    for i in range(0,nFolds):
        folds.append([[],[],[]])
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
    print("Folds:")
    for i in range(0,nFolds):
        print("Fold #"+str(i)+":")
        for f in compiledFolds[i]:
            print(f)
    
    return compiledFolds

def loadDataset(dataset):
    trainingFiles = []
    trainingLabels = []
    for file in dataset:
        trainingFiles.append(convertFile(file))
        
        label = file.split("/")[1].split("_")[0]
        idx = labels.index(label)
        output = []
        for i in range(len(labels)):
            if i == idx:
                output.append(1)
            else:
                output.append(0)
        trainingLabels.append(output)
    return (trainingFiles, trainingLabels)

def shuffle(datalist):
    listL = [[],[]]
    listA = datalist[0]
    listB = datalist[1]
    c = list(zip(listA,listB))
    random.shuffle(c)
    lisA, listB = zip(*c)
    listL[0] = listA
    listL[1] = listB
    return listL

def test():
    
    nInputNeurons = size*size*size*7
    nOutputNeurons = len(labels)

    # Basic Version
    #trainingFiles = loadDirectory("train")    
    #testingFiles = loadDirectory("test")

    # Random Train and Test Split Version
    #datasets = splitDataset("allData",0.4)
    #trainingFiles = loadDataset(datasets[0])
    #testingFiles = loadDataset(datasets[1])

    # Cross-Validation Version
    datasets = splitDatasetFolds("allData")
    results = []

    for k in range(0,nFolds):

        print("Loading Testing Files... (Fold #" + str(k)+")")
        testingFiles = loadDataset(datasets[k])
        trainingFiles = [[],[]]
        print("Loading Training Files...")
        for j in range(0,k):
            print("Loading Fold #"+str(j))
            filesAndLabels = loadDataset(datasets[j])
            trainingFiles[0].extend(filesAndLabels[0])
            print("# of Training Files = "+str(len(trainingFiles[0])))
            trainingFiles[1].extend(filesAndLabels[1])
        for j in range(k+1,nFolds):
            print("Loading Fold #"+str(j))
            filesAndLabels = loadDataset(datasets[j])
            trainingFiles[0].extend(filesAndLabels[0])
            print("# of Training Files = "+str(len(trainingFiles[0])))
            trainingFiles[1].extend(filesAndLabels[1])

        #trainingFiles = shuffle(trainingFiles)
        #testingFiles = shuffle(testingFiles)

        print("# of Training Files = "+str(len(trainingFiles[0])))
        print("# of Testing Files = "+str(len(testingFiles[0])))

        test_X = np.array(testingFiles[0])
        test_y = np.array(testingFiles[1])

        train_X = np.array(trainingFiles[0])
        train_y = np.array(trainingFiles[1])
        
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
        print("Initial Results:")
        print(sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
        print("Should have been:")
        print(train_y)
        test_accuracy = 0
        for epoch in range(nEpochs):
            # Train with each example
            for i in range(len(train_X)):
                sess.run(updates, feed_dict={inputs: train_X[i: i + 1], outputs: train_y[i: i + 1]})

            train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                     sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
            test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                     sess.run(predict, feed_dict={inputs: test_X, outputs: test_y}))

            print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

        print("Training Results:")
        print(sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
        print("Should have been:")
        print(train_y)
        print("Testing Results:")
        print(sess.run(predict, feed_dict={inputs: test_X, outputs: test_y}))
        print("Should have been:")
        print(test_y)
        sess.close()

        results.append(test_accuracy)

    print("Results:")
    total = 0
    for r in results:
        print(str(r*100)+"%")
        total = total + r
    print("Avr = " + str(total/nFolds*100) + "%")

def CreateModel():
    nInputNeurons = size*size*size*7
    nOutputNeurons = len(labels)
    trainingFiles = loadDirectory("allData")    


    print("# of Training Files = "+str(len(trainingFiles[0])))

    train_X = np.array(trainingFiles[0])
    train_y = np.array(trainingFiles[1])
        
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
    print("Initial Results:")
    print(sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
    print("Should have been:")
    print(train_y)
    for epoch in range(nEpochs):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={inputs: train_X[i: i + 1], outputs: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                    sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
            
        print("Epoch = %d, train accuracy = %.2f%%"
                % (epoch + 1, 100. * train_accuracy))

    print("Training Results:")
    print(sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
    print("Should have been:")
    print(train_y)

    save_path = saver.save(sess, "/models/model.ckpt")
    print("Model saved in path: %s" % save_path)
    sess.close()
    
def BluetoothMethod():
    server_socket=BluetoothSocket( RFCOMM )
    server_socket.bind(("", 3 ))
    print("Waiting for a connection...")
    server_socket.listen(1)

    client_socket, address = server_socket.accept()
    print("Connected to " + address)

    while runProgram:
        data = client_socket.recv(1024)
        print("received [%s]" % data)
        data_split = data.split(",")
        listOfPositions.append([float(data_split[0]),float(data_split[1]),float(data_split[2])])
        if len(listOfPositions) > windowSize:
            del listOfPositions[0]

    client_socket.close()
    server_socket.close()


def RunProgram():

    # Load Model
    tf.reset_default_graph()
    nInputNeurons = size*size*size*7
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
    saver.restore(sess, "/models/model.ckpt")


    # Run program
    
    for i in range(windowSize):
        listOfPositions.append([0,0,0])
    thread.start_new_thread(BluetoothMethod, ())
    while runProgram:
        posData = listOfPositions.copy()
        data = []
        data.append(Convert(posData))
        out = sess.run(predict, feed_dict={inputs: data, outputs: [[1,0,0]]})
        print(out)
    
    sess.close()

def Main():
    command = input("Enter command: ")
    if command == "create":
        CreateModel()
    elif command == "run":
        RunProgram()
    elif command == "test":
        test()

Main()
