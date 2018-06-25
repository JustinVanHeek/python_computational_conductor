import numpy as np
import tensorflow as tf
import os
import math
from sklearn.model_selection import train_test_split

# Main Variables
size            =   50      # Length of each dimension of the 3D image, larger = more detailed gestures
brushRadius     =   4       # Radius that values are applied to the 3D image from source point, larger = more generalized
nHiddenNeurons  =   438     # Number of hidden neurons in the neural network, larger = usually better at recognizing more details
nEpochs         =   20      # Number of training epochs for the neural network, larger = usually better accuracy
labels          =   ["beat2","beat3","beat4"]        # Labels of the gestures to recognize (Note: training files should have the naming convention of [labelname]_##.csv
maxSpeed        =   1000    # Maximum speed for normalization of the speed value
iterPerSecond   =   0.025   # Speed at which the data is being recorded

def loadPositionFile(fileName):
    print("Loading file " + fileName)
    lines = [line.rstrip('\n') for line in open(fileName)]
    data = []
    for line in lines:
        items = line.split(",")
        pos = []
        for item in items:
            pos.append(float(item))
        data.append(pos)
    return data

def convertFile(inputFile):
    positionData = loadPositionFile(inputFile)
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

    scale = (size-1)/largestDim
    offsetX = int((size - width*scale)/2.0)
    offsetY = int((size - height*scale)/2.0)
    offsetZ = int((size - depth*scale)/2.0)

    prevPos = [0,0,0]
    prevScaledPos = [0,0,0]
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
        unroundedY = (line[1]-minX)*scale
        unroundedZ = (line[2]-minX)*scale
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
        

def main():
    
    nInputNeurons = size*size*size*7
    nOutputNeurons = len(labels)
    
    trainingFiles = loadDirectory("train")    
    testingFiles = loadDirectory("test")

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
    
    for epoch in range(nEpochs):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={inputs: train_X[i: i + 1], outputs: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
        test_accuracy = 0
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={inputs: test_X, outputs: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    print(sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
    print(sess.run(predict, feed_dict={inputs: test_X, outputs: test_y}))

    sess.close()



main()
