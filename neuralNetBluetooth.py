import numpy as np
import tensorflow as tf
import os
import math
import random
import _thread as thread
import time
import glob
import matplotlib.pyplot as plt
from bluetooth import *
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 


# Main Settings
size            =   50      # Length of each dimension of the 3D image, larger = more detailed gestures
brushRadius     =   4       # Radius that values are applied to the 3D image from source point, larger = more generalized
nHiddenNeurons  =   400     # Number of hidden neurons in the neural network, larger = usually better at recognizing more details
nEpochs         =   30      # Number of training epochs for the neural network, larger = usually better accuracy
labels          =   ["beat2","beat3","beat4"]        # Labels of the gestures to recognize (Note: training files should have the naming convention of [labelname]_##.csv
iterPerSecond   =   0.025   # Speed at which the data is being recorded
nFolds          =   3       # Number of cross validation folds
windowSize      =   60      # Number of lines of positional data that are kept recorded
detailedEpochs  =   False

# Global Variables

currentBeat = []
listOfPositions = []        # The recorded positional data
runProgram = True           # Flag to run the main bluetooth data recording & prediction loops
nPosLines = 0
testedCurrentBeat = False
loadedFile = []
loaded = False

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#graph, = ax.scatter([],[],[])
ax.plot([],[],[])
def loadPositionFile(fileName):
    #print("Loading file " + fileName)
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
    #print('Converting file '+inputFile)
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

def LoadData(fileName):
    return convertFile(fileName)

def LoadDataThread(fileName):
    global loadedFile, loaded
    loadedFile = convertFile(fileName)
    loaded = True

def LoadResult(fileName):
    label = fileName.split("/")[1].split("_")[0]
    idx = labels.index(label)
    output = []
    for i in range(len(labels)):
        if i == idx:
            output.append(1)
        else:
            output.append(0)
    return output

def LoadIdx(fileName):
    label = fileName.split("/")[1].split("_")[0]
    idx = labels.index(label)
    return idx

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
    print(len(datasets))
    results = []

    for k in range(0,nFolds):

        print("Loading Testing Files... (Fold #" + str(k)+")")
        testingFiles = datasets[k]
        trainingFiles = []
        print("Loading Training Files...")
        for j in range(0,k):
            print("Loading Fold #"+str(j))
            trainingFiles.extend(datasets[j])
            print("# of Training Files = "+str(len(trainingFiles)))
        for j in range(k+1,nFolds):
            print("Loading Fold #"+str(j))
            trainingFiles.extend(datasets[j])
            print("# of Training Files = "+str(len(trainingFiles)))

        #trainingFiles = shuffle(trainingFiles)
        #testingFiles = shuffle(testingFiles)

        print("# of Training Files = "+str(len(trainingFiles)))
        print("# of Testing Files = "+str(len(testingFiles)))

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
            print("Epoch Progress: 0___________________100")
            print("                ", end='')
            l = len(trainingFiles)
            for i in range(l):
                if(i%int(l*0.05)) == 0:
                    print("#", end='')
                sess.run(updates, feed_dict={inputs: [LoadData(trainingFiles[i])], outputs: [LoadResult(trainingFiles[i])]})
                

            print(" Done epoch.")

            if detailedEpochs:
                #train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                #                         sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
                train_accuracy = 0
                for i in range(len(trainingFiles)):
                    r = sess.run(predict, feed_dict={inputs: [LoadData(trainingFiles[i])], outputs: [LoadResult(trainingFiles[i])]})
                    train_results.append(r[0])
                    c = LoadIdx(trainingFiles[i])
                    train_correct.append(c)
                    if r[0] == c:
                        train_accuracy = train_accuracy + 1
                train_accuracy = train_accuracy / len(trainingFiles)

                test_accuracy = 0
                for i in range(len(testingFiles)):
                    r = sess.run(predict, feed_dict={inputs: [LoadData(testingFiles[i])], outputs: [LoadResult(testingFiles[i])]})
                    test_results.append(r[0])
                    c = LoadIdx(testingFiles[i])
                    test_correct.append(c)
                    if r[0] == c:
                        test_accuracy = test_accuracy + 1
                test_accuracy = test_accuracy / len(testingFiles)    
                        
                #test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                #                         sess.run(predict, feed_dict={inputs: test_X, outputs: test_y}))

                print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                      % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

        train_accuracy = 0
        for i in range(len(trainingFiles)):
            r = sess.run(predict, feed_dict={inputs: [LoadData(trainingFiles[i])], outputs: [LoadResult(trainingFiles[i])]})
            train_results.append(r[0])
            c = LoadIdx(trainingFiles[i])
            train_correct.append(c)
            if r[0] == c:
                train_accuracy = train_accuracy + 1
        train_accuracy = train_accuracy / len(trainingFiles)

        test_accuracy = 0
        for i in range(len(testingFiles)):
            r = sess.run(predict, feed_dict={inputs: [LoadData(testingFiles[i])], outputs: [LoadResult(testingFiles[i])]})
            test_results.append(r[0])
            c = LoadIdx(testingFiles[i])
            test_correct.append(c)
            if r[0] == c:
                test_accuracy = test_accuracy + 1
        test_accuracy = test_accuracy / len(testingFiles)   
        print("Training Results: " + str(train_accuracy))
        print(train_results)
        print("Should have been:")
        print(train_correct)
        print("Testing Results: "+ str(test_accuracy))
        print(test_results)
        print("Should have been:")
        print(test_correct)
        sess.close()

        #results.append(test_accuracy)

    print("Results:")
    total = 0
    for r in results:
        print(str(r*100)+"%")
        total = total + r
    print("Avr = " + str(total/nFolds*100) + "%")

def CreateModel():
    global loaded
    nInputNeurons = size*size*size*7
    nOutputNeurons = len(labels)
    files = os.listdir("allData")
    trainingFiles = []
    for f in files:
        trainingFiles.append("allData/"+f)
  


    print("# of Training Files = "+str(len(trainingFiles)))

        
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
    print("Should have been:")
    start = time.time()
    timeRecorded = False
    for epoch in range(nEpochs):
        # Train with each example
        first = True
        d = []
        for i in range(len(trainingFiles)):
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
            print(str(endT-startT) + " seconds to load and train")

        train_accuracy = 0
        for i in range(len(trainingFiles)):
            r = sess.run(predict, feed_dict={inputs: [LoadData(trainingFiles[i])], outputs: [LoadResult(trainingFiles[i])]})
            if r[0] == LoadIdx(trainingFiles[i]):
                train_accuracy = train_accuracy + 1
        train_accuracy = train_accuracy / len(trainingFiles)
        #train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
        #                            sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
            
        print("Epoch = %d, train accuracy = %.2f%%"
                % (epoch + 1, 100. * train_accuracy))
        if not timeRecorded:
            timeRecorded = True
            end = time.time()
        remainingTime = (end - start)*(nEpochs-epoch)/60
        print(str(remainingTime) + " minutes remaining")
        

    #print("Training Results:")
    #print(sess.run(predict, feed_dict={inputs: train_X, outputs: train_y}))
    #print("Should have been:")
    #print(train_y)

    save_path = saver.save(sess, "/models/model.ckpt")
    print("Model saved in path: %s" % save_path)
    sess.close()
    
def BluetoothMethod():
    global listOfPositions, currentBeat, testedCurrentBeat
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

    print("Waiting for connection on RFCOMM channel %d" % port)
    client_sock, client_info = server_sock.accept()
    print("Accepted connection from ", client_info)


    while runProgram:          

        try:
            req = client_sock.recv(1024)
            if len(req) == 0:
                break
            #print("received [%s]" % req)
            messages = req.decode("utf-8")
            #print("RECIEVED: " + messages)
            messages = messages.split("BEAT")
            if len(messages) > 1:
                # Its a new beat so remove all old data
                currentBeat = listOfPositions.copy()
                testedCurrentBeat = False
                listOfPositions = []
                print("New Beat")
            for message in messages:
                if len(message) > 0:
                    message = message.split("END")
                    for m in message:
                        if len(m) > 0:
                            messageSplit = m.split(",")
                            pos = [float(messageSplit[0]),float(messageSplit[1]),float(messageSplit[2])]
                            listOfPositions.append(pos)
                            if len(listOfPositions) > windowSize:
                                del listOfPositions[0]


            data = None
            if req in ('temp', '*temp'):
                data = str(random.random())+'!'
            else:
                pass

            if data:
                print("sending [%s]" % data)
                client_sock.send()

        except IOError:
            pass

        except KeyboardInterrupt:

            print("disconnected")

            client_sock.close()
            server_sock.close()
            print("all done")

            break



def RunProgram():
    global testedCurrentBeat

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
    
    #for i in range(windowSize):
    #    listOfPositions.append([0,0,0])
    thread.start_new_thread(BluetoothMethod, ())
    lastThreePredictions = [0,0,0]
    while runProgram:
        posData = []
        if testedCurrentBeat:
            posData = listOfPositions.copy()
        else:
            posData = currentBeat.copy()
            testedCurrentBeat = True
        SavePosData(posData)
        data = []
        data.append(Convert(posData))
        out = sess.run(predict, feed_dict={inputs: data, outputs: [[1,0,0]]})

        #print("Predicted gesture is " + labels[out[0]])
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
        print("Predicted gesture is " + labels[mostLikely])

    
    sess.close()

fileNum = 0
def SavePosData(posData):
    global fileNum
    with open("posdata/position_data_"+str(fileNum), "w") as savefile:
        for line in posData:
            for item in line:
                savefile.write(str(item)+",")
            savefile.write("\n")
    fileNum = fileNum + 1
    UpdatePlot(posData)

def UpdatePlot(posData):
    global graph
    x = []
    y = []
    z = []
    minX = 0
    minY = 0
    minZ = 0
    maxX = 0
    maxY = 0
    maxZ = 0
    for line in posData:
        x.append(line[0])
        y.append(line[1])
        z.append(line[2])
        if line[0] < minX:
            minX = line[0]
        if line[0] > maxX:
            maxX = line[0]
        if line[1] < minY:
            minY = line[1]
        if line[1] > maxY:
            maxY = line[1]
        if line[2] < minZ:
            minZ = line[2]
        if line[2] > maxZ:
            maxZ = line[2]
    #graph.set_xdata(x)
    #graph.set_ydata(y)
    #graph.set_zdata(z)

    plt.cla()
    ax.set_xlim([minX, maxX])
    ax.set_ylim([minZ, maxZ])
    ax.set_zlim([minY, maxY])
    ax.plot(x,z,y)

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

    #plt.draw()
    #plt.pause(0.5)

def TestModel():

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
    files = os.listdir("test")
    for file in files:
        data = Convert(loadPositionFile("test/"+file))
        out = sess.run(predict, feed_dict={inputs: [data], outputs: [[1,0,0]]})

        print("Predicted gesture is " + labels[out[0]])
    
    sess.close()


def Main():
    command = input("Enter command: ")
    if command == "create":
        CreateModel()
    elif command == "run":
        RunProgram()
    elif command == "testload":
        TestModel()
    elif command == "test":
        test()
    elif command == "bluetooth":
        thread.start_new_thread(BluetoothMethod, ())
        while runProgram:
            SavePosData()

Main()
