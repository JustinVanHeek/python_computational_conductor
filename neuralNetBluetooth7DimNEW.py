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

# Settings
windowSize =    100     # The max size of listOfData
scaleAccel =    1/20     # Scale factor on acceleration data
scaleRot =      1/8   # Scale factor on rotation data
scaleRotV =     1/180

size            =   50      # Length of each dimension of the 3D image, larger = more detailed gestures
brushRadius     =   4       # Radius that values are applied to the 3D image from source point, larger = more generalized
nHiddenNeurons  =   100     # Number of hidden neurons in the neural network, larger = usually better at recognizing more details
nEpochs         =   25      # Number of training epochs for the neural network, larger = usually better accuracy
labels          =   ["2beatNormal","2beatStaccato","2beatLegato","3beatNormal","3beatStaccato","3beatLegato","4beatNormal","4beatStaccato",
                     "4beatLegato"]        # Labels of the gestures to recognize (Note: training files should have the naming convention of [labelname]_##.csv
nFolds          =   3       # Number of cross validation folds
detailedEpochs  =   True


# Global Variables
runProgram = True
beatName = "beat#"
userName = "user"
accelX = [0]*windowSize             # The recorded data
accelY = [0]*windowSize
accelZ = [0]*windowSize
rotX = [0]*windowSize
rotY = [0]*windowSize
rotZ = [0]*windowSize
rotVX = [0]*windowSize
rotVY = [0]*windowSize
rotVZ = [0]*windowSize
buffer = ""             # Contains the excess of a message if it does not end in ",END"
fig = None
ax = None
lineAccelX = None
lineAccelY = None
lineAccelZ = None
lineRotX = None
lineRotY = None
lineRotZ = None
lineRotVX = None
lineRotVY = None
lineRotVZ = None
axisX = np.linspace(0, windowSize, windowSize, endpoint=False)
beatStart = [-10]*windowSize
predBeatStart = [-10]*windowSize
lineBeatStart = None
linePredBeatStart = None
recordData = False
beatIdx = 0
newBeat = False
currentBeat = None
testData = False

def RunProgram():
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
            out = sess.run(predict, feed_dict={inputs: [data], outputs: [[1,0,0]]})
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


def CreateModel():
    global loaded
    nInputNeurons = size*size*6
    nOutputNeurons = len(labels)
    files = os.listdir("trainingDataSensor")
    files.sort()
    trainingFiles = []
    for f in files:
        trainingFiles.append("trainingDataSensor/"+f)
  


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

    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in path: %s" % save_path)
    sess.close()

def splitDatasetFolds(directory):
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
    print("Folds:")
    for i in range(0,nFolds):
        print("Fold #"+str(i)+":")
        for f in compiledFolds[i]:
            print(f)
    
    return compiledFolds

def LoadData(fileName):
    f = LoadFile(fileName)
    return Generate2DImage(f)

def LoadIdx(fileName):
    label = fileName.split("/")[1].split("_")[0]
    idx = labels.index(label)
    return idx

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

def test():
    
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
    datasets = splitDatasetFolds("trainingDataSensor")
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

def LoadFile(fileName):
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

def Generate2DImage(data):
    image = []
    image.extend(Generate2DImageFeature(data, 0, scaleAccel))
    image.extend(Generate2DImageFeature(data, 1, scaleAccel))
    image.extend(Generate2DImageFeature(data, 2, scaleAccel))
    image.extend(Generate2DImageFeature(data, 3, scaleRot))
    image.extend(Generate2DImageFeature(data, 4, scaleRot))
    image.extend(Generate2DImageFeature(data, 5, scaleRot))
    return image

def Generate2DImageFeature(data, col, scale):
    image = np.zeros((size,size))

    x = 0
    xIncrease = int(50/len(data))

    for line in data:
        y = line[col]*scale
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
                if(x+xBrush < size and y+yBrush < size):
                    brushDist = abs(xBrush)
                    if abs(yBrush) > brushDist:
                        brushDist = abs(yBrush)
                    if(x+xBrush) > 0 and (y+yBrush) > 0:
                        if(image[x+xBrush][y+yBrush] < (1-(float(brushDist)/brushRadius))):
                            image[x+xBrush][y+yBrush] = (1-(float(brushDist)/brushRadius))
                
        x = x + xIncrease
        
    oneLineImage = []
    for i in np.nditer(image):
        oneLineImage.append(i)
    return oneLineImage
        

def GetBeatIdx(start,end):
    lowestValue = rotX[start]
    lowestIdx = start
    for i in range (start,end+1):
        if rotX[i] < lowestValue:
            lowestValue = rotX[i]
            lowestIdx = i
    return lowestIdx

def GetData(start,end):
    data = []
    for i in range(start,end+1):
        data.append([accelX[i],accelY[i],accelZ[i],rotX[i],rotY[i],rotZ[i]])
    return data

fileNum = 0
def SaveData(data):
    global fileNum
    with open("generatedData/"+beatName+"_sensor_"+userName+"_"+str(fileNum), "w") as savefile:
        for line in data:
            for item in line:
                savefile.write(str(item)+",")
            savefile.write("\n")
    fileNum = fileNum + 1


def GetMinMax(listOfNumbers):
    minValue = listOfNumbers[0]
    maxValue = listOfNumbers[0]
    for num in listOfNumbers:
        if num < minValue:
            minValue = num
        if num > maxValue:
            maxValue = num
    return [minValue,maxValue]


def GetAverage(listOfNumbers):
    total = 0
    for num in listOfNumbers:
        total = total + num
    return total/len(listOfNumbers)


rotXInRange = False
rotXEntered = 0
rotXExited = 0
def IsNewBeat():
    global rotXInRange, rotXEntered, rotXExited
    minRotX, maxRotX = GetMinMax(rotX)
    half = (maxRotX-minRotX)/2
    level = minRotX + half/2
    if rotX[-1] < level and not rotXInRange:
        rotXInRange = True
        rotXEntered = windowSize - 1
    elif rotX[-1] > level and rotXInRange:
        rotXInRange = False
        rotXExited = windowSize - 1
        return True
    return False
    

def PlotData():
    global fig, ax, lineAccelX, lineAccelY, lineAccelZ, lineRotX, lineRotY, lineRotZ, lineBeatStart, linePredBeatStart, lineRotVX, lineRotVY, lineRotVZ
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

def InitializeGraph():
    global fig, ax, lineAccelX, lineAccelY, lineAccelZ, lineRotX, lineRotY, lineRotZ, lineBeatStart, linePredBeatStart, lineRotVX, lineRotVY, lineRotVZ
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lineAccelX, lineAccelY, lineAccelZ, lineRotX, lineRotY, lineRotZ, lineBeatStart, linePredBeatStart, lineRotVX, lineRotVY, lineRotVZ = ax.plot(axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize),axisX,np.zeros(windowSize))
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

def BluetoothMethod():
    global buffer, accelX, accelY, accelZ, rotX, rotY, rotZ, beatStart, rotVX, rotVY, rotVZ, beatIdx, rotXEntered, rotXExited, newBeat, currentBeat
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
                            predBeat = True
                            newIdx = GetBeatIdx(rotXEntered, rotXExited)
                            if recordData:
                                d = GetData(beatIdx, newIdx)
                                SaveData(d)
                            if testData:
                                currentBeat = GetData(beatIdx, newIdx)
                                newBeat = True
                            beatIdx = newIdx
                        if predBeat:
                            predBeatStart.append(10)
                            predBeat = False
                        else:
                            predBeatStart.append(-10)
                        del predBeatStart[0]
                        if beat:
                            beatStart.append(10)
                            beat = False
                        else:
                            beatStart.append(-10)
                        del beatStart[0]
                        beatIdx = beatIdx - 1
                        rotXEntered = rotXEntered - 1
                        rotXExited = rotXExited - 1
                        if beatIdx < 0:
                            beatIdx = 0
                        if rotXEntered < 0:
                            rotXEntered = 0
                        if rotXExited < 0:
                            rotXExited = 0
                    else:
                        print("Error recieving message " + m)
                buffer = message[-1]


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


def Main():
    global recordData, beatName, userName, testData
    command = input("Enter command: ")
    if command == "bluetooth":
        thread.start_new_thread(BluetoothMethod, ())
        InitializeGraph()
        while runProgram:
            PlotData()
    elif command == "generate":
        beatName = input("Enter Beat Label: ")
        userName = input("Enter Your Name: ")
        recordData = True
        thread.start_new_thread(BluetoothMethod, ())
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

Main()
