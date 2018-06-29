import plotly.tools as tool
import plotly.graph_objs as go
import plotly.plotly as py
import math

import numpy as np
tool.set_credentials_file(username='3ddata', api_key='MKkafWD9f4yeo19aA9pj')

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
    #oneLineImage = []
    #for x in np.nditer(imageUp):
    #    oneLineImage.append(x)
    #for x in np.nditer(imageDown):
    #    oneLineImage.append(x)
    #for x in np.nditer(imageLeft):
    #    oneLineImage.append(x)
    #for x in np.nditer(imageRight):
    #    oneLineImage.append(x)
    #for x in np.nditer(imageForward):
    #    oneLineImage.append(x)
    #for x in np.nditer(imageBack):
    #    oneLineImage.append(x)
    #for x in np.nditer(imageSpeed):
    #    oneLineImage.append(x)
    #
    #return oneLineImage
    return (imagePosition,imageUp,imageDown,imageLeft,imageRight,imageForward,imageBack,imageSpeed)

images = convertFile("plot.csv")
arr = images[0]
pointsA = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr[x][y][z] == 1):
                pointsA.append([x,y,z])
pointsB = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr[x][y][z] < 1 and arr[x][y][z] > 0.7):
                pointsB.append([x,y,z])
pointsC = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr[x][y][z] <= 0.7 and arr[x][y][z] > 0.3):
                pointsC.append([x,y,z])
pointsD = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr[x][y][z] <= 0.3 and arr[x][y][z] > 0):
                pointsD.append([x,y,z])

arr1 = images[1]
pointsA1 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr1[x][y][z] == 1):
                pointsA1.append([x,y,z])
pointsB1 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr1[x][y][z] < 1 and arr1[x][y][z] > 0.7):
                pointsB1.append([x,y,z])
pointsC1 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr1[x][y][z] <= 0.7 and arr1[x][y][z] > 0.3):
                pointsC1.append([x,y,z])
pointsD1 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr1[x][y][z] <= 0.3 and arr1[x][y][z] > 0):
                pointsD1.append([x,y,z])

arr2 = images[2]
pointsA2 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr2[x][y][z] == 1):
                pointsA2.append([x,y,z])
pointsB2 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr2[x][y][z] < 1 and arr2[x][y][z] > 0.7):
                pointsB2.append([x,y,z])
pointsC2 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr2[x][y][z] <= 0.7 and arr2[x][y][z] > 0.3):
                pointsC2.append([x,y,z])
pointsD2 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr2[x][y][z] <= 0.3 and arr2[x][y][z] > 0):
                pointsD2.append([x,y,z])

arr3 = images[3]
pointsA3 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr3[x][y][z] == 1):
                pointsA3.append([x,y,z])
pointsB3 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr3[x][y][z] < 1 and arr3[x][y][z] > 0.7):
                pointsB3.append([x,y,z])
pointsC3 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr3[x][y][z] <= 0.7 and arr3[x][y][z] > 0.3):
                pointsC3.append([x,y,z])
pointsD3 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr3[x][y][z] <= 0.3 and arr3[x][y][z] > 0):
                pointsD3.append([x,y,z])
arr4 = images[4]
pointsA4 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr4[x][y][z] == 1):
                pointsA4.append([x,y,z])
pointsB4 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr4[x][y][z] < 1 and arr4[x][y][z] > 0.7):
                pointsB4.append([x,y,z])
pointsC4 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr4[x][y][z] <= 0.7 and arr4[x][y][z] > 0.3):
                pointsC4.append([x,y,z])
pointsD4 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr4[x][y][z] <= 0.3 and arr4[x][y][z] > 0):
                pointsD4.append([x,y,z])

arr5 = images[5]
pointsA5 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr5[x][y][z] == 1):
                pointsA5.append([x,y,z])
pointsB5 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr5[x][y][z] < 1 and arr5[x][y][z] > 0.7):
                pointsB5.append([x,y,z])
pointsC5 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr5[x][y][z] <= 0.7 and arr5[x][y][z] > 0.3):
                pointsC5.append([x,y,z])
pointsD5 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr5[x][y][z] <= 0.3 and arr5[x][y][z] > 0):
                pointsD5.append([x,y,z])

arr6 = images[6]
pointsA6 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr6[x][y][z] == 1):
                pointsA6.append([x,y,z])
pointsB6 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr6[x][y][z] < 1 and arr6[x][y][z] > 0.7):
                pointsB6.append([x,y,z])
pointsC6 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr6[x][y][z] <= 0.7 and arr6[x][y][z] > 0.3):
                pointsC6.append([x,y,z])
pointsD6 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr6[x][y][z] <= 0.3 and arr6[x][y][z] > 0):
                pointsD6.append([x,y,z])

arr7 = images[7]
pointsA7 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr7[x][y][z] == 1):
                pointsA7.append([x,y,z])
pointsB7 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr7[x][y][z] < 1 and arr7[x][y][z] > 0.7):
                pointsB7.append([x,y,z])
pointsC7 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr7[x][y][z] <= 0.7 and arr7[x][y][z] > 0.3):
                pointsC7.append([x,y,z])
pointsD7 = []
for x in range(0,size):
    for y in range(0,size):
        for z in range(0,size):
            if(arr7[x][y][z] <= 0.3 and arr7[x][y][z] > 0):
                pointsD7.append([x,y,z])




if len(pointsA) > 0:                
    xA, zA, yA = np.array(pointsA).transpose()
    graphA = go.Scatter3d(
        x=xA,
        y=yA,
        z=zA,
        mode='markers',
        marker=dict(
            color='rgb(127, 127, 127)',
            size=12,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.3
        )
    )
if len(pointsB) > 0:                
    xB, zB, yB = np.array(pointsB).transpose()
    graphB = go.Scatter3d(
        x=xB,
        y=yB,
        z=zB,
        mode='markers',
        marker=dict(
            color='rgb(127, 127, 127)',
            size=9,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.2
        )
    )
if len(pointsC) > 0:                
    xC, zC, yC = np.array(pointsC).transpose()
    graphC = go.Scatter3d(
        x=xC,
        y=yC,
        z=zC,
        mode='markers',
        marker=dict(
            color='rgb(127, 127, 127)',
            size=6,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.15
        )
    )
if len(pointsD) > 0:                
    xD, zD, yD = np.array(pointsD).transpose()
    graphD = go.Scatter3d(
        x=xD,
        y=yD,
        z=zD,
        mode='markers',
        marker=dict(
            color='rgb(127, 127, 127)',
            size=3,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.1
        )
    )

if len(pointsA1) > 0:
    xA1, zA1, yA1 = np.array(pointsA1).transpose()
    graphA1 = go.Scatter3d(
        x=xA1,
        y=yA1,
        z=zA1,
        mode='markers',
        marker=dict(
            color='rgb(255, 0, 0)',
            size=12,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.3
        )
    )
if len(pointsB1) > 0:                
    xB1, zB1, yB1 = np.array(pointsB1).transpose()
    graphB1 = go.Scatter3d(
        x=xB1,
        y=yB1,
        z=zB1,
        mode='markers',
        marker=dict(
            color='rgb(255, 0, 0)',
            size=9,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.2
        )
    )
if len(pointsC1) > 0:                
    xC1, zC1, yC1 = np.array(pointsC1).transpose()
    graphC1 = go.Scatter3d(
        x=xC1,
        y=yC1,
        z=zC1,
        mode='markers',
        marker=dict(
            color='rgb(255, 0, 0)',
            size=6,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.15
        )
    )
if len(pointsD1) > 0:                
    xD1, zD1, yD1 = np.array(pointsD1).transpose()
    graphD1 = go.Scatter3d(
        x=xD1,
        y=yD1,
        z=zD1,
        mode='markers',
        marker=dict(
            color='rgb(255, 0, 0)',
            size=3,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.1
        )
    )
if len(pointsA2) > 0:
    xA2, zA2, yA2 = np.array(pointsA2).transpose()
    graphA2 = go.Scatter3d(
        x=xA2,
        y=yA2,
        z=zA2,
        mode='markers',
        marker=dict(
            color='rgb(127, 0, 0)',
            size=12,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.3
        )
    )
if len(pointsB2) > 0:                
    xB2, zB2, yB2 = np.array(pointsB2).transpose()
    graphB2 = go.Scatter3d(
        x=xB2,
        y=yB2,
        z=zB2,
        mode='markers',
        marker=dict(
            color='rgb(127, 0, 0)',
            size=9,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.2
        )
    )
if len(pointsC2) > 0:                
    xC2, zC2, yC2 = np.array(pointsC2).transpose()
    graphC2 = go.Scatter3d(
        x=xC2,
        y=yC2,
        z=zC2,
        mode='markers',
        marker=dict(
            color='rgb(127, 0, 0)',
            size=6,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.15
        )
    )
if len(pointsD2) > 0:                
    xD2, zD2, yD2 = np.array(pointsD2).transpose()
    graphD2 = go.Scatter3d(
        x=xD2,
        y=yD2,
        z=zD2,
        mode='markers',
        marker=dict(
            color='rgb(127, 0, 0)',
            size=3,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.1
        )
    )
if len(pointsA3) > 0:
    xA3, zA3, yA3 = np.array(pointsA3).transpose()
    graphA3 = go.Scatter3d(
        x=xA3,
        y=yA3,
        z=zA3,
        mode='markers',
        marker=dict(
            color='rgb(0, 255, 0)',
            size=12,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.3
        )
    )
if len(pointsB3) > 0:                
    xB3, zB3, yB3 = np.array(pointsB3).transpose()
    graphB3 = go.Scatter3d(
        x=xB3,
        y=yB3,
        z=zB3,
        mode='markers',
        marker=dict(
            color='rgb(0, 255, 0)',
            size=9,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.2
        )
    )
if len(pointsC3) > 0:                
    xC3, zC3, yC3 = np.array(pointsC3).transpose()
    graphC3 = go.Scatter3d(
        x=xC3,
        y=yC3,
        z=zC3,
        mode='markers',
        marker=dict(
            color='rgb(0, 255, 0)',
            size=6,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.15
        )
    )
if len(pointsD3) > 0:                
    xD3, zD3, yD3 = np.array(pointsD3).transpose()
    graphD3 = go.Scatter3d(
        x=xD3,
        y=yD3,
        z=zD3,
        mode='markers',
        marker=dict(
            color='rgb(0, 255, 0)',
            size=3,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.1
        )
    )
if len(pointsA4) > 0:
    xA4, zA4, yA4 = np.array(pointsA4).transpose()
    graphA4 = go.Scatter3d(
        x=xA4,
        y=yA4,
        z=zA4,
        mode='markers',
        marker=dict(
            color='rgb(0, 127, 0)',
            size=12,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.3
        )
    )
if len(pointsB4) > 0:                
    xB4, zB4, yB4 = np.array(pointsB4).transpose()
    graphB4 = go.Scatter3d(
        x=xB4,
        y=yB4,
        z=zB4,
        mode='markers',
        marker=dict(
            color='rgb(0, 127, 0)',
            size=9,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.2
        )
    )
if len(pointsC4) > 0:                
    xC4, zC4, yC4 = np.array(pointsC4).transpose()
    graphC4 = go.Scatter3d(
        x=xC4,
        y=yC4,
        z=zC4,
        mode='markers',
        marker=dict(
            color='rgb(0, 127, 0)',
            size=6,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.15
        )
    )
if len(pointsD4) > 0:                
    xD4, zD4, yD4 = np.array(pointsD4).transpose()
    graphD4 = go.Scatter3d(
        x=xD4,
        y=yD4,
        z=zD4,
        mode='markers',
        marker=dict(
            color='rgb(0, 127, 0)',
            size=3,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.1
        )
    )
if len(pointsA5) > 0:
    xA5, zA5, yA5 = np.array(pointsA5).transpose()
    graphA5 = go.Scatter3d(
        x=xA5,
        y=yA5,
        z=zA5,
        mode='markers',
        marker=dict(
            color='rgb(0, 0, 255)',
            size=12,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.3
        )
    )
if len(pointsB5) > 0:
    xB5, zB5, yB5 = np.array(pointsB5).transpose()
    graphB5 = go.Scatter3d(
        x=xB5,
        y=yB5,
        z=zB5,
        mode='markers',
        marker=dict(
            color='rgb(0, 0, 255)',
            size=9,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.2
        )
    )
if len(pointsC5) > 0:                
    xC5, zC5, yC5 = np.array(pointsC5).transpose()
    graphC5 = go.Scatter3d(
        x=xC5,
        y=yC5,
        z=zC5,
        mode='markers',
        marker=dict(
            color='rgb(0, 0, 255)',
            size=6,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.15
        )
    )
if len(pointsD5) > 0:                
    xD5, zD5, yD5 = np.array(pointsD5).transpose()
    graphD5 = go.Scatter3d(
        x=xD5,
        y=yD5,
        z=zD5,
        mode='markers',
        marker=dict(
            color='rgb(0, 0, 255)',
            size=3,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.1
        )
    )
if len(pointsA6) > 0:
    xA6, zA6, yA6 = np.array(pointsA6).transpose()
    graphA6 = go.Scatter3d(
        x=xA6,
        y=yA6,
        z=zA6,
        mode='markers',
        marker=dict(
            color='rgb(0, 0, 127)',
            size=12,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.3
        )
    )
if len(pointsB6) > 0:
    xB6, zB6, yB6 = np.array(pointsB6).transpose()
    graphB6 = go.Scatter3d(
        x=xB6,
        y=yB6,
        z=zB6,
        mode='markers',
        marker=dict(
            color='rgb(0, 0, 127)',
            size=9,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.2
        )
    )
if len(pointsC6) > 0:
    xC6, zC6, yC6 = np.array(pointsC6).transpose()
    graphC6 = go.Scatter3d(
        x=xC6,
        y=yC6,
        z=zC6,
        mode='markers',
        marker=dict(
            color='rgb(0, 0, 127)',
            size=6,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.15
        )
    )
if len(pointsD6) > 0:
    xD6, zD6, yD6 = np.array(pointsD6).transpose()
    graphD6 = go.Scatter3d(
        x=xD6,
        y=yD6,
        z=zD6,
        mode='markers',
        marker=dict(
            color='rgb(0, 0, 127)',
            size=3,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.1
        )
    )
if len(pointsA7) > 0:
    xA7, zA7, yA7 = np.array(pointsA7).transpose()
    graphA7 = go.Scatter3d(
        x=xA7,
        y=yA7,
        z=zA7,
        mode='markers',
        marker=dict(
            color='rgb(127, 0, 127)',
            size=12,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.3
        )
    )
if len(pointsB7) > 0:
    xB7, zB7, yB7 = np.array(pointsB7).transpose()
    graphB7 = go.Scatter3d(
        x=xB7,
        y=yB7,
        z=zB7,
        mode='markers',
        marker=dict(
            color='rgb(127, 0, 127)',
            size=9,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.2
        )
    )
if len(pointsC7) > 0:
    xC7, zC7, yC7 = np.array(pointsC7).transpose()
    graphC7 = go.Scatter3d(
        x=xC7,
        y=yC7,
        z=zC7,
        mode='markers',
        marker=dict(
            color='rgb(127, 0, 127)',
            size=6,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.15
        )
    )
if len(pointsD7) > 0:
    xD7, zD7, yD7 = np.array(pointsD7).transpose()
    graphD7 = go.Scatter3d(
        x=xD7,
        y=yD7,
        z=zD7,
        mode='markers',
        marker=dict(
            color='rgb(127, 0, 127)',
            size=3,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.1
        )
    )

data = []

try:
    graphA
except NameError:
    print("No A")
else:
    print("A")
    #data.append(graphA)

try:
    graphB
except NameError:
    print("No B")
else:
    print("B")
    #data.append(graphB)

try:
    graphC
except NameError:
    print("No C")
else:
    print("C")
    #data.append(graphC)

try:
    graphD
except NameError:
    print("No D")
else:
    print("D")
    #data.append(graphD)

try:
    graphA1
except NameError:
    print("No A1")
else:
    data.append(graphA1)

try:
    graphB1
except NameError:
    print("No B1")
else:
    data.append(graphB1)

try:
    graphC1
except NameError:
    print("No C1")
else:
    data.append(graphC1)

try:
    graphD1
except NameError:
    print("No D1")
else:
    data.append(graphD1)

try:
    graphA2
except NameError:
    print("No A2")
else:
    data.append(graphA2)

try:
    graphB2
except NameError:
    print("No B2")
else:
    data.append(graphB2)

try:
    graphC2
except NameError:
    print("No C2")
else:
    data.append(graphC2)

try:
    graphD2
except NameError:
    print("No D2")
else:
    data.append(graphD2)

try:
    graphA3
except NameError:
    print("No A3")
else:
    data.append(graphA3)

try:
    graphB3
except NameError:
    print("No B3")
else:
    data.append(graphB3)

try:
    graphC3
except NameError:
    print("No C3")
else:
    data.append(graphC3)

try:
    graphD
except NameError:
    print("No D3")
else:
    data.append(graphD3)

try:
    graphA4
except NameError:
    print("No A4")
else:
    data.append(graphA4)

try:
    graphB4
except NameError:
    print("No B4")
else:
    data.append(graphB4)

try:
    graphC4
except NameError:
    print("No C4")
else:
    data.append(graphC4)

try:
    graphD4
except NameError:
    print("No D4")
else:
    data.append(graphD4)

try:
    graphA5
except NameError:
    print("No A5")
else:
    data.append(graphA5)

try:
    graphB5
except NameError:
    print("No B5")
else:
    data.append(graphB5)

try:
    graphC5
except NameError:
    print("No C5")
else:
    data.append(graphC5)

try:
    graphD5
except NameError:
    print("No D5")
else:
    data.append(graphD5)

try:
    graphA6
except NameError:
    print("No A6")
else:
    data.append(graphA6)

try:
    graphB6
except NameError:
    print("No B6")
else:
    data.append(graphB6)

try:
    graphC6
except NameError:
    print("No C6")
else:
    data.append(graphC6)

try:
    graphD6
except NameError:
    print("No D6")
else:
    data.append(graphD6)

try:
    graphA7
except NameError:
    print("No A7")
else:
    print("A7")
    #data.append(graphA7)

try:
    graphB7
except NameError:
    print("No B7")
else:
    print("B7")
    #data.append(graphB7)

try:
    graphC7
except NameError:
    print("No C7")
else:
    print("C7")
    #data.append(graphC7)

try:
    graphD7
except NameError:
    print("No D7")
else:
    print("D7")
    #data.append(graphD7)




layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='simple-3d-scatter')
