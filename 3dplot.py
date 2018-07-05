import plotly.tools as tool
import plotly.graph_objs as go
import plotly.plotly as py
#from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
#tool.set_credentials_file(username='3ddata', api_key='MKkafWD9f4yeo19aA9pj')
tool.set_credentials_file(username='3ddata2', api_key='BRJtXyACquN0h3Q1gluJ')

graphs = os.listdir("graphs")

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

def graph(path, fileName):
    namePNG = "graphs/"+fileName+".png"
    if not (fileName+".png" in graphs):
        pos = loadPositionFile(path+"/"+fileName)
        arr = np.array(pos)

        x2, z2, y2 = arr.transpose()
        graph = go.Scatter3d(
            x=x2,
            y=y2,
            z=z2,
            mode='markers',
            marker=dict(
                color='rgb(127, 127, 127)',
                size=12,
                symbol='circle',
                line=dict(
                    color='rgb(204, 204, 204)',
                    width=1
                ),
                opacity=0.9
            )
        )
        data = [graph]
        layout = go.Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )
        fig = go.Figure(data=data, layout=layout)


        #py.plot(fig, filename='simple-3d-scatter')
        print("Generating PNG image...")
        py.image.save_as(fig, filename=namePNG)
        print("Image saved!")


    print("Showing "+fileName)
    #image = Image.open(namePNG)
    #image.show()
    

    img = mpimg.imread(namePNG)
    plt.imshow(img)
    #plt.show()

def main():
    path = "allData"
    files = os.listdir(path)
    for file in files:
        graph(path,file)
        f = plt.figure(file)
        f.show()
    input("Press Enter to Close Program: ")

main()
