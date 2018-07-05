import plotly.tools as tool
import plotly.graph_objs as go
import plotly.plotly as py

import numpy as np
tool.set_credentials_file(username='3ddata', api_key='MKkafWD9f4yeo19aA9pj')

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

pos = loadPositionFile("plot.csv")
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
py.plot(fig, filename='simple-3d-scatter')
