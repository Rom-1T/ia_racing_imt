import plotly
import plotly.graph_objs as go
import ipywidgets as widgets
import plotly.offline as py
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import datasets
import pandas as pd

def show_loss(resolution = 50, x_range=[-10, 10], y_range=[-10, 10]):
    iris_X = datasets.load_iris()['data']
    iris_y = datasets.load_iris()['target']
    iris_y[iris_y == 2] = 1

    colors = []
    for i in range(iris_y.shape[0]):
        if (iris_y[i] == 0):
            colors.append('green')
        else:
            colors.append('orange')

    scaled = iris_X[:,:2] - np.array([np.mean(iris_X[:,0]), np.mean(iris_X[:,1])])
    scaled = scaled / np.array([np.std(iris_X[:,0]), np.std(iris_X[:,1])])
    
    
    
    
    
    def plot_3d_function(function, resolution = 50, x_range = [-10, 10], y_range = [-10, 10]):
        n = resolution
        x = np.linspace(x_range[0], x_range[1], n)
        y = np.linspace(y_range[0], y_range[1], n)
        coords = []

        i = 0
        while(i < n):
            for j in range(n):
                coords += [[x[i], y[j], function(x[i],y[j])]]
            for j in range(n):
                coords += [[x[i+1], y[n-j-1], function(x[i+1],y[n-j-1])]]
            i = i+2

        coords = np.array(coords)
        coords_x = coords[:,0]
        coords_y = coords[:,1]
        coords_z = coords[:,2]

        return coords_x, coords_y, coords_z


    def loss_function(w1, w2):
        loss = 0
        p = 0
        for i in range(iris_y.shape[0]):
            if (w1*scaled[:,0][i] + w2*(scaled[:,1][i] - 1) > 0):
                prediction = 0
            else:
                prediction = 1
            loss += np.abs(prediction - iris_y[i])
        return loss/iris_y.shape[0]

    x_loss, y_loss, z_loss = plot_3d_function(loss_function, resolution, x_range, y_range)
    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()

    # Trace
    trace = go.Scatter3d(
        x=x_loss,  # <-- Put your data instead
        y=y_loss,  # <-- Put your data instead
        z=z_loss,
        marker={
            'size': 1,
            'opacity': 0.8,
        }) # <-- Put your data instead

    # Configure the layout.
    # layout = go.Layout(
    #     margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    # )


    plot_figure = go.Figure([trace])

    #plot_figure.update_layout(
    #    title={
    #        'text': "MSE in terms of b and w",
    #        'y':0.9,
    #        'x':0.5,
    #        'xanchor': 'center',
    #        'yanchor': 'top'},
    #    scene = dict(xaxis_title='b',
    #                    yaxis_title='w',
    #                    zaxis_title='MSE value'),
    #                    width=700,
    #                    margin=dict(r=20, b=10, l=10, t=10)
    #)


    plot_figure.show()
