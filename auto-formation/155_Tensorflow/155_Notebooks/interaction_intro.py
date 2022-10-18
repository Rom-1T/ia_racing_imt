import numpy as np
import ipywidgets as widgets
import bqplot as bq
import bqplot.pyplot as plt
from IPython.display import display
from ipywidgets import HTML

def show_optimization(f2, f2_p):
    
    xs = np.linspace(-40, 40, 100)
    ys = f2(xs)

    grad3_x_sc = plt.LinearScale(min = -16, max = 15)
    grad3_y_sc = plt.LinearScale(min = -20, max = 25)

    grad3_lines = plt.Lines(x=xs, y=ys, colors=['red', 'green'],scales = {'x': grad3_x_sc, 'y' : grad3_y_sc})

    grad3_label_min_global = plt.label(["Minimum global"],
                        x = [-10.7],
                        y = [f2(-10.7)],
                        x_offset = -30,
                        y_offset = 15,
                        scales = {'x': grad3_x_sc, 'y' : grad3_y_sc},
                        default_size = 12,
                        font_weight = 'bolder',
                        update_on_move = True,
                        colors = ["blue"])

    grad3_minimum_global = plt.Scatter(x = [-10.7],
                          y = [f2(-10.7)],
                          colors = ["blue"],
                          scales = {'x': grad3_x_sc, 'y' : grad3_y_sc})

    grad3_label_min_local = plt.label(["Minimum local"],
                        x = [7.02],
                        y = [f2(7.02)],
                        x_offset = -30,
                        y_offset = 15,
                        scales = {'x': grad3_x_sc, 'y' : grad3_y_sc},
                        default_size = 12,
                        font_weight = 'bolder',
                        update_on_move = True,
                        colors = ["red"])

    grad3_minimum_local = plt.Scatter(x = [7.02],
                          y = [f2(7.02)],
                          colors = ["red"],
                          scales = {'x': grad3_x_sc, 'y' : grad3_y_sc})

    xk = np.array([-5])

    grad3_label_f_prim = plt.label([" "],
                        x = xk,
                        y = f2(xk),
                        x_offset = 0,
                        y_offset = 0,
                        default_size = 20,
                        font_weight = 'bolder',
                        update_on_move = True,
                        colors = ["green"])

    grad3_label_x0 = plt.label(["w0"],
                        x = [-5],
                        y = [0],
                        x_offset = 10,
                        y_offset = 0,
                        default_size = 20,
                        font_weight = 'bolder',
                        update_on_move = True,
                        colors = ["red"])

    grad3_lines_x0 = plt.Lines(x = [-5, -5],
                         y = [-5, 25],
                         scales = {'x': grad3_x_sc, 'y' : grad3_y_sc},
                         line_style = "dashed",
                         colors = ["red"])

    grad3_point = plt.Scatter(x = xk,
                        y = f2(xk),
                        colors = ["green"],
                        scales = {'x': grad3_x_sc, 'y' : grad3_y_sc})

    grad3_point_lines = plt.Lines(x = xk,
                            y = f2(xk),
                            colors = ["green"],
                            scales = {'x': grad3_x_sc, 'y' : grad3_y_sc})

    grad3_ax_x = bq.Axis(scale=grad3_x_sc,
                    grid_lines='solid')

    grad3_ax_y = bq.Axis(scale=grad3_y_sc,
                    orientation='vertical',
                    grid_lines='solid')

    grad3_fig2 = plt.Figure(marks = [grad3_lines, grad3_point_lines, grad3_lines_x0,
                                     grad3_label_min_global, grad3_minimum_global, grad3_label_f_prim,
                                     grad3_label_min_local, grad3_minimum_local,
                                     grad3_label_x0, grad3_point],
                            axes = [grad3_ax_x, grad3_ax_y],
                            title = "Gradient Descent optimization")

    grad3_fig2.layout.height = '400px'
    grad3_fig2.layout.width = '600px'

    #display(fig2)

    grad3_x0 = widgets.FloatSlider(min = -15, max = 11, value = -5, step = 1, description = "w0")
    grad3_learning_rate = widgets.BoundedFloatText(min=0.001, max=0.6, step=0.01, value = 0.1, description = "Learning rate")
    grad3_etape_play = widgets.Play(value = 0,interval = 300, min=0, max=50, step=1, disabled=False)
    grad3_etape = widgets.IntSlider(value = 0, min = 0, max = 50, step = 1, description = "Step")
    grad3_hbox = widgets.HBox([grad3_etape_play,grad3_etape])
    widgets.jslink((grad3_etape_play, 'value'), (grad3_etape, 'value'))

    def gradient_plot2(change):

        xk = np.zeros(101)
        xk[0] = grad3_x0.value
        for k in np.arange(100)+1:
             xk[k] = xk[k-1] - grad3_learning_rate.value*f2_p(xk[k-1])
        grad3_point.x = xk[:grad3_etape.value+1]
        grad3_point.y = f2(xk[:grad3_etape.value+1])

        grad3_point_lines.x = xk[:grad3_etape.value+1]
        grad3_point_lines.y = f2(xk[:grad3_etape.value+1])

        grad3_label_x0.x = [grad3_x0.value]
        grad3_lines_x0.x = [grad3_x0.value, grad3_x0.value]
        grad3_lines_x0.y = [0, f2(grad3_x0.value)]

        grad3_label_f_prim.x = [-14]
        grad3_label_f_prim.y = [22]
        grad3_label_f_prim.text = ["Step " + str(grad3_etape.value) + ", f'(w"+str(grad3_etape.value)+") = "+str(np.round(f2_p(xk[grad3_etape.value]), 3))]

    grad3_etape_play.observe(gradient_plot2)
    grad3_x0.observe(gradient_plot2)
    grad3_learning_rate.observe(gradient_plot2)


    v_opti = widgets.VBox([grad3_x0, grad3_learning_rate, grad3_etape, grad3_etape_play])


    v_opti.layout.align_items = 'center'

    v_out = widgets.HBox([grad3_fig2, v_opti])
    v_out.layout.align_items = 'center'
    display(v_out)


def show_linear(X, y):
    def f(x, w, b):
        return w*x + b

    grad3_x_sc = plt.LinearScale(min = -2, max = 2)
    grad3_y_sc = plt.LinearScale(min = -100, max = 100)

    w0, b0 = 44, 0

    points = plt.Scatter(x = X,
                          y = y,
                          default_size = 20,
                          alpha=0.7,
                          colors = ["blue"],
                          scales = {'x': grad3_x_sc, 'y' : grad3_y_sc})

    lines = plt.Lines(x = [-2, 2],
                             y = [f(-2, w0, b0), f(2, w0, b0)],
                             scales = {'x': grad3_x_sc, 'y' : grad3_y_sc},
                             line_style = "solid",
                             colors = ["red"])

    grad3_ax_x = bq.Axis(scale=grad3_x_sc,
                    grid_lines='solid')

    grad3_ax_y = bq.Axis(scale=grad3_y_sc,
                    orientation='vertical',
                    grid_lines='solid')

    grad3_fig2 = plt.Figure(marks = [points, lines],
                            axes = [grad3_ax_x, grad3_ax_y],
                            title = "Linear model")

    grad3_fig2.layout.height = '400px'
    grad3_fig2.layout.width = '600px'

    widget_w = widgets.FloatSlider(min = -20, max = 100, value = 44, step = 1, description = "w")
    widget_b = widgets.FloatSlider(min = -20, max = 20, value = 0, step = 1, description = "b")


    def plot_lines(change):
        w = widget_w.value
        b = widget_b.value
        lines.y = [f(-2, w, b), f(2, w, b)]


    widget_w.observe(plot_lines)
    widget_b.observe(plot_lines)

    h = HTML(value='<font size="+1">Parameters</font>')
    v_opti = widgets.VBox([h, HTML(value=''), widget_w, widget_b])


    v_opti.layout.align_items = 'center'

    v_out = widgets.HBox([grad3_fig2, v_opti])
    v_out.layout.align_items = 'center'
    display(v_out)
    

def show_polynomial(X, y):
    def f(x, w2, w1, w0, b):
        return w2*x**3 + w1*x**2 + w0*x + b

    w2, w1, w0, b = 6, 8, 37, -8

    xs = np.linspace(-2, 2, 100)
    ys = f(xs, w2, w1, w0, b)

    grad3_x_sc = plt.LinearScale(min = -2, max = 2)
    grad3_y_sc = plt.LinearScale(min = -100, max = 100)



    points = plt.Scatter(x = X,
                          y = y,
                          default_size = 20,
                          alpha=0.7,
                          colors = ["blue"],
                          scales = {'x': grad3_x_sc, 'y' : grad3_y_sc})

    lines = plt.Lines(x = xs,
                        y = f(xs, w2, w1, w0, b),
                        scales = {'x': grad3_x_sc, 'y' : grad3_y_sc},
                        line_style = "solid",
                        colors = ["red"])

    grad3_ax_x = bq.Axis(scale=grad3_x_sc,
                    grid_lines='solid')

    grad3_ax_y = bq.Axis(scale=grad3_y_sc,
                    orientation='vertical',
                    grid_lines='solid')

    grad3_fig2 = plt.Figure(marks = [points, lines],
                            axes = [grad3_ax_x, grad3_ax_y],
                            title = "Polynomial model")

    grad3_fig2.layout.height = '400px'
    grad3_fig2.layout.width = '600px'


    widget_w2 = widgets.FloatSlider(min = -20, max = 100, value = 6, step = 1, description = "w2")
    widget_w1 = widgets.FloatSlider(min = -20, max = 100, value = 8, step = 1, description = "w1")
    widget_w0 = widgets.FloatSlider(min = -20, max = 100, value = 37, step = 1, description = "w0")
    widget_b = widgets.FloatSlider(min = -20, max = 20, value = -8, step = 1, description = "b")


    def plot_lines(change):
        w2 = widget_w2.value
        w1 = widget_w1.value
        w0 = widget_w0.value
        b = widget_b.value
        lines.y = f(xs, w2, w1, w0, b)


    widget_w2.observe(plot_lines)
    widget_w1.observe(plot_lines)
    widget_w0.observe(plot_lines)
    widget_b.observe(plot_lines)

    h = HTML(value='<font size="+1">Parameters</font>')
    v_opti = widgets.VBox([h, HTML(value=''), widget_w2, widget_w1, widget_w0, widget_b])


    v_opti.layout.align_items = 'center'

    v_out = widgets.HBox([grad3_fig2, v_opti])
    v_out.layout.align_items = 'center'
    display(v_out)
    
    
import plotly
import plotly.graph_objs as go
import ipywidgets as widgets
import plotly.offline as py
from sklearn.metrics import mean_squared_error
import numpy as np


def show_MSE(X, y, resolution = 50, x_range=[-40, 40], y_range=[0, 80]):
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
        y_pred = w1 + w2*X
        return mean_squared_error(y, y_pred)

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
