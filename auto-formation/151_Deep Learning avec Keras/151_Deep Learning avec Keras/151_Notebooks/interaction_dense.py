# import ipyvolume as ipv
import numpy as np
import ipywidgets as widgets
import bqplot as bq
import bqplot.pyplot as plt
from IPython.display import display
import time
from sklearn import datasets
import pandas as pd
import json as json
from sklearn.datasets import make_moons


def show_dotProduct():
	plt.clear()
	x = np.array([1.0, 0.0])

	w = np.array([0, 1.0])

	w_ort = np.array([-w[1], w[0]])
	w_ort = w_ort/np.linalg.norm(w_ort, ord = 2)

	#dot_fig = plt.figure()


	dot_lin_sc = bq.LinearScale(min = -5, max = 5)

	dot_x_point = plt.scatter(x = np.array([x[0]]),
						 y = np.array([x[1]]),
						 scales={'x': dot_lin_sc, 'y': dot_lin_sc},
						 colors = ['blue'])

	dot_x_lineto_w = plt.plot(x = [np.dot(w, x)*w[0], x[0]],
						  y = [np.dot(w, x)*w[1], x[1]],
						  colors = ['red'],
						  scales={'x': dot_lin_sc, 'y': dot_lin_sc},
						  line_style = 'dashed'
						  ) 

	dot_x_proj_w = plt.plot(x = [np.dot(w, x)*w[0], 0],
						y = [np.dot(w, x)*w[1], 0],
						colors = ['red'],
						scales={'x': dot_lin_sc, 'y': dot_lin_sc},
						line_style = 'dashed')

	dot_w_line = plt.plot(x =[0, w[0], 0.8*w[0] - w[1]/10, w[0], 0.8*w[0] + w[1]/10],
					  y = [0, w[1], 0.8*w[1] + w[0]/10, w[1], 0.8*w[1] - w[0]/10],
					  scales={'x': dot_lin_sc, 'y': dot_lin_sc},
					  colors = ['green'])

	dot_w_plane = plt.plot(x = [- w_ort[0]/np.linalg.norm(w_ort, ord = 2)*30, w_ort[0]/np.linalg.norm(w_ort, ord = 2)*30],
					   y = [- w_ort[1]/np.linalg.norm(w_ort, ord = 2)*30, w_ort[1]/np.linalg.norm(w_ort, ord = 2)*30],
					   scales={'x': dot_lin_sc, 'y': dot_lin_sc},
					   colors = ['green'])

	dot_label_x = plt.label([" "],
						x = [x[0]],
						y = [x[1]],
						x_offset = -50,
						y_offset = -15,
						default_size = 12,
						font_weight = 'bolder',
						update_on_move = True,
						colors = ["blue"])

	dot_label_w = plt.label(["w = ("+ str(w[0])+", " +str(w[1]) + ")"],
						x = [w[0]],
						y = [w[1]],
						x_offset = 10,
						y_offset = -10,
						default_size = 12,
						font_weight = 'bolder',
						colors = ["green"])

	dot_label_dot_xw = plt.label([" "],
						x = [w[0]],
						y = [w[1]],
						x_offset = 5,
						y_offset = 5,
						default_size = 24,
						font_weight = 'bolder',
						colors = ["blue"])


	dot_ax_x = bq.Axis(scale=dot_lin_sc,
					grid_lines='solid')

	dot_ax_y = bq.Axis(scale=dot_lin_sc,
					orientation='vertical',
					grid_lines='solid')

	dot_fig = plt.Figure(marks = [dot_x_point, dot_x_lineto_w, dot_x_proj_w, dot_w_line,
								  dot_w_plane, dot_label_x, dot_label_w, dot_label_dot_xw],
						 axes = [dot_ax_x, dot_ax_y],
						 title = "Classification using Dot Product")
	dot_fig.layout.height = '400px'
	dot_fig.layout.width = '400px'

	display(dot_fig)

	# progress.value += 1

	@widgets.interact(
			  x1 = widgets.FloatSlider(min=-4, max=4, step=0.1, value=-2),
			  x2 = widgets.FloatSlider(min=-4, max=4, step=0.1, value=1.0),
			  w1 = widgets.FloatSlider(min=-4, max=4, step=0.1, value=0),
			  w2 = widgets.FloatSlider(min=-4, max=4, step=0.1, value=2))

	def dot_classification(x1, x2, w1, w2):
		dot_x_point.x = [x1]
		dot_x_point.y = [x2]
		
		w = np.array([w1, w2])
		w_ort = np.array([-w[1], w[0]])
		w_ort = w_ort/np.linalg.norm(w_ort, ord = 2)
		
		dot_w_line.x =[0, w[0], 0.8*w[0] - w[1]/10, w[0], 0.8*w[0] + w[1]/10]
		dot_w_line.y = [0, w[1], 0.8*w[1] + w[0]/10, w[1], 0.8*w[1] - w[0]/10]
		
		dot_w_plane.x = [- w_ort[0]/np.linalg.norm(w_ort, ord = 2)*30, w_ort[0]/np.linalg.norm(w_ort, ord = 2)*30]
		dot_w_plane.y = [- w_ort[1]/np.linalg.norm(w_ort, ord = 2)*30, w_ort[1]/np.linalg.norm(w_ort, ord = 2)*30]
		
		dot_label_w.x = [w1]
		dot_label_w.y = [w2]
		dot_label_w.text = ["w = (" + str(w1) + ", " + str(w2) + ")"]
		
		dot_label_x.x = [x1]
		dot_label_x.y = [x2]
		dot_label_x.text = ["x = (" + str(x1) + ", " + str(x2) + ")"]
		
		norm_w = np.linalg.norm(w, ord = 2)
		
		dot_label_dot_xw.text = ["<w,x> = " + str(np.round(np.dot(w, np.array([x1, x2])), 3))] 
		dot_label_dot_xw.x = [np.dot(w/norm_w, np.array([x1, x2]))*(w[0]/norm_w)]
		dot_label_dot_xw.y = [np.dot(w/norm_w, np.array([x1, x2]))*(w[1]/norm_w)]
		
		if(np.dot(w/norm_w, np.array([x1, x2])) < 0):
			dot_label_x.colors = ['red']
			dot_label_dot_xw.colors = ['red']
			dot_x_point.colors = ['red']
		elif(np.dot(w/norm_w, np.array([x1, x2])) > 0):
			dot_label_x.colors = ['blue']
			dot_label_dot_xw.colors = ['blue']
			dot_x_point.colors = ['blue']
		else:
			dot_label_x.colors = ['green']
			dot_label_dot_xw.colors = ['green']
			dot_x_point.colors = ['green']
		
		dot_x_lineto_w.x = [np.dot(w/norm_w, np.array([x1, x2]))*(w[0]/norm_w), x1]
		dot_x_lineto_w.y = [np.dot(w/norm_w, np.array([x1, x2]))*(w[1]/norm_w), x2]
		
		dot_x_proj_w.x = [0,np.dot(w/norm_w, np.array([x1, x2]))*(w[0]/norm_w)]
		dot_x_proj_w.y = [0, np.dot(w/norm_w, np.array([x1, x2]))*(w[1]/norm_w)]
		
		


def show_data():
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

	################################################################################    
		
								# Scales

	from IPython.display import display

	sep2_x_sc = plt.LinearScale(min = -4, max = 4)
	sep2_y_sc = plt.LinearScale(min = -4, max = 4)
		 
	sep2_ax_x = plt.Axis(scale=sep2_x_sc,
					grid_lines='solid',
					label='Sepal Length')

	sep2_ax_y = plt.Axis(scale=sep2_y_sc,
					orientation='vertical',
					grid_lines='solid',
					label='Sepal Width')
							
							  # Scatter plot
		
	sep2_bar = plt.Scatter(x = scaled[:,0],
					  y = scaled[:,1]-1,
					  colors = colors,
					  default_size = 10,
					  scales={'x': sep2_x_sc, 'y': sep2_y_sc})

								 # Vector
		
	w1, w2 = 1.0, 1.0
	w = np.array([w1, w2])

	sep2_vector_line = plt.Lines(x = np.array([0, w1]),
							y = np.array([0, w2]),
							colors = ['red', 'red'],
							scales={'x': sep2_x_sc, 'y': sep2_y_sc})

	sep2_vector_label = plt.Label(x = [w1],
							 y = [w2],
							 text = ['(w1, w2)'],
							 size = [10])

	sep2_vector_plane = plt.Lines(x = [-30*(w2 / np.linalg.norm(w)), 30*(w2 / np.linalg.norm(w))],
							 y = [30*(w1 / np.linalg.norm(w)), -30*(w1 / np.linalg.norm(w))],
							 colors = ['red', 'red'],
							 scales={'x': sep2_x_sc, 'y': sep2_y_sc})

	sep2_f = plt.Figure(marks=[sep2_bar, sep2_vector_line, sep2_vector_label, sep2_vector_plane],
				   axes=[sep2_ax_x, sep2_ax_y],
				   title='Iris Dataset',
				   legend_location='bottom-right')

	sep2_f.layout.height = '400px'
	sep2_f.layout.width = '400px'

	display(sep2_f)

	# progress.value += 1

	@widgets.interact(
			  w1 = widgets.FloatSlider(min=-4, max=4, step=0.11, value=1.0),
			  w2 = widgets.FloatSlider(min=-4, max=4, step=0.11, value=1.0))

					  # Fonction qui va interagir avec les widgets
		
	def h(w1, w2):
		sep2_vector_line.x = [0, w1, 0.8*w1 - w2/10, w1, 0.8*w1 + w2/10]
		sep2_vector_line.y = [0, w2, 0.8*w2 + w1/10, w2, 0.8*w2 - w1/10]
		w = np.array([w1, w2])
		sep2_vector_plane.x = [-30*w2 / np.linalg.norm(w), 30*w2 / np.linalg.norm(w)]
		sep2_vector_plane.y = [30*w1 / np.linalg.norm(w), -30*w1 / np.linalg.norm(w)]
		

		

def show_optimization():
	def f2(xs):
		x = xs/5
		return x**4 + x**3 - 6*x**2 + 1 

	def f2_p(xs):
		x = xs/5
		return 4*(x**3) + 3*(x)**2 - 12*x

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

	xk = np.random.normal(0,5,1)

	grad3_label_f_prim = plt.label([" "],
						x = xk,
						y = f2(xk),
						x_offset = 0,
						y_offset = 0,
						default_size = 20,
						font_weight = 'bolder',
						update_on_move = True,
						colors = ["green"])

	grad3_label_x0 = plt.label(["x0"],
						x = [-5],
						y = [0],
						x_offset = 10,
						y_offset = 0,
						default_size = 20,
						font_weight = 'bolder',
						update_on_move = True,
						colors = ["red"])

	grad3_lines_x0 = plt.Lines(x = [-5, -5],
						 y = [0, 25],
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
							title = "Non-Convex Gradient Descent")

	grad3_fig2.layout.height = '400px'
	grad3_fig2.layout.width = '600px'

	#display(fig2)

	grad3_x0 = widgets.FloatSlider(min = -15, max = 11, value = -5, step = 1, description = "x0")
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
		grad3_label_f_prim.text = ["Step " + str(grad3_etape.value) + ", f'(x"+str(grad3_etape.value)+") = "+str(np.round(f2_p(xk[grad3_etape.value]), 3))]

	grad3_etape_play.observe(gradient_plot2)
	grad3_x0.observe(gradient_plot2)
	grad3_learning_rate.observe(gradient_plot2)

	display(widgets.VBox([grad3_fig2, grad3_x0, grad3_learning_rate, grad3_etape, grad3_etape_play]))


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
    # plotly.offline.init_notebook_mode()

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

    
    
def show_mlp():
	X, y = make_moons(n_samples = 100)

	X_0 = X[y == 0]
	X_1 = X[y == 1]

	# Scales

	moons_x_sc= plt.LinearScale(min = -1, max = 2)

	moons_y_sc= plt.LinearScale(min = -1, max = 1)

	# Axes

	moons_ax_x = plt.Axis(scale = moons_x_sc,
						  grid_lines = 'solid',
						  label ='x')

	moons_ax_y = plt.Axis(scale = moons_y_sc,
						  grid_lines = 'solid',
						  orientation = 'vertical',
						  label = 'y')

	# Scatter plots

	moons_x0 = plt.Scatter(x = X_0[:,0], y = X_0[:,1],
						   scales = {'x': moons_x_sc, 'y': moons_y_sc},
						   colors = ['blue'])

	moons_x1 = plt.Scatter(x = X_1[:,0], y = X_1[:,1],
						   scales = {'x': moons_x_sc, 'y': moons_y_sc},
						   colors = ['red'])


	with open('decision_boundaries.txt') as json_file:
		decision_boundaries2 = json.load(json_file)
		
	db = np.sort(np.array(decision_boundaries2['relu']['3']['1']), axis = 0)
	moons_db = plt.Scatter(x = db[:,0],
						   y = db[:,1],
						   scales = {'x': moons_x_sc, 'y': moons_y_sc},
						   default_size = 4,
						   size = [3],
						   colors = ['green'])

	moons_fig2 = plt.Figure(marks = [moons_x0, moons_x1, moons_db],
							axes = [moons_ax_x, moons_ax_y],
							animation_duration = 500,
							title = "Decision Boundaries on Moons Dataset")
	moons_fig2.layout.height = '400px'
	moons_fig2.layout.width = '500px'

	display(moons_fig2)

	style = {'description_width': 'initial'}

	@widgets.interact(activation = widgets.Dropdown(options=['relu', 'tanh'],
												  value='relu',
												  description='Activation Function',
												  disabled = False,
												  style = style),
					  n_layers = widgets.Dropdown(options=['2', '3', '30', '90'],
												  value='3',
												  description='Hidden Layer Size',
												  style = style,
												  disabled = False),
					  learning_rate = widgets.Dropdown(options=['0.001', '0.01', '0.1', '1'],
												  value='0.001',
												  description='Learning Rate',
												  style = style,
												  disabled = False)
					  )
	def moons_interaction(activation,n_layers,learning_rate):
		db = np.array(decision_boundaries2[activation][n_layers][learning_rate])
		
		moons_db.x = db[:,0]
		moons_db.y = db[:,1]

from IPython.display import display
def show_conv():
	# Interactive Convolution
	if(True):
		import cv2
		from scipy import ndimage
		import matplotlib as mpl
		from io import BytesIO
		from skimage import data
		from IPython.display import Markdown

		# Image

		taj_mahal = cv2.imread("taj_mahal.jpg",0)

		# Fonctions
		if(True):
			# Array to bytes converter
			if(True):
				def arr2bytes(arr):
					"""Display a 2- or 3-d numpy array as an image."""
					if arr.ndim == 2:
						format, cmap = 'png', mpl.cm.gray
					elif arr.ndim == 3:
						format, cmap = 'jpg', None
					else:
						raise ValueError("Only 2- or 3-d arrays can be displayed as images.")
					# Don't let matplotlib autoscale the color range so we can control overall luminosity
					vmax = 255 if arr.dtype == 'uint8' else 1.0
					with BytesIO() as buffer:
						mpl.image.imsave(buffer, arr, format=format, cmap=cmap, vmin=0, vmax=vmax)
						out = buffer.getvalue()
					return out

		# Widgets
		if(True): 
			# Image widget
			if(True):
				img_widget = widgets.Image(value = arr2bytes(taj_mahal),
										   format = 'png',
										   width = 600,
										   height = 600)

			# Kernel Widgets
			if(True):
				w1 = widgets.FloatText(value = -1, layout = widgets.Layout(width = '40px'))
				w2 = widgets.FloatText(value = -1, layout = widgets.Layout(width = '40px'))
				w3 = widgets.FloatText(value = -1, layout = widgets.Layout(width = '40px'))
				w4 = widgets.FloatText(value = -1, layout = widgets.Layout(width = '40px'))
				w5 = widgets.FloatText(value = 8, layout = widgets.Layout(width = '40px'))
				w6 = widgets.FloatText(value = -1, layout = widgets.Layout(width = '40px'))
				w7 = widgets.FloatText(value = -1, layout = widgets.Layout(width = '40px'))
				w8 = widgets.FloatText(value = -1, layout = widgets.Layout(width = '40px'))
				w9 = widgets.FloatText(value = -1, layout = widgets.Layout(width = '40px'))

				kernel_grid = widgets.GridBox(children=[w1, w2 ,w3, w4, w5, w6, w7, w8, w9],
											  layout=widgets.Layout(width='300 px',
																	grid_template_rows='30px 30px 30px',
																	grid_template_columns='',
																	grid_template_areas='''
																						"w1 w2 w3"
																						"w4 w5 w6"
																						"w7 w8 w9"'''))

			# Label
			if(True):
				label = widgets.Label("Convolution Kernel", layout = widgets.Layout(margin = "10px"))
				label2 = widgets.Output(layout = widgets.Layout(margin = "32px 0px 0px 0px"))

			# Widget Box
			if(True):
				kernel_box1 = widgets.HBox(children = [kernel_grid])
				
				kernel_box = widgets.VBox(children = [label, kernel_box1],
										  layout = widgets.Layout(height = "200px 200px",
																  justify_content = "center",
																  margin = "50px"))
				
			# Radio Buttons
			if(True):
				radio_buttons = widgets.RadioButtons(
										options = ['Identity', 'Contrast','Edge Detection',
												   'Vertical Edge Detection', 'Horizontal Edge Detection',
												   'Gaussian Blur (Low Intensity)'],
										value = 'Identity',
										#description = 'Pizza topping:',
										disabled=False
										)

		# Interaction
		if(True):
			# Handler
			radio_buttons.prev = "Hello"
			if(True):
				def interactive_convolution(change):
					kernel_const = 1
					if(radio_buttons.prev != radio_buttons.value):
						# Unobserve
						if(True):
							w1.unobserve(interactive_convolution, 'value')
							w2.unobserve(interactive_convolution, 'value')
							w3.unobserve(interactive_convolution, 'value')
							w4.unobserve(interactive_convolution, 'value')
							w5.unobserve(interactive_convolution, 'value')
							w6.unobserve(interactive_convolution, 'value')
							w7.unobserve(interactive_convolution, 'value')
							w8.unobserve(interactive_convolution, 'value')
							w9.unobserve(interactive_convolution, 'value')
							
						# Identity
						if(radio_buttons.value == "Identity"):
							w1.value = 0
							w2.value = 0
							w3.value = 0

							w4.value = 0
							w5.value = 1
							w6.value = 0

							w7.value = 0
							w8.value = 0
							w9.value = 0
							radio_buttons.prev = "Identity"

						# Contrast
						if(radio_buttons.value == "Contrast"):
							w1.value = 0
							w2.value = -1
							w3.value = 0

							w4.value = -1
							w5.value = 5
							w6.value = -1

							w7.value = 0
							w8.value = -1
							w9.value = 0
							radio_buttons.prev = "Contrast"
							
						# Edge Detection
						if(radio_buttons.value == "Edge Detection"):
							w1.value = -1
							w2.value = -1
							w3.value = -1

							w4.value = -1
							w5.value = 8
							w6.value = -1

							w7.value = -1
							w8.value = -1
							w9.value = -1
							radio_buttons.prev = "Edge Detection"
							
						# Vertical Edge Detection
						if(radio_buttons.value== "Vertical Edge Detection"):
							w1.value = 1
							w2.value = 0
							w3.value = -1

							w4.value = 1
							w5.value = 0
							w6.value = -1

							w7.value = 1
							w8.value = 0
							w9.value = -1
							radio_buttons.prev = "Vertical Edge Detection"

						# Horizontal Edge Detection
						if(radio_buttons.value== "Horizontal Edge Detection"):
							w1.value = 1
							w2.value = 1
							w3.value = 1

							w4.value = 0
							w5.value = 0
							w6.value = 0

							w7.value = -1
							w8.value = -1
							w9.value = -1
							radio_buttons.prev = "Horizontal Edge Detection"

						# Gaussian Blur (Low Intensity)
						if(radio_buttons.value == "Gaussian Blur (Low Intensity)"):
							kernel_const = 1/16

							w1.value = 1
							w2.value = 2
							w3.value = 1

							w4.value = 2
							w5.value = 4
							w6.value = 2

							w7.value = 1
							w8.value = 2
							w9.value = 1
							radio_buttons.prev = "Gaussian Blur (Low Intensity)"
						
						# Observe
						if(True):
							w1.observe(interactive_convolution, 'value')
							w2.observe(interactive_convolution, 'value')
							w3.observe(interactive_convolution, 'value')
							w4.observe(interactive_convolution, 'value')
							w5.observe(interactive_convolution, 'value')
							w6.observe(interactive_convolution, 'value')
							w7.observe(interactive_convolution, 'value')
							w8.observe(interactive_convolution, 'value')
							w9.observe(interactive_convolution, 'value')

						radio_buttons.changed = False

					kernel = np.array([[w1.value, w2.value, w3.value],
									   [w4.value, w5.value, w6.value],
									   [w7.value, w8.value, w9.value]])
					convolved_img = ndimage.convolve(taj_mahal/255, kernel, mode = 'mirror')
					convolved_img = convolved_img * kernel_const
					img_widget.value = arr2bytes(convolved_img)

			# Observes
			if(True):
				w1.observe(interactive_convolution, 'value')
				w2.observe(interactive_convolution, 'value')
				w3.observe(interactive_convolution, 'value')
				w4.observe(interactive_convolution, 'value')
				w5.observe(interactive_convolution, 'value')
				w6.observe(interactive_convolution, 'value')
				w7.observe(interactive_convolution, 'value')
				w8.observe(interactive_convolution, 'value')
				w9.observe(interactive_convolution, 'value')
				radio_buttons.observe(interactive_convolution, 'value')

		# Display
		if(True):
			
			display(widgets.HBox(children = [kernel_box, img_widget], layout = widgets.Layout(width = "200px 200px")), radio_buttons)
			#interactive_convolution()
    
	
	
def show_plane():
	def plot_3d_vector(w1, w2, w3, bias = 0, radius = 0.5, length = 1, resolution = 100):
		# Vecteurs du plan 
		w = np.array([w1, w2, w3])
		w  =length * w / np.linalg.norm(w, ord = 2)
		p1 = np.array([0, -w3, w2])                     
		p2 = np.array([-(w2*w2/w3 + w3)/w1, w2/w3, 1])

		p1 = p1 / np.linalg.norm(p1, ord = 2)
		p2 = p2 / np.linalg.norm(p2 , ord = 2)
		
		# Coordonnées des deux cercles
		if(resolution % 2 == 0):
			n_thetas = resolution + 1
		else:
			n_thetas = resolution
		
		thetas = np.linspace(0, 2*np.pi, n_thetas)
	   
		base_circle = np.array([radius * (p1 * np.cos(thetas[i]) + p2 * np.sin(thetas[i])) for i in range(n_thetas)])

		top_circle_1 = base_circle + w * 0.5

		top_circle_2 = np.array([(radius * 1.5) * (p1 * np.cos(thetas[i]) + p2 * np.sin(thetas[i])) for i in range(n_thetas)])
		top_circle_2 = top_circle_2 + w * 0.5
		
		# Coordonnées de la trajectoire du plot
		
		circles = [top_circle_1[i,:] for i in range(n_thetas)]

		i = 0
		while(i < n_thetas-2):
			circles += [top_circle_1[i,:], top_circle_1[i+1,:], base_circle[i+1,:], base_circle[i+2,:]]
			i += 2
		
			
		circles += [base_circle[i,:] for i in range(n_thetas)]    
		circles += [top_circle_1[0,:]]
		i = 0
		while(i < n_thetas-2):
			circles += [top_circle_1[i,:], top_circle_1[i+1,:], top_circle_2[i+1,:], top_circle_2[i+2,:]]
			i += 2
		
		circles += [top_circle_2[0,:]]
		circles += [top_circle_2[i,:] for i in range(n_thetas)] 
		
		i = 0
		while(i < n_thetas-2):
			circles += [w, top_circle_2[i,:], w, top_circle_2[i+1,:]]
			i += 2
		
		circles = np.array(circles)    
		
		circles_x = circles[:,0]
		circles_y = circles[:,1]
		circles_z = circles[:,2] - bias/w3    
		
		return circles_x, circles_y, circles_z                          
		
		
		
		# Dimensions du plan       
	d1 = 20                                                                     
	d2 = 20                                          
	resolution = 50

	################################################################################
	#                                                                              #
	#                               Initialisation                                 #
	#                                                                              #
	################################################################################

						  # Vecteur définissant le plan

	w1, w2, w3 = 1.0, 1.0, 1.0   # vecteur orthogonal au plan


	p1 = np.array([0, -w3, w2])    # vecteurs du plan pour trouver les 4 coins
	p2 = np.array([-(w2*w2/w3 + w3)/w1, w2/w3, 1])

	p1 = d1 * p1 / np.linalg.norm(p1, ord = 2)            
	p2 = d2 * p2 / np.linalg.norm(p2 , ord = 2)

								# 4 coins du plan
	point2 = p1 + p2          
	point3 = p1 - p2
	point4 = - point2
	point1 = - point3

					 # Coordonnées pour dessiner la surface avec un grillage

	L = np.linspace(0, 1, resolution)
	L1 = []     # coordonnées x des points du grillage  
	L2 = []     # y
	L3 = []     # z

	for i in range(len(L)-1):
		i = len(L)-1-i
		L1 += [point1[0]*L[i]+point4[0]*(1-L[i]),point2[0]*L[i]+point3[0]*(1-L[i]),point2[0]*L[i-1]+point3[0]*(1-L[i-1])]
		L2 += [point1[1]*L[i]+point4[1]*(1-L[i]),point2[1]*L[i]+point3[1]*(1-L[i]),point2[1]*L[i-1]+point3[1]*(1-L[i-1])]
		L3 += [point1[2]*L[i]+point4[2]*(1-L[i]),point2[2]*L[i]+point3[2]*(1-L[i]),point2[2]*L[i-1]+point3[2]*(1-L[i-1])]
	L1 += [point4[0]]
	L2 += [point4[1]]
	L3 += [point4[2]]
	for i in range(len(L)-1):
		i = len(L)-1-i
		L1 += [point4[0]*L[i]+point3[0]*(1-L[i]),point1[0]*L[i]+point2[0]*(1-L[i]),point1[0]*L[i-1]+point2[0]*(1-L[i-1])]
		L2 += [point4[1]*L[i]+point3[1]*(1-L[i]),point1[1]*L[i]+point2[1]*(1-L[i]),point1[1]*L[i-1]+point2[1]*(1-L[i-1])]
		L3 += [point4[2]*L[i]+point3[2]*(1-L[i]),point1[2]*L[i]+point2[2]*(1-L[i]),point1[2]*L[i-1]+point2[2]*(1-L[i-1])]

						  # Initialisation de la figure

	d3plane_fig = ipv.figure()

					# Coordonnées d'un nuage de points aléatoire
		
	scat_x, scat_y, scat_z = np.random.normal(0, 5, (3,100))

		 # Couleurs des points (bleu si classification positive / vert sinon)
		
	cross_prods = np.array([np.dot(np.array([scat_x[i], scat_y[i], scat_z[i]]),np.array([w1, w2, w3])) for i in range(100)])
	colors = np.zeros((100,3))
	colors[np.where(cross_prods < 0)] = np.array([0, 0, 255])
	colors[np.where(cross_prods >= 0)] = np.array([0, 255, 0])

						  # Scatterplot aléatoire

	d3plane_scat = ipv.scatter(scat_x, scat_y, scat_z, color = colors)

						   # Plot de la surface
		
	d3plane_xdd = ipv.plot(np.array(L1),np.array(L2),np.array(L3))

						 # Plot du vecteur directeur

	vect_x, vect_y, vect_z = plot_3d_vector(w1, w2, w3, resolution = 10, length = 5)

	d3plane_vect = ipv.plot(vect_x, vect_y, vect_z, color = 'green')

							 # Styling de la figure
		
	d3plane_fig.animation = 0             # on enlève les animations  
	d3plane_fig.animation_exponent = 0

	ipv.xyzlim(-15,15)           # étendue des axes x,y,z

	d3plane_scat.geo = "sphere"         # glyphe sphere pour les points du nuage aléatoire

	ipv.show()       

	################################################################################
	#                                                                              #
	#                            Partie interactive                                #
	#                                                                              #
	################################################################################
						
				   # Sliders pour les coordonnées du vecteur + biais 

	#:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:#
	#                                                                                            #
	#               Se débrouiller pour ne mettre aucune des coordonnées de w à 0                #
	#                                                                                            #
	#:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:DANGER:#      

	@widgets.interact(
			  w1 = widgets.FloatSlider(min=-5, max=5, step=0.3, value=1.0),
			  w2 = widgets.FloatSlider(min=-5, max=5, step=0.3, value=1.0),
			  w3 = widgets.FloatSlider(min=-5, max=5, step=0.3, value=1.0),
			  bias = widgets.FloatSlider(min = -10, max = 10, step = 0.2, value = 0.0),
			  resolution = widgets.IntSlider(min = 10, max = 120, step = 10, value = 70))

					  # Fonction qui va interagir avec les widgets
		
	def g(w1, w2, w3, bias, resolution):
		
							# Vecteur directeurs du plan
		
		p1 = np.array([0, -w3, w2])                     
		p2 = np.array([-(w2*w2/w3 + w3)/w1, w2/w3, 1])

		p1 = d1 * p1 / np.linalg.norm(p1, ord = 2)
		p2 = d2 * p2 / np.linalg.norm(p2 , ord = 2)
		
								# 4 coins du plan
		
		point2 = p1 + p2          
		point3 = p1 - p2
		point4 = - point2
		point1 = - point3

						# Coordonnées pour dessiner la surface avec un grillage

		L = np.linspace(0, 1, resolution)
		L1 = []   # coordonnées x du grillage
		L2 = []   # y
		L3 = []   # z

		for i in range(len(L)-1):
			i = len(L)-1-i
			L1 += [point1[0]*L[i]+point4[0]*(1-L[i]),point2[0]*L[i]+point3[0]*(1-L[i]),point2[0]*L[i-1]+point3[0]*(1-L[i-1])]
			L2 += [point1[1]*L[i]+point4[1]*(1-L[i]),point2[1]*L[i]+point3[1]*(1-L[i]),point2[1]*L[i-1]+point3[1]*(1-L[i-1])]
			L3 += [point1[2]*L[i]+point4[2]*(1-L[i]),point2[2]*L[i]+point3[2]*(1-L[i]),point2[2]*L[i-1]+point3[2]*(1-L[i-1])]
			
		L1 += [point4[0]]
		L2 += [point4[1]]
		L3 += [point4[2]]
		
		for i in range(len(L)-1):
			i = len(L)-1-i
			L1 += [point4[0]*L[i]+point3[0]*(1-L[i]),point1[0]*L[i]+point2[0]*(1-L[i]),point1[0]*L[i-1]+point2[0]*(1-L[i-1])]
			L2 += [point4[1]*L[i]+point3[1]*(1-L[i]),point1[1]*L[i]+point2[1]*(1-L[i]),point1[1]*L[i-1]+point2[1]*(1-L[i-1])]
			L3 += [point4[2]*L[i]+point3[2]*(1-L[i]),point1[2]*L[i]+point2[2]*(1-L[i]),point1[2]*L[i-1]+point2[2]*(1-L[i-1])]

			   # Couleurs des points (bleu si classification positive / vert sinon)
		
		cross_prods = np.array([np.dot(np.array([scat_x[i], scat_y[i], scat_z[i]]),np.array([w1, w2, w3])) for i in range(100)])
		colors = np.zeros((100,3))
		colors[np.where(cross_prods + bias < 0)] = np.array([0, 0, 255])
		colors[np.where(cross_prods + bias >= 0)] = np.array([0, 255, 0])
		d3plane_scat.color = colors
		
		
						 # M.à.j du grillage
		
		d3plane_xdd.x = np.array(L1)
		d3plane_xdd.y = np.array(L2)
		d3plane_xdd.z = np.array(L3) - bias/w3
		
						# M.à.j du vecteur orthogonal
			
		d3plane_vect.x, d3plane_vect.y, d3plane_vect.z = plot_3d_vector(w1, w2, w3, bias, resolution = 10, length = 5)
		
		
def show_optimization_square():
	xs = np.linspace(-10, 10, 100)
	ys = xs**2 + 2 #two random walks

	grad1_x_sc = plt.LinearScale(min = -10, max = 10)
	grad1_y_sc = plt.LinearScale(min = 0, max = 100)

	grad1_lines = plt.Lines(x=xs, y=ys, colors=['red'], scales = {'x': grad1_x_sc, 'y' : grad1_y_sc})

	grad1_label_min = plt.label(["Minimum"],
						x = [0],
						y = [2],
						scales = {'x': grad1_x_sc, 'y' : grad1_y_sc},
						x_offset = -30,
						y_offset = -15,
						default_size = 12,
						font_weight = 'bolder',
						update_on_move = True,
						colors = ["blue"])

	grad1_minimum = plt.scatter(x = [0],
						  y = [2],
						  colors = ["blue"])

	xk = np.random.normal(0,5,1)

	grad1_label_f_prim = plt.label([" "],
						x = xk,
						y = xk ** 2 + 2,
						x_offset = 0,
						y_offset = 0,
						default_size = 20,
						font_weight = 'bolder',
						update_on_move = True,
						colors = ["green"])

	grad1_point = plt.scatter(x = xk,
						y = xk ** 2 + 2,
						colors = ["green"])

	grad1_ax_x = bq.Axis(scale=grad1_x_sc,
					grid_lines='solid')

	grad1_ax_y = bq.Axis(scale=grad1_y_sc,
					orientation='vertical',
					grid_lines='solid')

	grad1_fig = plt.Figure(marks = [grad1_lines, grad1_label_min, grad1_minimum, grad1_label_f_prim, grad1_point],
						   axes = [grad1_ax_x, grad1_ax_y],
						   title = "Slope and Derivatives")

	grad1_fig.layout.height = '400px'
	grad1_fig.layout.width = '600px'


	display(grad1_fig)

	@widgets.interact(
			  x = widgets.FloatSlider(min=-10, max=10, step=0.2, value= -5))

	def gradient_plot(x):
		grad1_point.x = [x]
		grad1_point.y = [x**2 + 2]
		
		grad1_label_f_prim.x = [x]
		grad1_label_f_prim.y = [x**2 + 2]
		if(x < 0):
			grad1_label_f_prim.text = ["f'(x) = " + str(2 * x) + " < 0"]
			grad1_label_f_prim.x_offset = 10
			grad1_label_f_prim.colors = ["red"]
		if(x > 0):
			grad1_label_f_prim.text = ["f'(x) = " + str(2 * x) + " > 0"]
			grad1_label_f_prim.x_offset = - 140
			grad1_label_f_prim.colors = ["green"]
			
			
def show_dataset():
	plt.clear()
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

	sep1_x_sc = plt.LinearScale(min = -4, max = 4)
	sep1_y_sc = plt.LinearScale(min = -4, max = 4)
		 
	sep1_ax_x = plt.Axis(scale=sep1_x_sc,
					grid_lines='solid',
					label='Sepal Length')

	sep1_ax_y = plt.Axis(scale=sep1_y_sc,
					orientation='vertical',
					grid_lines='solid',
					label='Sepal Width')
							
							  # Scatter plot
		
	sep1_bar = plt.Scatter(x = scaled[:,0],
						   y = scaled[:,1]-1,
						   colors = colors,
						   default_size = 10,
						   scales={'x': sep1_x_sc, 'y': sep1_y_sc})


	sep1_f = plt.Figure(marks=[sep1_bar],
				   axes=[sep1_ax_x, sep1_ax_y],
				   title='Iris Dataset',
				   legend_location='bottom-right')

	sep1_f.layout.height = '400px'
	sep1_f.layout.width = '400px'

	display(sep1_f)
	
	

	
	
def show_gradient_descent():
	xs = np.linspace(-10, 10, 100)
	ys = xs**2 + 2

	grad2_x_sc = plt.LinearScale(min = -10, max = 10)
	grad2_y_sc = plt.LinearScale(min = 0, max = 100)

	grad2_lines = plt.Lines(x=xs, y=ys, colors=['red', 'green'],scales = {'x': grad2_x_sc, 'y' : grad2_y_sc})

	grad2_label_min = plt.label(["Minimum"],
						x = [0],
						y = [2],
						x_offset = -30,
						y_offset = -15,
						scales = {'x': grad2_x_sc, 'y' : grad2_y_sc},
						default_size = 12,
						font_weight = 'bolder',
						update_on_move = True,
						colors = ["blue"])

	grad2_minimum = plt.Scatter(x = [0],
						  y = [2],
						  colors = ["blue"],
						  scales = {'x': grad2_x_sc, 'y' : grad2_y_sc})

	xk = np.random.normal(0,5,1)

	grad2_label_f_prim = plt.label([" "],
						x = xk,
						y = xk ** 2 + 2,
						x_offset = 0,
						y_offset = 0,
						default_size = 20,
						font_weight = 'bolder',
						update_on_move = True,
						colors = ["green"])

	grad2_label_x0 = plt.label(["x0"],
						x = [-5],
						y = [0],
						x_offset = 10,
						y_offset = 0,
						default_size = 20,
						font_weight = 'bolder',
						update_on_move = True,
						colors = ["red"])

	grad2_lines_x0 = plt.Lines(x = [-5, -5],
						 y = [0, 25],
						 scales = {'x': grad2_x_sc, 'y' : grad2_y_sc},
						 line_style = "dashed",
						 colors = ["red"])

	grad2_point = plt.Scatter(x = xk,
						y = xk ** 2 + 2,
						colors = ["green"],
						scales = {'x': grad2_x_sc, 'y' : grad2_y_sc})

	grad2_point_lines = plt.Lines(x = xk,
							y = xk ** 2 + 2,
							colors = ["green"],
							scales = {'x': grad2_x_sc, 'y' : grad2_y_sc})

	grad2_ax_x = bq.Axis(scale=grad2_x_sc,
					grid_lines='solid')

	grad2_ax_y = bq.Axis(scale=grad2_y_sc,
					orientation='vertical',
					grid_lines='solid')

	grad2_fig2 = plt.Figure(marks = [grad2_lines, grad2_point_lines, grad2_lines_x0,
									 grad2_label_min, grad2_minimum, grad2_label_f_prim,
									 grad2_label_x0, grad2_point],
							axes = [grad2_ax_x, grad2_ax_y],
							title = "Convex Gradient Descent")

	grad2_fig2.layout.height = '400px'
	grad2_fig2.layout.width = '600px'

	#display(fig2)

	grad2_x0 = widgets.FloatSlider(min = -10, max = 10, value = -5, step = 0.2, description = "x0")
	grad2_learning_rate = widgets.BoundedFloatText(min=0.001, max=1.5, step=0.01, value = 0.9, description = "Learning rate")
	grad2_etape_play = widgets.Play(value = 0,interval = 50, min=0, max=50, step=1, disabled=False)
	grad2_etape = widgets.IntSlider(value = 0, min = 0, max = 50, step = 1, description = "Step")
	grad2_hbox = widgets.HBox([grad2_etape_play,grad2_etape])
	widgets.jslink((grad2_etape_play, 'value'), (grad2_etape, 'value'))

	def grad2_gradient_plot(change):

		xk = np.zeros(101)
		xk[0] = grad2_x0.value
		for k in np.arange(100)+1:
			 xk[k] = xk[k-1] - grad2_learning_rate.value*2*xk[k-1]
		grad2_point.x = xk[:grad2_etape.value+1]
		grad2_point.y = xk[:grad2_etape.value+1] ** 2
			
		grad2_point_lines.x = xk[:grad2_etape.value+1]
		grad2_point_lines.y = xk[:grad2_etape.value+1] ** 2
		
		grad2_label_x0.x = [grad2_x0.value]
		grad2_lines_x0.x = [grad2_x0.value, grad2_x0.value]
		grad2_lines_x0.y = [0, grad2_x0.value ** 2]
		
		grad2_label_f_prim.x = [-8]
		grad2_label_f_prim.y = [90]
		grad2_label_f_prim.text = ["Step " + str(grad2_etape.value) + ", f'(x"+str(grad2_etape.value)+") = "+str(np.round(2*xk[grad2_etape.value], 3))]

	grad2_etape_play.observe(grad2_gradient_plot)
	grad2_x0.observe(grad2_gradient_plot)
	grad2_learning_rate.observe(grad2_gradient_plot)

	display(widgets.VBox([grad2_fig2, grad2_x0, grad2_learning_rate, grad2_etape, grad2_etape_play]))

	
	
	
	
def show_dense():
	plt.clear()

	mlp_x_sc = plt.LinearScale(min = -1, max = 6)
	mlp_y_sc = plt.LinearScale(min = -5, max = 5)

	mlp_ax_x = bq.Axis(scale= mlp_x_sc)

	mlp_ax_y = bq.Axis(scale= mlp_y_sc, orientation='vertical')

	mlp_point1 = plt.Scatter(x = [0], y = [0], scales = {'x': mlp_x_sc, 'y': mlp_y_sc})

	mlp_point2 = plt.Scatter(x = [0], y = [0], scales = {'x': mlp_x_sc, 'y': mlp_y_sc})

	mlp_point3 = plt.Scatter(x = [0], y = [0], scales = {'x': mlp_x_sc, 'y': mlp_y_sc})

	mlp_input_label = plt.Label(text = ['Input'], x = [0], y = [4],
								scales = {'x': mlp_x_sc, 'y' : mlp_y_sc},
								default_size = 12,
								default_opacities = [0.5],
								font_weight = 'bolder',
								update_on_move = True,
								colors = ["blue"])

	mlp_input_layer_label = plt.Label(text = ['Input Layer'], x = [-0.8], y = [-5],
									  scales = {'x': mlp_x_sc, 'y' : mlp_y_sc},
									  default_size = 20,
									  default_opacities = [0.9],
									  font_weight = 'bolder',
									  update_on_move = True,
									  colors = ["steelblue"])

	mlp_hidden_layer_label = plt.Label(text = ['Hidden Layer'], x = [1.1], y = [-5],
									   scales = {'x': mlp_x_sc, 'y' : mlp_y_sc},
									   default_size = 20,
									   default_opacities = [0.65],
									   font_weight = 'bolder',
									   update_on_move = True,
									   colors = ["red"])

	mlp_out_layer_label = plt.Label(text = ['Output Layer'], x = [4.1], y = [-5],
									   scales = {'x': mlp_x_sc, 'y' : mlp_y_sc},
									   default_size = 20,
									   default_opacities = [0.65],
									   font_weight = 'bolder',
									   update_on_move = True,
									   colors = ["green"])

	mlp_dotprod1_label = plt.Label(text = ['Dot Product 1'], x = [2], y = [4],
								   scales = {'x': mlp_x_sc, 'y' : mlp_y_sc},
								   default_size = 12,
								   default_opacities = [0.5],
								   font_weight = 'bolder',
								   update_on_move = True,
								   colors = ["blue"])

	mlp_dotprod2_label = plt.Label(text = ['Dot Product 2'], x = [2], y = [4],
								   scales = {'x': mlp_x_sc, 'y' : mlp_y_sc},
								   default_size = 12,
								   default_opacities = [0.5],
								   font_weight = 'bolder',
								   update_on_move = True,
								   colors = ["blue"])

	mlp_activ1_label = plt.Label(text = ["Activation 1"], x = [3], y = [4],
								 scales = {'x': mlp_x_sc, 'y' : mlp_y_sc},
								 default_size = 12,
								 default_opacities = [0.5],
								 font_weight = 'bolder',
								 update_on_move = True,
								 colors = ["blue"])

	mlp_activ2_label = plt.Label(text = ["Activation 2"], x = [3], y = [4],
							   scales = {'x': mlp_x_sc, 'y' : mlp_y_sc},
							   default_size = 12,
							   default_opacities = [0.5],
							   font_weight = 'bolder',
							   update_on_move = True,
							   opacity = [0.5],
							   colors = ["blue"])

	mlp_conc_label = plt.Label(text = ["Concatenation of Outputs"], x = [4], y = [4],
							   scales = {'x': mlp_x_sc, 'y' : mlp_y_sc},
							   default_size = 12,
							   default_opacities = [0.5],
							   font_weight = 'bolder',
							   update_on_move = True,
							   colors = ["blue"])

	mlp_class_label = plt.Label(text = ["Classification"], x = [5], y = [4],
							   scales = {'x': mlp_x_sc, 'y' : mlp_y_sc},
							   default_size = 12,
							   default_opacities = [0.5],
							   font_weight = 'bolder',
							   update_on_move = True,
							   colors = ["blue"])

	mlp_perceptrons = plt.Scatter(x = [0, 2, 2, 2, 5],
								  y = [0, -3.5, 0, 3.5, 0],
								  size = [900, 900 , 900, 900, 900],
								  default_size = 900,
								  colors = ['steelblue','red', 'red', 'red', 'green'],
								  scales = {'x': mlp_x_sc, 'y': mlp_y_sc})

	mlp_arete1_1 = plt.Scatter(x = np.zeros(80),
							 y = np.zeros(80),
							 size = [10, 10, 10],
							 default_size = 10,
							 scales = {'x': mlp_x_sc, 'y': mlp_y_sc})

	mlp_arete1_2 = plt.Scatter(x = np.zeros(80),
							 y = np.zeros(80),
							 size = [10, 10, 10],
							 default_size = 10,
							 scales = {'x': mlp_x_sc, 'y': mlp_y_sc})

	mlp_arete1_3 = plt.Scatter(x = np.zeros(80),
							 y = np.zeros(80),
							 size = [10, 10, 10],
							 default_size = 10,
							 scales = {'x': mlp_x_sc, 'y': mlp_y_sc})

	mlp_arete2_1 = plt.Scatter(x = np.zeros(100) + 2,
							  y = np.zeros(100) + 3.5,
							  colors = ['red'],
							  size = [10, 10, 10],
							  default_size = 10,
							  scales = {'x': mlp_x_sc, 'y': mlp_y_sc})

	mlp_arete2_2 = plt.Scatter(x = np.zeros(100) + 2 ,
							  y = np.zeros(100),
							  colors = ['red'],
							  size = [10, 10, 10],
							  default_size = 10,
							  scales = {'x': mlp_x_sc, 'y': mlp_y_sc})

	mlp_arete2_3 = plt.Scatter(x = np.zeros(100) + 2,
							   y = np.zeros(100) - 3.5,
							   colors = ['red'],
							   size = [10, 10, 10],
							   default_size = 10,
							   scales = {'x': mlp_x_sc, 'y': mlp_y_sc})

	mlp_arete3_1 = plt.Scatter(x = np.zeros(50) + 5,
							   y = np.zeros(50) ,
							   colors = ['green'],
							   size = [10, 10, 10],
							   default_size = 10,
							   scales = {'x': mlp_x_sc, 'y': mlp_y_sc})



	mlp_fig = plt.Figure(marks = [mlp_input_layer_label, mlp_hidden_layer_label,mlp_out_layer_label,
								  mlp_arete1_1,mlp_arete1_2, mlp_arete1_3,
								  mlp_arete2_1, mlp_arete2_2, mlp_arete2_3,
								  mlp_arete3_1,
								  mlp_perceptrons, mlp_point1, mlp_point2, mlp_point3, 
								  mlp_input_label, mlp_dotprod1_label, mlp_dotprod2_label,
								  mlp_activ1_label, mlp_conc_label, mlp_activ2_label,
								  mlp_class_label],
						 #axes = [mlp_ax_x, mlp_ax_y],
						 animation_duration = 1000)

	####################################################
	mlp_activ2_label.x = [4.5]
	mlp_activ2_label.y = [2]
	mlp_activ2_label.default_opacities = [0.01]
	mlp_activ2_label.opacity = [0.01]

	mlp_dotprod2_label.x = [4]
	mlp_dotprod2_label.y = [2]
	mlp_dotprod2_label.default_opacities = [0.01]
	mlp_dotprod2_label.opacity = [0.01]

	mlp_conc_label.x = [2.4]
	mlp_conc_label.y = [3]
	mlp_conc_label.default_opacities = [0.01]
	mlp_conc_label.opacity = [0.01]

	mlp_activ1_label.x = [1.6]
	mlp_activ1_label.y = [5]
	mlp_activ1_label.default_opacities = [0.01]
	mlp_activ1_label.opacity = [0.01]

	mlp_input_label.x = [-0.2]
	mlp_input_label.y = [2]
	mlp_input_label.default_opacities = [0.01]
	mlp_input_label.opacity = [0.01]

	mlp_dotprod1_label.x = [0.5]
	mlp_dotprod1_label.y = [4.2]
	mlp_dotprod1_label.default_opacities = [0.01]
	mlp_dotprod1_label.opacity = [0.01]

	mlp_class_label.x = [5]
	mlp_class_label.y = [2]
	mlp_class_label.default_opacities = [0.01]
	mlp_class_label.opacity = [0.01]

	display(mlp_fig)

	@widgets.interact(frame = widgets.Play(value = 0, interval = 2000, min = 0, max = 9, step=1, disabled=False))

	def mlp_anim(frame):
		if(frame == 0):
			mlp_class_label.default_opacities = [0.01]
			mlp_class_label.opacity = [0.01]
			
			mlp_input_label.default_opacities = [0.9]
			mlp_input_label.opacity = [0.9]
			
			mlp_arete1_1.x = np.zeros(80)
			mlp_arete1_1.y = np.zeros(80)
			
			mlp_arete1_2.x = np.zeros(80)
			mlp_arete1_2.y = np.zeros(80)
			
			mlp_arete1_3.x = np.zeros(80)
			mlp_arete1_3.y = np.zeros(80)
			
			mlp_arete2_1.x = np.zeros(100) + 2
			mlp_arete2_1.y = np.zeros(100) + 3.5

			mlp_arete2_2.x = np.zeros(100) + 2
			mlp_arete2_2.y = np.zeros(100)

			mlp_arete2_3.x = np.zeros(100) + 2
			mlp_arete2_3.y = np.zeros(100) - 3.5
			
			mlp_arete3_1.x = np.zeros(100) + 5
			mlp_arete3_1.y = np.zeros(100)
			
			mlp_point1.x, mlp_point1.y = [0], [0]
			mlp_point2.x, mlp_point2.y = [0], [0]
			mlp_point3.x, mlp_point3.y = [0], [0]
			
			mlp_point1.names = ['x']
			mlp_point2.names = ['x']
			mlp_point3.names = ['x']
			
			mlp_point1.colors = ['steelblue']
			mlp_point2.colors = ['steelblue']
			mlp_point3.colors = ['steelblue']
			return
		if(frame == 1):
			mlp_input_label.default_opacities = [0.01]
			mlp_input_label.opacity = [0.01]
			
			mlp_dotprod1_label.default_opacities = [0.9]
			mlp_dotprod1_label.opacity = [0.9]
			
			mlp_arete1_1.x = np.append(np.linspace(0, 1, 50), np.zeros(30) + 1)
			mlp_arete1_1.y = np.append(np.linspace(0, 1.75, 50), np.zeros(30) + 1.75)
			
			
			mlp_arete1_2.x = np.append(np.linspace(0, 1, 50), np.zeros(30) + 1)
			
			mlp_arete1_3.x = np.append(np.linspace(0, 1, 50), np.zeros(30) + 1)
			mlp_arete1_3.y = np.append(np.linspace(0, -1.75, 50), np.zeros(30) - 1.75)
			
			mlp_point1.x, mlp_point1.y = [1], [1.75]
			mlp_point2.x, mlp_point2.y = [1], [0]
			mlp_point3.x, mlp_point3.y = [1], [-1.75]
			
			mlp_point1.names = ['<w1, x> + b1']
			mlp_point2.names = ['<w2, x> + b2']
			mlp_point3.names = ['<w3, x> + b3']
			return
		if(frame == 2):
			
			mlp_dotprod1_label.default_opacities = [0.01]
			mlp_dotprod1_label.opacity = [0.01]
			
			mlp_activ1_label.default_opacities = [0.9]
			mlp_activ1_label.opacity = [0.9]
			
			mlp_arete1_1.x = np.append(np.linspace(0, 1, 50), np.linspace(1,2, 30))
			mlp_arete1_1.y = np.append(np.linspace(0, 1.75, 50), np.linspace(1.75, 3.5, 30))
			
			mlp_arete1_2.x = np.append(np.linspace(0, 1, 50), np.linspace(1,2, 30))
			
			
			mlp_arete1_3.x = np.append(np.linspace(0, 1, 50), np.linspace(1,2, 30))
			mlp_arete1_3.y = np.append(np.linspace(0, -1.75, 50), np.linspace(-1.75, -3.5, 30))
			
			mlp_point1.x, mlp_point1.y = [2], [3.5]
			mlp_point2.x, mlp_point2.y = [2], [0]
			mlp_point3.x, mlp_point3.y = [2], [-3.5]
			
			mlp_point1.names = [' ']
			mlp_point2.names = [' ']
			mlp_point3.names = [' ']
			return
		if(frame == 3):
			mlp_point1.colors = ['red']
			mlp_point2.colors = ['red']
			mlp_point3.colors = ['red']
			
			
			return
		if(frame == 4):
			mlp_activ1_label.default_opacities = [0.01]
			mlp_activ1_label.opacity = [0.01]
			
			mlp_conc_label.default_opacities = [0.9]
			mlp_conc_label.opacity = [0.9]
			
			mlp_point1.names = ['O1']
			mlp_point2.names = ['O2']
			mlp_point3.names = ['O3']

			mlp_point1.x, mlp_point1.y = [3], [1]
			mlp_point2.x, mlp_point2.y = [3], [0]
			mlp_point3.x, mlp_point3.y = [3], [-1]
			return
		if(frame == 5):
			mlp_point1.x, mlp_point1.y = [4], [0]
			mlp_point2.x, mlp_point2.y = [4], [0]
			mlp_point3.x, mlp_point3.y = [4], [0]
			
			mlp_arete2_1.x = np.append(np.linspace(2, 4, 50), np.zeros(50) + 4)
			mlp_arete2_1.y = np.append(np.linspace(3.5, 0, 50), np.zeros(50))
			
			
			mlp_arete2_2.x = np.append(np.linspace(2, 4, 50), np.zeros(50) + 4)
			
			mlp_arete2_3.x = np.append(np.linspace(2, 4, 50), np.zeros(50) + 4)
			mlp_arete2_3.y = np.append(np.linspace(-3.5, 0, 50), np.zeros(50))
			
			mlp_point1.names = ['O']
			mlp_point2.names = ['O']
			mlp_point3.names = ['O']
			return
		if(frame == 6):
			mlp_conc_label.default_opacities = [0.01]
			mlp_conc_label.opacity = [0.01]
			
			mlp_dotprod2_label.default_opacities = [0.9]
			mlp_dotprod2_label.opacity = [0.9]
			
			mlp_point1.x, mlp_point1.y = [4.5], [0]
			mlp_point2.x, mlp_point2.y = [4.5], [0]
			mlp_point3.x, mlp_point3.y = [4.5], [0]
			
			mlp_arete2_1.x = np.append(np.linspace(2, 4, 50), np.linspace(4, 5, 50))
			mlp_arete2_1.y = np.append(np.linspace(3.5, 0, 50), np.zeros(50))
			
			
			mlp_arete2_2.x = np.append(np.linspace(2, 4, 50), np.linspace(4, 5, 50))
			
			mlp_arete2_3.x = np.append(np.linspace(2, 4, 50), np.linspace(4, 5, 50))
			mlp_arete2_3.y = np.append(np.linspace(-3.5, 0, 50), np.zeros(50))

			mlp_point1.names = ['<w4, O> + b4']
			mlp_point2.names = ['<w4, O> + b4']
			mlp_point3.names = ['<w4, O> + b4']
			return
		if(frame == 7):
			mlp_dotprod2_label.default_opacities = [0.01]
			mlp_dotprod2_label.opacity = [0.01]
			
			mlp_activ2_label.default_opacities = [0.9]
			mlp_activ2_label.opacity = [0.9]
			
			mlp_point1.x, mlp_point1.y = [5], [0]
			mlp_point2.x, mlp_point2.y = [5], [0]
			mlp_point3.x, mlp_point3.y = [5], [0]
			
			mlp_point1.names = [' ']
			mlp_point2.names = [' ']
			mlp_point3.names = [' ']
			return
		if(frame == 8):
			
			mlp_point1.colors = ['green']
			mlp_point2.colors = ['green']
			mlp_point3.colors = ['green']
			return        
		if(frame == 9):
			mlp_activ2_label.default_opacities = [0.01]
			mlp_activ2_label.opacity = [0.01]
			
			mlp_class_label.default_opacities = [0.9]
			mlp_class_label.opacity = [0.9]
			
			if(np.random.normal(0,1,1) > 0):
				mlp_point1.x, mlp_point1.y = [5.5], [0]
				mlp_point2.x, mlp_point2.y = [5.5], [0]
				mlp_point3.x, mlp_point3.y = [5.5], [0]
				
				mlp_arete3_1.x = np.linspace(5, 5.5, 50)
				mlp_arete3_1.y = np.zeros(50)
				
				mlp_point1.names = ['1']
				mlp_point2.names = ['1']
				mlp_point3.names = ['1']
			else:
				mlp_point1.x, mlp_point1.y = [5.5], [0]
				mlp_point2.x, mlp_point2.y = [5.5], [0]
				mlp_point3.x, mlp_point3.y = [5.5], [0]
				
				mlp_arete3_1.x = np.linspace(5, 5.5, 50)
				mlp_arete3_1.y = np.zeros(50)
				
				mlp_point1.names = ['0']
				mlp_point2.names = ['0']
				mlp_point3.names = ['0']




def show_dataset_moon():

	# Data
	X, y = make_moons(n_samples = 100)

	X_0 = X[y == 0]
	X_1 = X[y == 1]

	# Scales

	moons_x_sc= plt.LinearScale(min = -1, max = 2)

	moons_y_sc= plt.LinearScale(min = -1, max = 1)

	# Axes

	moons_ax_x = plt.Axis(scale = moons_x_sc,
						  grid_lines = 'solid',
						  label ='x')

	moons_ax_y = plt.Axis(scale = moons_y_sc,
						  grid_lines = 'solid',
						  orientation = 'vertical',
						  label = 'y')

	# Scatter plots

	moons_x0 = plt.Scatter(x = X_0[:,0], y = X_0[:,1],
						   scales = {'x': moons_x_sc, 'y': moons_y_sc},
						   colors = ['blue'])

	moons_x1 = plt.Scatter(x = X_1[:,0], y = X_1[:,1],
						   scales = {'x': moons_x_sc, 'y': moons_y_sc},
						   colors = ['red'])

	# Figure

	moons_fig = plt.Figure(marks = [moons_x0, moons_x1],
						   axes = [moons_ax_x, moons_ax_y],
						   title = "Moons Dataset")

	moons_fig.layout.height = '400px'
	moons_fig.layout.width = '500px'

	display(moons_fig)
