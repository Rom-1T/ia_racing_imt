import ipywidgets as widgets
import bqplot as bq
import bqplot.pyplot as plt
from IPython.display import display
import numpy as np
import cv2
from scipy import ndimage
import matplotlib as mpl
from matplotlib import image as im_matplotlib
from io import BytesIO
from IPython.display import Markdown

def create_square(size = 1, starting_x = 0, starting_y = 0):
				x = np.array([starting_x, starting_x + size, starting_x + size, starting_x, starting_x])
				y = np.array([starting_y, starting_y, starting_y + size, starting_y + size, starting_y])
				return x,y

def show_dense():
	# Dense Layer
	if(True):
		# Functions
		if(True):

			def create_square(size = 1, starting_x = 0, starting_y = 0):
				x = np.array([starting_x, starting_x + size, starting_x + size, starting_x, starting_x])
				y = np.array([starting_y, starting_y, starting_y + size, starting_y + size, starting_y])
				return x,y

		# Scales and Axes
		if(True):
			dense_x_sc = plt.LinearScale(min = 0, max = 40)
			dense_y_sc = plt.LinearScale(min = 0, max = 20)

			dense_ax_x = bq.Axis(scale= dense_x_sc)
			dense_ax_y = bq.Axis(scale= dense_y_sc, orientation = 'vertical')

		# Labels
		if(True):
			prev_layer_label = plt.Label(text = ['Previous Layer'], x = [-1], y = [20.1],
										 scales = {'x': dense_x_sc, 'y' : dense_y_sc},
										 font_weight = 'bolder',
										 update_on_move = True,
										 colors = ["steelblue"])

			dense_layer_label = plt.Label(text = ['Dense Layer'], x = [16.9], y = [20],
										 scales = {'x': dense_x_sc, 'y' : dense_y_sc},
										 font_weight = 'bolder',
										 update_on_move = True,
										 colors = ["red"])

			next_layer_label = plt.Label(text = ['Another Dense Layer'], x = [30], y = [20],
										 scales = {'x': dense_x_sc, 'y' : dense_y_sc},
										 font_weight = 'bolder',
										 update_on_move = True,
										 colors = ["green"])

			dense_inputs_label = plt.Label(text = ['Collection of inputs'],
									 x = [15], y = [5],
									 scales = {'x': dense_x_sc, 'y': dense_y_sc},
									 #default_size = 20,
									 #default_opacities = [0.9],
									 font_weight = 'bolder',
									 #update_on_move = True,
									 colors = ['steelblue'])  

			dense_labels = {
				'inputs' : plt.Label(text = ['Step 0: Collection of inputs'],
									 x = [8.5], y = [2],
									 scales = {'x': dense_x_sc, 'y': dense_y_sc},
									 default_size = 30,
									 default_opacities = [0.9],
									 font_weight = 'bolder',
									 update_on_move = True,
									 colors = ['steelblue']),  

				'dotproduct1' : plt.Label(text = ['Step 1: Dot Products'],
										  x = [10.8], y = [2],
										  scales = {'x': dense_x_sc, 'y': dense_y_sc},
										  default_size = 30,
										  default_opacities = [0.9],
										  font_weight = 'bolder',
										  update_on_move = True,
										  colors = ['steelblue']),

				'activation1' : plt.Label(text = ['Step 2: Activations'],
										  x = [11.8], y = [2],
										  scales = {'x': dense_x_sc, 'y': dense_y_sc},
										  default_size = 30,
										  default_opacities = [0.9],
										  font_weight = 'bolder',
										  update_on_move = True,
										  colors = ['steelblue']),

				'dotproduct2' : plt.Label(text = ['Step 3: Dot Products'],
										  x = [10.8], y = [2],
										  scales = {'x': dense_x_sc, 'y': dense_y_sc},
										  default_size = 30,
										  default_opacities = [0.9],
										  font_weight = 'bolder',
										  update_on_move = True,
										  colors = ['steelblue']),

				'activation2' : plt.Label(text = ['Step 4: Activations'],
										  x = [11.8], y = [2],
										  scales = {'x': dense_x_sc, 'y': dense_y_sc},
										  default_size = 30,
										  default_opacities = [0.9],
										  font_weight = 'bolder',
										  update_on_move = True,
										  colors = ['steelblue']),
							}
			for x in dense_labels.keys():
				dense_labels[x].default_opacities = [0]
				dense_labels[x].opacity = [0]

		# Inputs
		if(True):
			size = 1
			square_x, square_y = create_square(size = size)
			# Inputs 1
			if(True):
				dense_input_1_1 = bq.Lines(x = square_x + 1.5, y = square_y + 17.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				dense_input_1_2 = bq.Lines(x = square_x + 1.5, y = square_y + 13.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				dense_input_1_3 = bq.Lines(x = square_x + 1.5, y = square_y + 9.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				dense_input_1_4 = bq.Lines(x = square_x + 1.5, y = square_y + 5.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				dense_input_1_5 = bq.Lines(x = square_x + 1.5, y = square_y + 1.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

			# Inputs 2
			if(True):
				dense_input_2_1 = bq.Lines(x = square_x + 1.5, y = square_y + 17.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				dense_input_2_2 = bq.Lines(x = square_x + 1.5, y = square_y + 13.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				dense_input_2_3 = bq.Lines(x = square_x + 1.5, y = square_y + 9.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				dense_input_2_4 = bq.Lines(x = square_x + 1.5, y = square_y + 5.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				dense_input_2_5 = bq.Lines(x = square_x + 1.5, y = square_y + 1.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

			# Inputs 3
			if(True):
				dense_input_3_1 = bq.Lines(x = square_x + 1.5, y = square_y + 17.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				dense_input_3_2 = bq.Lines(x = square_x + 1.5, y = square_y + 13.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				dense_input_3_3 = bq.Lines(x = square_x + 1.5, y = square_y + 9.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				dense_input_3_4 = bq.Lines(x = square_x + 1.5, y = square_y + 5.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				dense_input_3_5 = bq.Lines(x = square_x + 1.5, y = square_y + 1.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

		# Weights
		if(True):
			size = 1
			square_x, square_y = create_square(size = size)
			# Weights 1
			if(True):
				dense_weight_1_1 = bq.Lines(x = square_x + 19.5, y = square_y + 13.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				dense_weight_1_2 = bq.Lines(x = square_x + 19.5, y = square_y + 13.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				dense_weight_1_3 = bq.Lines(x = square_x + 19.5, y = square_y + 13.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				dense_weight_1_4= bq.Lines(x = square_x + 19.5, y = square_y + 13.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				dense_weight_1_5 = bq.Lines(x = square_x + 19.5, y = square_y + 13.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

			# Weights 2
			if(True):
				dense_weight_2_1 = bq.Lines(x = square_x + 19.5, y = square_y + 9.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				dense_weight_2_2 = bq.Lines(x = square_x + 19.5, y = square_y + 9.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				dense_weight_2_3 = bq.Lines(x = square_x + 19.5, y = square_y + 9.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				dense_weight_2_4= bq.Lines(x = square_x + 19.5, y = square_y + 9.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				dense_weight_2_5 = bq.Lines(x = square_x + 19.5, y = square_y + 9.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

			# Weights 3
			if(True):
				dense_weight_3_1 = bq.Lines(x = square_x + 19.5, y = square_y + 5.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				dense_weight_3_2 = bq.Lines(x = square_x + 19.5, y = square_y + 5.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				dense_weight_3_3 = bq.Lines(x = square_x + 19.5, y = square_y + 5.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				dense_weight_3_4= bq.Lines(x = square_x + 19.5, y = square_y + 5.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				dense_weight_3_5 = bq.Lines(x = square_x + 19.5, y = square_y + 5.5,
										scales = {'x': dense_x_sc, 'y': dense_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

			# Weights 4
			if(True):
				dense_weight_4_1 = bq.Lines(x = square_x + 37.5, y = square_y + 17,
											scales = {'x': dense_x_sc, 'y': dense_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				dense_weight_4_2 = bq.Lines(x = square_x + 37.5, y = square_y + 17,
											scales = {'x': dense_x_sc, 'y': dense_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				dense_weight_4_3 = bq.Lines(x = square_x + 37.5, y = square_y + 17,
											scales = {'x': dense_x_sc, 'y': dense_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

			# Weights 5
			if(True):
				dense_weight_5_1 = bq.Lines(x = square_x + 37.5, y = square_y + 12,
											scales = {'x': dense_x_sc, 'y': dense_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				dense_weight_5_2 = bq.Lines(x = square_x + 37.5, y = square_y + 12,
											scales = {'x': dense_x_sc, 'y': dense_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				dense_weight_5_3 = bq.Lines(x = square_x + 37.5, y = square_y + 12,
											scales = {'x': dense_x_sc, 'y': dense_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

			# Weights 6
			if(True):
				dense_weight_6_1 = bq.Lines(x = square_x + 37.5, y = square_y + 7,
											scales = {'x': dense_x_sc, 'y': dense_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				dense_weight_6_2 = bq.Lines(x = square_x + 37.5, y = square_y + 7,
											scales = {'x': dense_x_sc, 'y': dense_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				dense_weight_6_3 = bq.Lines(x = square_x + 37.5, y = square_y + 7,
											scales = {'x': dense_x_sc, 'y': dense_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

			# Weights 7
			if(True):
				dense_weight_7_1 = bq.Lines(x = square_x + 37.5, y = square_y + 2,
											scales = {'x': dense_x_sc, 'y': dense_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				dense_weight_7_2 = bq.Lines(x = square_x + 37.5, y = square_y + 2,
											scales = {'x': dense_x_sc, 'y': dense_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				dense_weight_7_3 = bq.Lines(x = square_x + 37.5, y = square_y + 2,
											scales = {'x': dense_x_sc, 'y': dense_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

		# Scatterplot of Perceptrons
		if(True):
			dense_mlp_perceptrons = plt.Scatter(x = [2, 2, 2, 2, 2,
											   20, 20, 20,
											   38, 38, 38, 38],
										  y = [2, 6, 10, 14, 18,
											   6, 10, 14,
											   2.5, 7.5, 12.5, 17.5],
										  size = [900, 900 , 900, 900, 900],
										  default_size = 900,
										  colors = ['steelblue','steelblue','steelblue','steelblue','steelblue',
													'red', 'red','red',
													'green', 'green', 'green', 'green'],
										  scales = {'x': dense_x_sc, 'y': dense_y_sc})

		# Figure
		if(True):
			dense_fig = bq.Figure(marks = [dense_input_1_1, dense_input_1_2, dense_input_1_3,
										   dense_input_1_5, dense_input_1_4,
										   dense_input_2_1, dense_input_2_2, dense_input_2_3,
										   dense_input_2_5, dense_input_2_4,
										   dense_input_3_1, dense_input_3_2, dense_input_3_3,
										   dense_input_3_5, dense_input_3_4,
										   dense_weight_1_1, dense_weight_1_2, dense_weight_1_3,
										   dense_weight_1_4, dense_weight_1_5,
										   dense_weight_2_1, dense_weight_2_2, dense_weight_2_3,
										   dense_weight_2_4, dense_weight_2_5,
										   dense_weight_3_1, dense_weight_3_2, dense_weight_3_3,
										   dense_weight_3_4, dense_weight_3_5,
										   dense_weight_4_1, dense_weight_4_2, dense_weight_4_3,
										   dense_weight_5_1, dense_weight_5_2, dense_weight_5_3,
										   dense_weight_6_1, dense_weight_6_2, dense_weight_6_3,
										   dense_weight_7_1, dense_weight_7_2, dense_weight_7_3,
										   dense_mlp_perceptrons,
										   prev_layer_label, dense_layer_label, next_layer_label,
										   dense_labels['inputs'], dense_labels['dotproduct1'],
										   dense_labels['activation1'], dense_labels['dotproduct2'],
										   dense_labels['activation2']],
								  #axes = [dense_ax_x, dense_ax_y],
								  animation_duration = 1000,
								  title = 'Dense Layer - Forward Pipeline',
								  background_style = {'fill': 'white'})

			dense_fig.layout.height = '475px'
			dense_fig.layout.width = '800px'
		
		# Animation
		if(True):
			def dense_anim(frame):
				if(frame == 0):
					# Labels
					if(True):
						for x in dense_labels.keys():
							dense_labels[x].default_opacities = [0]
							dense_labels[x].opacity = [0]

					# Initialisation
					if(True):
						square_x, square_y = create_square(size = 1)

						# Inputs 1
						if(True):
							dense_input_1_1.x = square_x + 1.5
							dense_input_1_1.y = square_y + 17.5

							dense_input_1_2.x = square_x + 1.5
							dense_input_1_2.y = square_y + 13.5

							dense_input_1_3.x = square_x + 1.5
							dense_input_1_3.y = square_y + 9.5

							dense_input_1_4.x = square_x + 1.5
							dense_input_1_4.y = square_y + 5.5

							dense_input_1_5.x = square_x + 1.5
							dense_input_1_5.y = square_y + 1.5

						# Inputs 2
						if(True):
							dense_input_2_1.x = square_x + 1.5
							dense_input_2_1.y = square_y + 17.5

							dense_input_2_2.x = square_x + 1.5
							dense_input_2_2.y = square_y + 13.5

							dense_input_2_3.x = square_x + 1.5
							dense_input_2_3.y = square_y + 9.5

							dense_input_2_4.x = square_x + 1.5
							dense_input_2_4.y = square_y + 5.5

							dense_input_2_5.x = square_x + 1.5
							dense_input_2_5.y = square_y + 1.5

						# Inputs 3
						if(True):
							dense_input_3_1.x = square_x + 1.5
							dense_input_3_1.y = square_y + 17.5

							dense_input_3_2.x = square_x + 1.5
							dense_input_3_2.y = square_y + 13.5

							dense_input_3_3.x = square_x + 1.5
							dense_input_3_3.y = square_y + 9.5

							dense_input_3_4.x = square_x + 1.5
							dense_input_3_4.y = square_y + 5.5

							dense_input_3_5.x = square_x + 1.5
							dense_input_3_5.y = square_y + 1.5

						# Weights 1   
						if(True):
							dense_weight_1_1.x = square_x + 19.5
							dense_weight_1_1.y = square_y  + 13.5

							dense_weight_1_2.x = square_x + 19.5
							dense_weight_1_2.y = square_y  + 13.5

							dense_weight_1_3.x = square_x + 19.5
							dense_weight_1_3.y = square_y  + 13.5

							dense_weight_1_4.x = square_x + 19.5
							dense_weight_1_4.y = square_y  + 13.5

							dense_weight_1_5.x = square_x + 19.5
							dense_weight_1_5.y = square_y  + 13.5

						# Weights 2  
						if(True):
							dense_weight_2_1.x = square_x + 19.5
							dense_weight_2_1.y = square_y  + 9.5

							dense_weight_2_2.x = square_x + 19.5
							dense_weight_2_2.y = square_y  + 9.5

							dense_weight_2_3.x = square_x + 19.5
							dense_weight_2_3.y = square_y  + 9.5

							dense_weight_2_4.x = square_x + 19.5
							dense_weight_2_4.y = square_y  + 9.5

							dense_weight_2_5.x = square_x + 19.5
							dense_weight_2_5.y = square_y  + 9.5

						# Weights 3   
						if(True):
							dense_weight_3_1.x = square_x + 19.5
							dense_weight_3_1.y = square_y  + 5.5

							dense_weight_3_2.x = square_x + 19.5
							dense_weight_3_2.y = square_y  + 5.5

							dense_weight_3_3.x = square_x + 19.5
							dense_weight_3_3.y = square_y  + 5.5

							dense_weight_3_4.x = square_x + 19.5
							dense_weight_3_4.y = square_y  + 5.5

							dense_weight_3_5.x = square_x + 19.5
							dense_weight_3_5.y = square_y  + 5.5

						# Weights 4   
						if(True):
							dense_weight_4_1.x = square_x + 37.5
							dense_weight_4_1.y = square_y  + 17

							dense_weight_4_2.x = square_x + 37.5
							dense_weight_4_2.y = square_y  + 17

							dense_weight_4_3.x = square_x + 37.5
							dense_weight_4_3.y = square_y  + 17

						# Weights 5   
						if(True):
							dense_weight_5_1.x = square_x + 37.5
							dense_weight_5_1.y = square_y  + 12

							dense_weight_5_2.x = square_x + 37.5
							dense_weight_5_2.y = square_y  + 12

							dense_weight_5_3.x = square_x + 37.5
							dense_weight_5_3.y = square_y  + 12

						# Weights 6   
						if(True):
							dense_weight_6_1.x = square_x + 37.5
							dense_weight_6_1.y = square_y  + 7

							dense_weight_6_2.x = square_x + 37.5
							dense_weight_6_2.y = square_y  + 7

							dense_weight_6_3.x = square_x + 37.5
							dense_weight_6_3.y = square_y  + 7

						# Weights 7   
						if(True):
							dense_weight_7_1.x = square_x + 37.5
							dense_weight_7_1.y = square_y  + 2

							dense_weight_7_2.x = square_x + 37.5
							dense_weight_7_2.y = square_y  + 2

							dense_weight_7_3.x = square_x + 37.5
							dense_weight_7_3.y = square_y  + 2


						dense_fig.animation_duration = 1000
					return

				if(frame == 1):
					if(True):
						square_x, square_y = create_square(size = 1)
						# Inputs 1
						if(True):
							dense_input_1_1.x = square_x + 6
							dense_input_1_1.y = square_y + 17.5

							dense_input_1_2.x = square_x + 6
							dense_input_1_2.y = square_y + 13.5

							dense_input_1_3.x = square_x + 6
							dense_input_1_3.y = square_y + 9.5

							dense_input_1_4.x = square_x + 6
							dense_input_1_4.y = square_y + 5.5

							dense_input_1_5.x = square_x + 6
							dense_input_1_5.y = square_y + 1.5

						# Inputs 2
						if(True):
							dense_input_2_1.x = square_x + 6
							dense_input_2_1.y = square_y + 17.5

							dense_input_2_2.x = square_x + 6
							dense_input_2_2.y = square_y + 13.5

							dense_input_2_3.x = square_x + 6
							dense_input_2_3.y = square_y + 9.5

							dense_input_2_4.x = square_x + 6
							dense_input_2_4.y = square_y + 5.5

							dense_input_2_5.x = square_x + 6
							dense_input_2_5.y = square_y + 1.5

						# Inputs 3
						if(True):
							dense_input_3_1.x = square_x + 6
							dense_input_3_1.y = square_y + 17.5

							dense_input_3_2.x = square_x + 6
							dense_input_3_2.y = square_y + 13.5

							dense_input_3_3.x = square_x + 6
							dense_input_3_3.y = square_y + 9.5

							dense_input_3_4.x = square_x + 6
							dense_input_3_4.y = square_y + 5.5

							dense_input_3_5.x = square_x + 6
							dense_input_3_5.y = square_y + 1.5

						# Weights 1   
						if(True):
							dense_weight_1_1.x = square_x + 19.5
							dense_weight_1_1.y = square_y  + 13.5

							dense_weight_1_2.x = square_x + 19.5
							dense_weight_1_2.y = square_y  + 13.5

							dense_weight_1_3.x = square_x + 19.5
							dense_weight_1_3.y = square_y  + 13.5

							dense_weight_1_4.x = square_x + 19.5
							dense_weight_1_4.y = square_y  + 13.5

							dense_weight_1_5.x = square_x + 19.5
							dense_weight_1_5.y = square_y  + 13.5

						# Weights 2  
						if(True):
							dense_weight_2_1.x = square_x + 19.5
							dense_weight_2_1.y = square_y  + 9.5

							dense_weight_2_2.x = square_x + 19.5
							dense_weight_2_2.y = square_y  + 9.5

							dense_weight_2_3.x = square_x + 19.5
							dense_weight_2_3.y = square_y  + 9.5

							dense_weight_2_4.x = square_x + 19.5
							dense_weight_2_4.y = square_y  + 9.5

							dense_weight_2_5.x = square_x + 19.5
							dense_weight_2_5.y = square_y  + 9.5

						# Weights 3   
						if(True):
							dense_weight_3_1.x = square_x + 19.5
							dense_weight_3_1.y = square_y  + 5.5

							dense_weight_3_2.x = square_x + 19.5
							dense_weight_3_2.y = square_y  + 5.5

							dense_weight_3_3.x = square_x + 19.5
							dense_weight_3_3.y = square_y  + 5.5

							dense_weight_3_4.x = square_x + 19.5
							dense_weight_3_4.y = square_y  + 5.5

							dense_weight_3_5.x = square_x + 19.5
							dense_weight_3_5.y = square_y  + 5.5

						# Weights 4   
						if(True):
							dense_weight_4_1.x = square_x + 37.5
							dense_weight_4_1.y = square_y  + 17

							dense_weight_4_2.x = square_x + 37.5
							dense_weight_4_2.y = square_y  + 17

							dense_weight_4_3.x = square_x + 37.5
							dense_weight_4_3.y = square_y  + 17

						# Weights 5   
						if(True):
							dense_weight_5_1.x = square_x + 37.5
							dense_weight_5_1.y = square_y  + 12

							dense_weight_5_2.x = square_x + 37.5
							dense_weight_5_2.y = square_y  + 12

							dense_weight_5_3.x = square_x + 37.5
							dense_weight_5_3.y = square_y  + 12

						# Weights 6   
						if(True):
							dense_weight_6_1.x = square_x + 37.5
							dense_weight_6_1.y = square_y  + 7

							dense_weight_6_2.x = square_x + 37.5
							dense_weight_6_2.y = square_y  + 7

							dense_weight_6_3.x = square_x + 37.5
							dense_weight_6_3.y = square_y  + 7

						# Weights 7   
						if(True):
							dense_weight_7_1.x = square_x + 37.5
							dense_weight_7_1.y = square_y  + 2

							dense_weight_7_2.x = square_x + 37.5
							dense_weight_7_2.y = square_y  + 2

							dense_weight_7_3.x = square_x + 37.5
							dense_weight_7_3.y = square_y  + 2

						# Label
						if(True):
							dense_labels['dotproduct1'].default_opacities = [0]
							dense_labels['dotproduct1'].opacity = [0]

							dense_labels['inputs'].default_opacities = [0.9]
							dense_labels['inputs'].opacity = [0.9]

					return

				if(frame == 2):
					if(True):
						square_x, square_y = create_square(size = 1)

						dense_input_1_1.x = square_x + 6
						dense_input_1_1.y = square_y + 13.5

						dense_input_1_2.x = square_x + 8.5
						dense_input_1_2.y = square_y + 13.5

						dense_input_1_3.x = square_x + 11
						dense_input_1_3.y = square_y + 13.5

						dense_input_1_4.x = square_x + 13.5
						dense_input_1_4.y = square_y + 13.5

						dense_input_1_5.x = square_x + 16
						dense_input_1_5.y = square_y + 13.5

						dense_weight_1_1.x = square_x + 7
						dense_weight_1_1.y = square_y + 13.5

						dense_weight_1_2.x = square_x + 9.5
						dense_weight_1_2.y = square_y + 13.5

						dense_weight_1_3.x = square_x + 12
						dense_weight_1_3.y = square_y + 13.5

						dense_weight_1_4.x = square_x + 14.5
						dense_weight_1_4.y = square_y + 13.5

						dense_weight_1_5.x = square_x + 17
						dense_weight_1_5.y = square_y + 13.5
					if(True):
						square_x, square_y = create_square(size = 1)

						dense_input_2_1.x = square_x + 6
						dense_input_2_1.y = square_y + 9.5

						dense_input_2_2.x = square_x + 8.5
						dense_input_2_2.y = square_y + 9.5

						dense_input_2_3.x = square_x + 11
						dense_input_2_3.y = square_y + 9.5

						dense_input_2_4.x = square_x + 13.5
						dense_input_2_4.y = square_y + 9.5

						dense_input_2_5.x = square_x + 16
						dense_input_2_5.y = square_y + 9.5

						dense_weight_2_1.x = square_x + 7
						dense_weight_2_1.y = square_y + 9.5

						dense_weight_2_2.x = square_x + 9.5
						dense_weight_2_2.y = square_y + 9.5

						dense_weight_2_3.x = square_x + 12
						dense_weight_2_3.y = square_y + 9.5

						dense_weight_2_4.x = square_x + 14.5
						dense_weight_2_4.y = square_y + 9.5

						dense_weight_2_5.x = square_x + 17
						dense_weight_2_5.y = square_y + 9.5
					if(True):
						square_x, square_y = create_square(size = 1)

						dense_input_3_1.x = square_x + 6
						dense_input_3_1.y = square_y + 5.5

						dense_input_3_2.x = square_x + 8.5
						dense_input_3_2.y = square_y + 5.5

						dense_input_3_3.x = square_x + 11
						dense_input_3_3.y = square_y + 5.5

						dense_input_3_4.x = square_x + 13.5
						dense_input_3_4.y = square_y + 5.5

						dense_input_3_5.x = square_x + 16
						dense_input_3_5.y = square_y + 5.5

						dense_weight_3_1.x = square_x + 7
						dense_weight_3_1.y = square_y + 5.5

						dense_weight_3_2.x = square_x + 9.5
						dense_weight_3_2.y = square_y + 5.5

						dense_weight_3_3.x = square_x + 12
						dense_weight_3_3.y = square_y + 5.5

						dense_weight_3_4.x = square_x + 14.5
						dense_weight_3_4.y = square_y + 5.5

						dense_weight_3_5.x = square_x + 17
						dense_weight_3_5.y = square_y + 5.5   

					# Labels 
					if(True):
						dense_labels['inputs'].default_opacities = [0]
						dense_labels['inputs'].opacity = [0]
						dense_labels['activation1'].default_opacities = [0]
						dense_labels['activation1'].opacity = [0]

						dense_labels['dotproduct1'].default_opacities = [0.9]
						dense_labels['dotproduct1'].opacity = [0.9]

					return

				if(frame == 3):
					if(True):
						square_x, square_y = create_square(size = 1)
						# Inputs
						if(True):
							dense_input_1_1.x = square_x + 19.5
							dense_input_1_2.x = square_x + 19.5
							dense_input_1_3.x = square_x + 19.5
							dense_input_1_4.x = square_x + 19.5
							dense_input_1_5.x = square_x + 19.5

							dense_input_2_1.x = square_x + 19.5
							dense_input_2_2.x = square_x + 19.5
							dense_input_2_3.x = square_x + 19.5
							dense_input_2_4.x = square_x + 19.5
							dense_input_2_5.x = square_x + 19.5

							dense_input_3_1.x = square_x + 19.5
							dense_input_3_2.x = square_x + 19.5
							dense_input_3_3.x = square_x + 19.5
							dense_input_3_4.x = square_x + 19.5
							dense_input_3_5.x = square_x + 19.5

						# Weights
						if(True):
							square_x, square_y = create_square(size = 1)
							dense_weight_1_1.x = square_x + 22
							dense_weight_1_2.x = square_x + 22
							dense_weight_1_3.x = square_x + 22
							dense_weight_1_4.x = square_x + 22
							dense_weight_1_5.x = square_x + 19.5

							dense_weight_2_1.x = square_x + 22
							dense_weight_2_2.x = square_x + 22
							dense_weight_2_3.x = square_x + 22
							dense_weight_2_4.x = square_x + 22
							dense_weight_2_5.x = square_x + 19.5

							dense_weight_3_1.x = square_x + 22
							dense_weight_3_2.x = square_x + 22
							dense_weight_3_3.x = square_x + 22
							dense_weight_3_4.x = square_x + 22
							dense_weight_3_5.x = square_x + 19.5

							dense_weight_1_1.y = square_y + 13.5
							dense_weight_1_2.y = square_y + 13.5
							dense_weight_1_3.y = square_y + 13.5
							dense_weight_1_4.y = square_y + 13.5
							dense_weight_1_5.y = square_y + 13.5

							dense_weight_2_1.y = square_y + 9.5
							dense_weight_2_2.y = square_y + 9.5
							dense_weight_2_3.y = square_y + 9.5
							dense_weight_2_4.y = square_y + 9.5
							dense_weight_2_5.y = square_y + 9.5

							dense_weight_3_1.y = square_y + 5.5
							dense_weight_3_2.y = square_y + 5.5
							dense_weight_3_3.y = square_y + 5.5
							dense_weight_3_4.y = square_y + 5.5
							dense_weight_3_5.y = square_y + 5.5

							dense_weight_4_1.x = square_x + 37.5
							dense_weight_4_2.x = square_x + 37.5
							dense_weight_4_3.x = square_x + 37.5

							dense_weight_5_1.x = square_x + 37.5
							dense_weight_5_2.x = square_x + 37.5
							dense_weight_5_3.x = square_x + 37.5

							dense_weight_6_1.x = square_x + 37.5
							dense_weight_6_2.x = square_x + 37.5
							dense_weight_6_3.x = square_x + 37.5

							dense_weight_7_1.x = square_x + 37.5
							dense_weight_7_2.x = square_x + 37.5
							dense_weight_7_3.x = square_x + 37.5

						# Labels 
						if(True):
							dense_labels['dotproduct1'].default_opacities = [0]
							dense_labels['dotproduct1'].opacity = [0]
							dense_labels['dotproduct2'].default_opacities = [0]
							dense_labels['dotproduct2'].opacity = [0]  

							dense_labels['activation1'].default_opacities = [0.9]
							dense_labels['activation1'].opacity = [0.9]



					return

				if(frame == 4):
					if(True): 
						square_x, square_y = create_square(size = 1)

						# Weights 1
						if(True):
							dense_weight_1_1.x = square_x + 29
							dense_weight_1_2.x = square_x + 29
							dense_weight_1_3.x = square_x + 29
							dense_weight_1_4.x = square_x + 29

							dense_weight_1_1.y = square_y + 17
							dense_weight_1_2.y = square_y + 12
							dense_weight_1_3.y = square_y + 7
							dense_weight_1_4.y = square_y + 2

						# Weights 2
						if(True):
							dense_weight_2_1.x = square_x + 31.5
							dense_weight_2_2.x = square_x + 31.5
							dense_weight_2_3.x = square_x + 31.5
							dense_weight_2_4.x = square_x + 31.5

							dense_weight_2_1.y = square_y + 17
							dense_weight_2_2.y = square_y + 12
							dense_weight_2_3.y = square_y + 7
							dense_weight_2_4.y = square_y + 2

						# Weights 3
						if(True):
							dense_weight_3_1.y = square_y + 17
							dense_weight_3_2.y = square_y + 12
							dense_weight_3_3.y = square_y + 7
							dense_weight_3_4.y = square_y + 2

							dense_weight_3_1.x = square_x + 34
							dense_weight_3_2.x = square_x + 34
							dense_weight_3_3.x = square_x + 34
							dense_weight_3_4.x = square_x + 34

						# Weights 4
						if(True):
							dense_weight_4_1.x = square_x + 30
							dense_weight_4_2.x = square_x + 32.5
							dense_weight_4_3.x = square_x + 35

						# Weights 5
						if(True):
							dense_weight_5_1.x = square_x + 30
							dense_weight_5_2.x = square_x + 32.5
							dense_weight_5_3.x = square_x + 35

						# Weights 6
						if(True):
							dense_weight_6_1.x = square_x + 30
							dense_weight_6_2.x = square_x + 32.5
							dense_weight_6_3.x = square_x + 35

						# Weights 7
						if(True):
							dense_weight_7_1.x = square_x + 30
							dense_weight_7_2.x = square_x + 32.5
							dense_weight_7_3.x = square_x + 35
						# Labels 
						if(True):
							dense_labels['activation1'].default_opacities = [0]
							dense_labels['activation1'].opacity = [0]
							dense_labels['activation2'].default_opacities = [0]
							dense_labels['activation2'].opacity = [0]

							dense_labels['dotproduct2'].default_opacities = [0.9]
							dense_labels['dotproduct2'].opacity = [0.9]       

					return

				if(frame == 5):
					if(True): 
						square_x, square_y = create_square(size = 1)
						# Weights 1
						if(True):
							dense_weight_1_1.x = square_x + 37.5
							dense_weight_1_2.x = square_x + 37.5
							dense_weight_1_3.x = square_x + 37.5
							dense_weight_1_4.x = square_x + 37.5

						# Weights 2
						if(True):
							dense_weight_2_1.x = square_x + 37.5
							dense_weight_2_2.x = square_x + 37.5
							dense_weight_2_3.x = square_x + 37.5
							dense_weight_2_4.x = square_x + 37.5

						# Weights 3
						if(True):

							dense_weight_3_1.x = square_x + 37.5
							dense_weight_3_2.x = square_x + 37.5
							dense_weight_3_3.x = square_x + 37.5
							dense_weight_3_4.x = square_x + 37.5

						# Weights 4
						if(True):
							dense_weight_4_1.x = square_x + 40
							dense_weight_4_2.x = square_x + 37.5
							dense_weight_4_3.x = square_x + 37.5

						# Weights 5
						if(True):
							dense_weight_5_1.x = square_x + 40
							dense_weight_5_2.x = square_x + 37.5
							dense_weight_5_3.x = square_x + 37.5

						# Weights 6
						if(True):
							dense_weight_6_1.x = square_x + 40
							dense_weight_6_2.x = square_x + 37.5
							dense_weight_6_3.x = square_x + 37.5

						# Weights 7
						if(True):
							dense_weight_7_1.x = square_x + 40
							dense_weight_7_2.x = square_x + 37.5
							dense_weight_7_3.x = square_x + 37.5   

					# Labels
					if(True):
						dense_labels['dotproduct2'].default_opacities = [0]
						dense_labels['dotproduct2'].opacity = [0]

						dense_labels['activation2'].default_opacities = [0.9]
						dense_labels['activation2'].opacity = [0.9]

					return

		# Buttons
		if(True):
			dense_prev_button = widgets.Button(description = 'Previous',
											   disabled = True)
			dense_next_button = widgets.Button(description = 'Next',
											   disabled = False)
		
			dense_frame = {'value' : 0}

			def dense_prev_button_press(*args):
				if(dense_frame['value'] > 0):
					dense_frame['value'] += -1

				dense_anim(dense_frame['value'])
				if(dense_frame['value'] == 0):
					dense_prev_button.disabled = True
				if(dense_frame['value'] == 4):
					dense_next_button.disabled = False      
			def dense_next_button_press(*args):
				if(dense_frame['value'] < 5):
					dense_frame['value'] += 1
				dense_anim(dense_frame['value'])
				if(dense_frame['value'] == 1):
					dense_prev_button.disabled = False
				if(dense_frame['value'] == 5):
					dense_next_button.disabled = True

			dense_prev_button.on_click(dense_prev_button_press)
			dense_next_button.on_click(dense_next_button_press)
		
		# Display
		if(True):
			dense_anim(0)

			dense_buttons = widgets.HBox([dense_prev_button, dense_next_button])
			display(dense_fig,dense_buttons)
			
			

			
def show_operation_conv():
	# Convolution Layer

	if(True):
		# Functions
		if(True):
			def square_mark(scale_x, scale_y, size = 1, x = 0, y = 0, fill_color = 'red', color = 'red'):
				square_x, square_y = create_square(size = size)
				mark = bq.Lines(x = square_x + x, y = square_y + y,
								scales = {'x' : scale_x, 'y': scale_y},
								fill = 'inside',
								opacities = [0.9, 0.9, 0.9, 0.9, 0.9],
								something = [0.9, 0.9, 0.9, 0.9, 0.9],
								colors = [color],
								fill_colors = [fill_color])
				return mark

		# Scales and Axes
		if(True):
			conv_x_sc = plt.LinearScale(min = 0, max = 40)
			conv_y_sc = plt.LinearScale(min = 0, max = 20)

			conv_ax_x = bq.Axis(scale= conv_x_sc)
			conv_ax_y = bq.Axis(scale= conv_y_sc, orientation = 'vertical')

		# Kernels
		if(True):
			size = 1.5
			square_x, square_y = create_square(size = size)
			kernel_x = 10
			kernel_y = 5.2
			# Kernel 1
			if(True):
				conv_kernel_1 = []
				for i in range(3):
					for j in range(3):
						conv_kernel_1 += [square_mark(x = kernel_x + size*j, y = kernel_y + (2-i)*size ,
													  size = size,
													  scale_x = conv_x_sc, scale_y = conv_y_sc,
													  fill_color = 'pink', color='red')]
			# Kernel 2
			if(True):
				conv_kernel_2 = []
				for i in range(3):
					for j in range(3):
						conv_kernel_2 += [square_mark(x = kernel_x + size*j, y = kernel_y + (2-i)*size ,
													  size = size,
													  scale_x = conv_x_sc, scale_y = conv_y_sc,
													  fill_color = 'pink', color='red')]      
			# Kernel 3
			if(True):
				conv_kernel_3 = []

				for i in range(3):
					for j in range(3):
						conv_kernel_3 += [square_mark(x = kernel_x + size*j, y = kernel_y + (2-i)*size ,
													  size = size,
													  scale_x = conv_x_sc, scale_y = conv_y_sc,
													  fill_color = 'pink', color='red')]           
			# Kernel 4
			if(True):
				conv_kernel_4 = []

				for i in range(3):
					for j in range(3):
						conv_kernel_4 += [square_mark(x = kernel_x + size*j, y = kernel_y + (2-i)*size ,
													  size = size,
													  scale_x = conv_x_sc, scale_y = conv_y_sc,
													  fill_color = 'pink', color='red')]
		
		# Inputs
		if(True):
			size = 1.5
			square_x, square_y = create_square(size = size)
			input_x = 1
			input_y = 6
			# Input 1
			if(True):
				input1_x = input_x
				input1_y = input_y
				conv_input_1 = []
				for i in range(3):
					for j in range(3):
						conv_input_1 += [square_mark(x = input1_x + j*size, y = input1_y + (2-i)*size,
													 size = size,
													 scale_x = conv_x_sc, scale_y = conv_y_sc,
													 fill_color = '#D6EAF8', color = 'steelblue')]
			# Input 2
			if(True):
				input2_x = input_x + size
				input2_y = input_y
				conv_input_2 = []
				for i in range(3):
					for j in range(3):
						conv_input_2 += [square_mark(x = input2_x + j*size, y = input2_y + (2-i)*size,
													 size = size,
													 scale_x = conv_x_sc, scale_y = conv_y_sc,
													 fill_color = '#BB8FCE', color = 'purple')]
			# Input 3
			if(True):
				input3_x = input_x
				input3_y = input_y - size
				conv_input_3 = []
				for i in range(3):
					for j in range(3):
						conv_input_3 += [square_mark(x = input3_x + j*size, y = input3_y + (2-i)*size,
													 size = size,
													 scale_x = conv_x_sc, scale_y = conv_y_sc,
													 fill_color = 'gray', color = 'black')]
			# Input 4
			if(True):
				input4_x = input_x + size
				input4_y = input_y - size
				conv_input_4 = []
				for i in range(3):
					for j in range(3):
						conv_input_4 += [square_mark(x = input4_x + j*size, y = input4_y + (2-i)*size,
													 size = size,
													 scale_x = conv_x_sc, scale_y = conv_y_sc,
													 fill_color = '#2ECC71', color = 'green')]

		# Background Input
		if(True):
			inputbg_x = input_x
			inputbg_y = input_y
			conv_bg_input = []
			for i in range(4):
				for j in range(4):
					conv_bg_input += [square_mark(x = input1_x + j*size, y = input1_y + (2-i)*size,
												  size = size,
												  scale_x = conv_x_sc, scale_y = conv_y_sc,
												  fill_color = 'lightblue', color = 'lightblue')]              
		
		# Background Kernel
		if(True):
			kernelbg_x = kernel_x
			kernelbg_y = kernel_y
			conv_bg_kernel = []
			for i in range(3):
				for j in range(3):
						conv_bg_kernel += [square_mark(x = kernel_x + size*j, y = kernel_y + (2-i)*size ,
													   size = size,
													   scale_x = conv_x_sc, scale_y = conv_y_sc,
													   fill_color = 'pink', color='pink')]

		# Step Labels
		if(True):
			conv_step_labels = {
				'patches' : plt.Label(text = ['Step 1 : Convolution Patches'],
									  x = [8.5],
									  y = [2],
									  colors = ['steelblue'],
									  scales = {'x': conv_x_sc, 'y': conv_y_sc},
									  default_size = 28,
									  default_opacities = [0.9],
									  update_on_move = True,
									  font_weight = 'bolder'),

				'conv_prod' : plt.Label(text = ['Step 2 : Convolution Products'],
										x = [8.2],
										y = [2],
										colors = ['steelblue'],
										scales = {'x': conv_x_sc, 'y': conv_y_sc},
										default_size = 28,
										default_opacities = [0.9],
										update_on_move = True,
										font_weight = 'bolder'),

				'concat' : plt.Label(text = ['Step 3 : Concatenation and Activation'],
									 x = [5],
									 y = [2],
									 colors = ['steelblue'],
									 scales = {'x': conv_x_sc, 'y': conv_y_sc},
										default_size = 28,
										default_opacities = [0.9],
										update_on_move = True,
										font_weight = 'bolder'),

				'output' : plt.Label(text = ['Step 4: Output'],
									  x = [14.3],
									  y = [2],
									  colors = ['steelblue'],
									  scales = {'x': conv_x_sc, 'y': conv_y_sc},
									  default_size = 28,
									  default_opacities = [0.9],
									  update_on_move = True,
									  font_weight = 'bolder'),

				'input' : plt.Label(text = ['Input Matrix'],
									x = [0.8],
									y = [11.5],
									colors = ['#3498DB'],
									scales = {'x' : conv_x_sc, 'y': conv_y_sc},
									default_size = 19,
									update_on_move = True,
									font_weight = 'bolder'),

				'kernel' : plt.Label(text = ['Convolution Kernel'],
									x = [8.2],
									y = [10.5],
									colors = ['#C0392B'],
									scales = {'x' : conv_x_sc, 'y': conv_y_sc},
									default_size = 16,
									update_on_move = True,
									font_weight = 'bolder')
			}
			conv_step_labels_list = []

			for label in conv_step_labels.keys():
				conv_step_labels_list += [conv_step_labels[label]]

		# Figure
		if(True):
			conv_fig = bq.Figure(marks =  conv_bg_input + conv_bg_kernel \
										 + conv_kernel_1 + conv_kernel_2 + conv_kernel_3 + conv_kernel_4  \
										 + conv_input_1 + conv_input_2 + conv_input_3 + conv_input_4       \
										 + conv_step_labels_list \
										 ,
								 background_style = {'fill': 'white'},
								 title = 'Convolution Neuron - Forward Pipeline',
								 #axes = [dense_ax_x, dense_ax_y],
								 animation_duration = 1000)

			conv_fig.layout.height = '475px'
			conv_fig.layout.width = '800px'

		# Animation
		if(True):
			def conv_anim(frame):
				if(frame == 0):
					if(True):
						size = 1.5
						square_x, square_y = create_square(size = size)

						input_x = 1
						input_y = 6

						input1_x = input_x
						input1_y = input_y

						input2_x = input_x + size
						input2_y = input_y

						input3_x = input_x
						input3_y = input_y - size

						input4_x = input_x + size
						input4_y = input_y - size

						# Labels
						if(True):
							conv_step_labels['patches'].default_opacities = [0.9]
							conv_step_labels['patches'].opacity = [0.9]

							conv_step_labels['conv_prod'].default_opacities = [0]
							conv_step_labels['conv_prod'].opacity = [0]

							conv_step_labels['concat'].default_opacities = [0]
							conv_step_labels['concat'].opacity = [0]

							conv_step_labels['output'].default_opacities = [0]
							conv_step_labels['output'].opacity = [0]

						# Inputs
						if(True):
							for i in range(3):
								for j in range(3):
									conv_input_4[3*i + j].x = square_x + input4_x + j*size
									conv_input_4[3*i + j].y = square_y + input4_y + (2-i)*size
									conv_input_4[3*i + j].opacities = [0.001]

									conv_input_3[3*i + j].x = square_x + input3_x + j*size
									conv_input_3[3*i + j].y = square_y + input3_y + (2-i)*size
									conv_input_3[3*i + j].opacities = [0.001]


									conv_input_2[3*i + j].x = square_x + input2_x + j*size
									conv_input_2[3*i + j].y = square_y + input2_y + (2-i)*size
									conv_input_2[3*i + j].opacities = [0.001]

									conv_input_1[3*i + j].x = square_x + input1_x + j*size
									conv_input_1[3*i + j].y = square_y + input1_y + (2-i)*size
									conv_input_1[3*i + j].opacities = [0.9]

						# Kernels
						if(True):
							kernel_x = 10
							kernel_y = 5.2

							square_x, square_y = create_square(size = size)
							for i in range(3):
								for j in range(3):
									conv_kernel_1[3*i + j].x = square_x + kernel_x + size*j
									conv_kernel_1[3*i + j].y = square_y + kernel_y + (2-i)*size
									conv_kernel_1[3*i + j].opacities = [0.9]

									conv_kernel_2[3*i + j].x = square_x + kernel_x + size*j
									conv_kernel_2[3*i + j].y = square_y + kernel_y + (2-i)*size
									conv_kernel_2[3*i + j].opacities = [0.9]

									conv_kernel_3[3*i + j].x = square_x + kernel_x + size*j
									conv_kernel_3[3*i + j].y = square_y + kernel_y + (2-i)*size
									conv_kernel_3[3*i + j].opacities = [0.9]

									conv_kernel_4[3*i + j].x = square_x + kernel_x + size*j
									conv_kernel_4[3*i + j].y = square_y + kernel_y + (2-i)*size
									conv_kernel_4[3*i + j].opacities = [0.9]
					return
				if(frame == 1):
					if(True):
						size = 1
						size2 = 1.5
						square_x, square_y = create_square(size = size)
						square_x2, square_y2 = create_square(size = size2)
						input_x = 1
						input_y = 6
						kernel_x = 10
						kernel_y = 5.2

						input1_x = input_x
						input1_y = input_y
						input2_x = input_x + 1.5
						input2_y = input_y

						# Frame return
						if(True):
							for i in range(3):
								for j in range(3):
									conv_input_2[3*i + j].x = square_x2 + input2_x + j*size2
									conv_input_2[3*i + j].y = square_y2 + input2_y + (2-i)*size2
									conv_input_3[3*i + j].opacities = [0.001]

									conv_kernel_2[3*i + j].x = square_x2 + kernel_x + size2*j
									conv_kernel_2[3*i + j].y = square_y2 + kernel_y + (2-i)*size2
									conv_kernel_2[3*i + j].opacities = [0.9]

						# Convolution Patch 1
						if(True):
							for i in range(3):
								for j in range(3):
									conv_input_1[3*i + j].x = square_x + 17.5 + size*j
									conv_input_1[3*i + j].y = square_y + 15 + size*(3-i)
							for i in range(3):
								for j in range(3):
									conv_kernel_1[3*i + j].x = square_x + 21.5 + size*j
									conv_kernel_1[3*i + j].y = square_y + 15  + size*(3-i)


									conv_input_2[3*i + j].opacities = [0.9]
					return
				if(frame == 2):
					if(True):
						size = 1
						size2 = 1.5
						square_x, square_y = create_square(size = size)
						square_x2, square_y2 = create_square(size = size2)

						input_x = 1
						input_y = 6            
						input3_x = input_x
						input3_y = input_y - size2
						kernel_x = 10
						kernel_y = 5.2

						# Frame return
						if(True):
							for i in range(3):
								for j in range(3):
									conv_input_3[3*i + j].x = square_x2 + input3_x + j*size2
									conv_input_3[3*i + j].y = square_y2 + input3_y + (2-i)*size2
									conv_input_3[3*i + j].opacities = [0.001]

									conv_kernel_3[3*i + j].x = square_x2 + kernel_x + size2*j
									conv_kernel_3[3*i + j].y = square_y2 + kernel_y + (2-i)*size2
									conv_kernel_3[3*i + j].opacities = [0.9]



						# Convolution Patch 2
						if(True):
							for i in range(3):
								for j in range(3):
									conv_input_2[3*i + j].x = square_x + 27.5 + size*j
									conv_input_2[3*i + j].y = square_y + 15 + size*(3-i)

									conv_input_4[3*i + j].opacities = [0]
									conv_input_3[3*i + j].opacities = [0.9]

							for i in range(3):
								for j in range(3):
									conv_kernel_2[3*i + j].x = square_x + 31.5 + size*j
									conv_kernel_2[3*i + j].y = square_y + 15  + size*(3-i)


					return
				if(frame == 3):
					if(True):
						size = 1
						size2 = 1.5
						square_x, square_y = create_square(size = size)
						square_x2, square_y2 = create_square(size = size2)

						input_x = 1
						input_y = 6            
						input4_x = input_x + size2
						input4_y = input_y - size2
						kernel_x = 10
						kernel_y = 5.2

						# Frame return
						if(True):
							for i in range(3):
								for j in range(3):
									conv_input_4[3*i + j].x = square_x2 + input4_x + j*size2
									conv_input_4[3*i + j].y = square_y2 + input4_y + (2-i)*size2
									conv_input_4[3*i + j].opacities = [0.001]

									conv_kernel_4[3*i + j].x = square_x2 + kernel_x + size2*j
									conv_kernel_4[3*i + j].y = square_y2 + kernel_y + (2-i)*size2
									conv_kernel_4[3*i + j].opacities = [0.9]


						# Convolution Patch 3
						if(True):
							for i in range(3):
								for j in range(3):
									conv_input_3[3*i + j].x = square_x + 17.5 + size*j
									conv_input_3[3*i + j].y = square_y + 11 + size*(3-i)

									conv_input_4[3*i + j].opacities = [0.9]

							for i in range(3):
								for j in range(3):
									conv_kernel_3[3*i + j].x = square_x + 21.5 + size*j
									conv_kernel_3[3*i + j].y = square_y + 11  + size*(3-i)

					return
				if(frame == 4):
					if(True):
						size = 1
						square_x, square_y = create_square(size = size)

						# Frame Return
						if(True):
							# Convolution Patch 3
							if(True):
								for i in range(3):
									for j in range(3):
										conv_input_3[3*i + j].x = square_x + 17.5 + size*j
										conv_input_3[3*i + j].y = square_y + 11 + size*(3-i)

								for i in range(3):
									for j in range(3):
										conv_kernel_3[3*i + j].x = square_x + 21.5 + size*j
										conv_kernel_3[3*i + j].y = square_y + 11  + size*(3-i)
							# Convolution Patch 2
							if(True):
								for i in range(3):
									for j in range(3):
										conv_input_2[3*i + j].x = square_x + 27.5 + size*j
										conv_input_2[3*i + j].y = square_y + 15 + size*(3-i)

								for i in range(3):
									for j in range(3):
										conv_kernel_2[3*i + j].x = square_x + 31.5 + size*j
										conv_kernel_2[3*i + j].y = square_y + 15  + size*(3-i)
							# Convolution Patch 1
							if(True):
								for i in range(3):
									for j in range(3):
										conv_input_1[3*i + j].x = square_x + 17.5 + size*j
										conv_input_1[3*i + j].y = square_y + 15 + size*(3-i)
								for i in range(3):
									for j in range(3):
										conv_kernel_1[3*i + j].x = square_x + 21.5 + size*j
										conv_kernel_1[3*i + j].y = square_y + 15  + size*(3-i)

							conv_step_labels['patches'].default_opacities = [0.9]
							conv_step_labels['patches'].opacity = [0.9]

							conv_step_labels['conv_prod'].default_opacities = [0]
							conv_step_labels['conv_prod'].opacity = [0]


						# Convolution Patch 4
						if(True):
							for i in range(3):
								for j in range(3):
									conv_input_4[3*i + j].x = square_x + 27.5 + size*j
									conv_input_4[3*i + j].y = square_y + 11 + size*(3-i)

							for i in range(3):
								for j in range(3):
									conv_kernel_4[3*i + j].x = square_x + 31.5 + size*j
									conv_kernel_4[3*i + j].y = square_y + 11  + size*(3-i)
					return
				if(frame == 5):
					size = 0.6
					# Frame Return
					if(True):
						for i in range(3):
							for j in range(3):
								conv_input_1[3*i + j].opacities = [0.9]
								conv_input_2[3*i + j].opacities = [0.9]
								conv_input_3[3*i + j].opacities = [0.9]
								conv_input_4[3*i + j].opacities = [0.9]

								conv_kernel_1[3*i + j].opacities = [0.9]
								conv_kernel_2[3*i + j].opacities = [0.9]
								conv_kernel_3[3*i + j].opacities = [0.9]
								conv_kernel_4[3*i + j].opacities = [0.9]

					# Labels
					if(True):
						conv_step_labels['patches'].default_opacities = [0]
						conv_step_labels['patches'].opacity = [0]

						conv_step_labels['conv_prod'].default_opacities = [0.9]
						conv_step_labels['conv_prod'].opacity = [0.9]

					# Convolution 1
					if(True):
						square_x, square_y = create_square(size = size)
						for i in range(3):
							for j in range(3):
								conv_input_1[3*i + j].x = square_x + 12.5 + (size * 2.6) * j + i* (size * 7.8)
								conv_input_1[3*i + j].y = square_y + 17.5

						for i in range(3):
							for j in range(3):
								conv_kernel_1[3*i + j].x = square_x + (12.6 + size) + (size * 2.6) * j + i* (size * 7.8)
								conv_kernel_1[3*i + j].y = square_y + 17.5
					# Convolution 2  
					if(True):
						for i in range(3):
							for j in range(3):
								conv_input_2[3*i + j].x = square_x + 28 + (size * 2.6) * j + i* (size * 7.8)
								conv_input_2[3*i + j].y = square_y + 17.5

						for i in range(3):
							for j in range(3):
								conv_kernel_2[3*i + j].x = square_x + (28.1 + size) + (size * 2.6) * j + i* (size * 7.8)
								conv_kernel_2[3*i + j].y = square_y + 17.5
					# Convolution 3
					if(True):
						square_x, square_y = create_square(size = size)
						for i in range(3):
							for j in range(3):
								conv_input_3[3*i + j].x = square_x + 12.5 + (size * 2.6) * j + i* (size * 7.8)
								conv_input_3[3*i + j].y = square_y + 13.5

						for i in range(3):
							for j in range(3):
								conv_kernel_3[3*i + j].x = square_x + (12.6 + size) + (size * 2.6) * j + i* (size * 7.8)
								conv_kernel_3[3*i + j].y = square_y + 13.5
					# Convolution 4
					if(True):
						square_x, square_y = create_square(size = size)
						for i in range(3):
							for j in range(3):
								conv_input_4[3*i + j].x = square_x + 28 + (size * 2.6) * j + i* (size * 7.8)
								conv_input_4[3*i + j].y = square_y + 13.5

						for i in range(3):
							for j in range(3):
								conv_kernel_4[3*i + j].x = square_x + (28.1 + size) + (size * 2.6) * j + i* (size * 7.8)
								conv_kernel_4[3*i + j].y = square_y + 13.5     
					return
				if(frame == 6):
					if(True):
						size = 1

						# Frame Return
						if(True):
							# Labels
							if(True):
								conv_step_labels['concat'].default_opacities = [0]
								conv_step_labels['concat'].opacity = [0]

								conv_step_labels['conv_prod'].default_opacities = [0.9]
								conv_step_labels['conv_prod'].opacity = [0.9]


						# Dot Product 1
						if(True):
							square_x, square_y = create_square(size = size)
							for i in range(3):
								for j in range(3):
									conv_input_1[3*i + j].x = square_x + 17.75
									conv_input_1[3*i + j].y = square_y + 17.25

							for i in range(3):
								for j in range(3):
									conv_kernel_1[3*i + j].x = square_x + 17.75
									conv_kernel_1[3*i + j].y = square_y + 17.25

						# Dot Product 2
						if(True):
							square_x, square_y = create_square(size = size)
							for i in range(3):
								for j in range(3):
									conv_input_2[3*i + j].x = square_x + 33.25
									conv_input_2[3*i + j].y = square_y + 17.25

							for i in range(3):
								for j in range(3):
									conv_kernel_2[3*i + j].x = square_x + 33.25
									conv_kernel_2[3*i + j].y = square_y + 17.25

						# Dot Product 3
						if(True):
							square_x, square_y = create_square(size = size)
							for i in range(3):
								for j in range(3):
									conv_input_3[3*i + j].x = square_x + 17.75
									conv_input_3[3*i + j].y = square_y + 13.25

							for i in range(3):
								for j in range(3):
									conv_kernel_3[3*i + j].x = square_x + 17.75
									conv_kernel_3[3*i + j].y = square_y + 13.25

						# Dot Product 4
						if(True):
							square_x, square_y = create_square(size = size)
							for i in range(3):
								for j in range(3):
									conv_input_4[3*i + j].x = square_x + 33.25
									conv_input_4[3*i + j].y = square_y + 13.25
							for i in range(3):
								for j in range(3):
									conv_kernel_4[3*i + j].x = square_x + 33.25
									conv_kernel_4[3*i + j].y = square_y + 13.25
					return
				if(frame == 7):
					size = 1
					# Labels
					if(True):
						conv_step_labels['output'].default_opacities = [0]
						conv_step_labels['output'].opacity = [0]

						conv_step_labels['concat'].default_opacities = [0.9]
						conv_step_labels['concat'].opacity = [0.9]

						conv_step_labels['conv_prod'].default_opacities = [0]
						conv_step_labels['conv_prod'].opacity = [0]



					# Dot Product 1
					if(True):
						square_x, square_y = create_square(size = size)
						for i in range(3):
							for j in range(3):
								if((3*i + j) != 8):
									conv_input_1[3*i + j].opacities = [0]
									conv_kernel_1[3*i + j].opacities = [0]

						conv_input_1[8].x = square_x + 25.25
						conv_kernel_1[8].x = square_x + 25.25 

						conv_input_1[8].y = square_y + 16.25
						conv_kernel_1[8].y = square_y + 16.25
					# Dot Product 2
					if(True):
						square_x, square_y = create_square(size = size)
						for i in range(3):
							for j in range(3):
								if((3*i + j) != 8):
									conv_input_2[3*i + j].opacities = [0]
									conv_kernel_2[3*i + j].opacities = [0]

						conv_input_2[8].x = square_x + 25.25 + size
						conv_kernel_2[8].x = square_x + 25.25 + size

						conv_input_2[8].y = square_y + 16.25
						conv_kernel_2[8].y = square_y + 16.25
					# Dot Product 3    
					if(True):
						square_x, square_y = create_square(size = size)
						for i in range(3):
							for j in range(3):
								if((3*i + j) != 8):
									conv_input_3[3*i + j].opacities = [0]
									conv_kernel_3[3*i + j].opacities = [0]

						conv_input_3[8].x = square_x + 25.25 
						conv_kernel_3[8].x = square_x + 25.25

						conv_input_3[8].y = square_y + 16.25 - size
						conv_kernel_3[8].y = square_y + 16.25 - size
					# Dot Product 4
					if(True):
						square_x, square_y = create_square(size = size)
						for i in range(3):
							for j in range(3):
								if((3*i + j) != 8):
									conv_input_4[3*i + j].opacities = [0]
									conv_kernel_4[3*i + j].opacities = [0]

						conv_input_4[8].x = square_x + 25.25 + size
						conv_kernel_4[8].x = square_x + 25.25 + size

						conv_input_4[8].y = square_y + 16.25 - size
						conv_kernel_4[8].y = square_y + 16.25 - size
				if(frame == 8):
					if(True):

						# Labels
						if(True):
							conv_step_labels['concat'].default_opacities = [0]
							conv_step_labels['concat'].opacity = [0]

							conv_step_labels['output'].default_opacities = [0.9]
							conv_step_labels['output'].opacity = [0.9]

						size = 1
						square_x,square_y = create_square(size = size)

						# Dot Product 1
						if(True):
							conv_input_1[8].x = square_x + 25.25
							conv_kernel_1[8].x = square_x + 25.25 

							conv_input_1[8].y = square_y + 7.5
							conv_kernel_1[8].y = square_y + 7.5

						# Dot Product 2
						if(True):
							conv_input_2[8].x = square_x + 25.25 + size
							conv_kernel_2[8].x = square_x + 25.25 + size

							conv_input_2[8].y = square_y + 7.5
							conv_kernel_2[8].y = square_y + 7.5

						# Dot Product 3
						if(True):
							conv_input_3[8].x = square_x + 25.25
							conv_kernel_3[8].x = square_x + 25.25

							conv_input_3[8].y = square_y + 7.5 - size
							conv_kernel_3[8].y = square_y + 7.5 - size

						# Dot Product 4
						if(True):
							conv_input_4[8].x = square_x + 25.25 + size
							conv_kernel_4[8].x = square_x + 25.25 + size

							conv_input_4[8].y = square_y + 7.5 - size
							conv_kernel_4[8].y = square_y + 7.5 - size
				if(frame == 9):
					if(True):
						size = 1
						square_x,square_y = create_square(size = size)

						# Labels
						if(True):
							conv_step_labels['concat'].default_opacities = [0]
							conv_step_labels['concat'].opacity = [0]

							conv_step_labels['output'].default_opacities = [0.9]
							conv_step_labels['output'].opacity = [0.9]


						# Dot Product 1
						if(True):
							conv_input_1[8].x = square_x + 50
							conv_kernel_1[8].x = square_x + 50 

							conv_input_1[8].y = square_y + 7.5
							conv_kernel_1[8].y = square_y + 7.5

						# Dot Product 2
						if(True):
							conv_input_2[8].x = square_x + 50 + size
							conv_kernel_2[8].x = square_x + 50 + size

							conv_input_2[8].y = square_y + 7.5
							conv_kernel_2[8].y = square_y + 7.5

						# Dot Product 3
						if(True):
							conv_input_3[8].x = square_x + 50
							conv_kernel_3[8].x = square_x + 50

							conv_input_3[8].y = square_y + 7.5 - size
							conv_kernel_3[8].y = square_y + 7.5 - size

						# Dot Product 4
						if(True):
							conv_input_4[8].x = square_x + 50 + size
							conv_kernel_4[8].x = square_x + 50 + size

							conv_input_4[8].y = square_y + 7.5 - size
							conv_kernel_4[8].y = square_y + 7.5 - size

		# Buttons
		if(True):
			conv_prev_button = widgets.Button(description = 'Previous',
											   disabled = True)
			conv_next_button = widgets.Button(description = 'Next',
											   disabled = False)
			conv_frame = {'value' : 0}

			def conv_prev_button_press(*args):
				if(conv_frame['value'] > 0):
					conv_frame['value'] += -1
				conv_anim(conv_frame['value'])
				if(conv_frame['value'] == 0):
					conv_prev_button.disabled = True
				if(conv_frame['value'] == 8):
					conv_next_button.disabled = False  

			def conv_next_button_press(*args):
				if(conv_frame['value'] < 9):
					conv_frame['value'] += 1
					conv_anim(conv_frame['value'])
				if(conv_frame['value'] == 1):
					conv_prev_button.disabled = False
				if(conv_frame['value'] == 9):
					conv_next_button.disabled = True

			conv_prev_button.on_click(conv_prev_button_press)
			conv_next_button.on_click(conv_next_button_press)

		# Display
		if(True):
			conv_anim(0)

			conv_buttons = widgets.HBox([conv_prev_button, conv_next_button])
			display(conv_fig, conv_buttons)

			
			
			
			

import ipywidgets as widgets

def show_conv(img_path='taj_mahal.jpg'):
    # Interactive Convolution
	if(True):
		# Image

		taj_mahal = cv2.imread(img_path, 0)

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
						im_matplotlib.imsave(buffer, arr, format=format, cmap=cmap, vmin=0, vmax=vmax)
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
				w1 = widgets.FloatText(value = 0, layout = widgets.Layout(width = '40px'))
				w2 = widgets.FloatText(value = 0, layout = widgets.Layout(width = '40px'))
				w3 = widgets.FloatText(value = 0, layout = widgets.Layout(width = '40px'))
				w4 = widgets.FloatText(value = 0, layout = widgets.Layout(width = '40px'))
				w5 = widgets.FloatText(value = 1, layout = widgets.Layout(width = '40px'))
				w6 = widgets.FloatText(value = 0, layout = widgets.Layout(width = '40px'))
				w7 = widgets.FloatText(value = 0, layout = widgets.Layout(width = '40px'))
				w8 = widgets.FloatText(value = 0, layout = widgets.Layout(width = '40px'))
				w9 = widgets.FloatText(value = 0, layout = widgets.Layout(width = '40px'))

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
			
			
			
			
			
			
def show_maxpool():
	# Max Pooling Layer    
	if(True):
		# Functions
		if(True):
			def create_square(size = 1, starting_x = 0, starting_y = 0):
				x = np.array([starting_x, starting_x + size, starting_x + size, starting_x, starting_x])
				y = np.array([starting_y, starting_y, starting_y + size, starting_y + size, starting_y])
				return x,y

			def square_mark(scale_x, scale_y, size = 1, x = 0, y = 0, fill_color = 'red', color = 'red'):
				square_x, square_y = create_square(size = size)
				mark = bq.Lines(x = square_x + x, y = square_y + y,
								scales = {'x' : scale_x, 'y': scale_y},
								fill = 'inside',
								opacities = [1],
								fill_opacities = [1],
								colors = [color],
								fill_colors = [fill_color])
				return mark
			
			def rgb2hex(red = 0, green = 0, blue = 0):
				return '#{:02x}{:02x}{:02x}'.format(red, green, blue)
			
			def hex2rgb(hex):
				hex = hex.lstrip('#')
				hlen = len(hex)
				return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

		# Scales and Axes
		if(True):
			max_x_sc = plt.LinearScale(min = 0, max = 40)
			max_y_sc = plt.LinearScale(min = 0, max = 20)

			max_ax_x = bq.Axis(scale= max_x_sc)
			max_ax_y = bq.Axis(scale= max_y_sc, orientation = 'vertical')
		
		# Inputs
		if(True):
			size = 1.5
			square_x, square_y = create_square(size = size)
			input_x = 14
			input_y = 2
			# Input 1
			if(True):
				input1_x = input_x
				input1_y = input_y
				max_input_1 = []
				for i in range(8):
					for j in range(8):
						max_input_1 += [square_mark(x = input1_x + j*size, y = input1_y + (7-i)*size,
													size = size,
													scale_x = max_x_sc, scale_y = max_y_sc,
													fill_color = rgb2hex(0, 0, np.random.randint(0, 255)), color = 'steelblue')]
				
		# Step Labels
		if(True):
			max_step_labels = {
				'input' : plt.Label(text = ['Step 0 : Collection of Inputs'],
									 x = [8.2],
									 y = [18],
									 colors = ['steelblue'],
									 scales = {'x': max_x_sc, 'y': max_y_sc},
									 default_size = 30,
									 default_opacities = [0.9],
									 update_on_move = True,
									 font_weight = 'bolder'),

				'split' : plt.Label(text = ['Step 1 : Split'],
										x = [14.6],
										y = [18],
										colors = ['steelblue'],
										scales = {'x': max_x_sc, 'y': max_y_sc},
										default_size = 30,
										default_opacities = [0.9],
										update_on_move = True,
										font_weight = 'bolder'),

				'filter' : plt.Label(text = ['Step 2 : Max Filtering'],
									 x = [11],
									 y = [18],
									 colors = ['steelblue'],
									 scales = {'x': max_x_sc, 'y': max_y_sc},
									 default_size = 30,
									 default_opacities = [0.9],
									 update_on_move = True,
									 font_weight = 'bolder'),

				'pool' : plt.Label(text = ['Step 3: Pooling'],
									x = [13.5],
									y = [18],
									colors = ['steelblue'],
									scales = {'x': max_x_sc, 'y': max_y_sc},
									default_size = 30,
									default_opacities = [0.9],
									update_on_move = True,
									font_weight = 'bolder'),

				'output' : plt.Label(text = ['Step 4 : Output'],
										x = [13.6],
										y = [18],
										colors = ['steelblue'],
										scales = {'x': max_x_sc, 'y': max_y_sc},
										default_size = 30,
										default_opacities = [0.9],
										update_on_move = True,
										font_weight = 'bolder')
			}
			max_step_labels_list = []

			for label in max_step_labels.keys():
				max_step_labels_list += [max_step_labels[label]]
			
		# Figure
		if(True):
			max_fig = plt.Figure(marks = max_input_1 + max_step_labels_list,
								 #axes = [max_ax_x, max_ax_y],
								 background_style = {'fill': 'white'},
								 title = "Max-Pooling Layer - Forward Pipeline",
								 animation_duration = 1000)
			max_fig.layout.height = '475px'
			max_fig.layout.width = '800px'
		
		# Animation
		if(True):
			max_list = {'list' : []}
			max_color_index = {'list' : []}
			colors = {'list' : []}
			def max_anim(frame):
				if(frame == 0):
					if(True):
						# Labels
						if(True):
							for label in max_step_labels_list:
								label.default_opacities = [0]
								label.opacity = [0]
							max_step_labels['input'].default_opacities = [0.9]
							max_step_labels['input'].opacity = [0.9]
						
						# Inputs
						if(True):
							size = 1.5
							input1_x = 14
							input1_y = 2
							square_x, square_y = create_square(size)
							for i in range(8):
								for j in range(8):
									max_input_1[8*i + j].x = square_x + input1_x + j*size
									max_input_1[8*i + j].y = square_y + input1_y + (7-i)*size
									max_input_1[8*i + j].opacities = [1]
					return
				if(frame == 1):
					if(True):
						# Labels
						if(True):
							max_step_labels['input'].default_opacities = [0]
							max_step_labels['input'].opacity = [0]
							
							max_step_labels['split'].default_opacities = [0.9]
							max_step_labels['split'].opacity = [0.9]
							
							max_step_labels['filter'].default_opacities = [0]
							max_step_labels['filter'].opacity = [0]
						
						
						# Split
						if(True):
							size = 1.5
							input1_x = 13.25
							input1_y = 2
							square_x, square_y = create_square(size)

							for i in range(8):
								for j in range(8):
									max_input_1[8*i + j].x = square_x + np.linspace(input1_x-2, input1_x + 8*size + 2, 11)[j + j//2]
									max_input_1[8*i + j].y = square_y + np.linspace(input1_y-2, input1_y + 8*size + 2, 12)[10-(i + i//2)]

									max_input_1[8*i + j].opacities = [1]
					return
				if(frame == 2):
					if(True):
						# Frame Return
						if(True):
							size = 1.5
							input1_x = 13.25
							input1_y = 2
							square_x, square_y = create_square(size)
							max_color_index['list'] = []
							for i in range(8):
								for j in range(8):
									max_input_1[8*i + j].x = square_x + np.linspace(input1_x-2, input1_x + 8*size + 2, 11)[j + j//2]
									max_input_1[8*i + j].y = square_y + np.linspace(input1_y-2, input1_y + 8*size + 2, 12)[10-(i + i//2)]
		
						# Labels
						if(True):
							max_step_labels['split'].default_opacities = [0]
							max_step_labels['split'].opacity = [0]
							
							max_step_labels['filter'].default_opacities = [0.9]
							max_step_labels['filter'].opacity = [0.9]
							
							max_step_labels['pool'].default_opacities = [0]
							max_step_labels['pool'].opacity = [0]
						
						
						# Filtering
						if(True):
							# Finding Max of each pool
							if(True):
								for i in [0, 16, 32, 48]:
									for j in [0, 2, 4, 6]:
										colors['list'] = []
										colors['list'] += [hex2rgb(max_input_1[i + j].fill_colors[0])]
										colors['list'] += [hex2rgb(max_input_1[i + j + 1].fill_colors[0])]
										colors['list'] += [hex2rgb(max_input_1[i + 8 + j].fill_colors[0])]
										colors['list'] += [hex2rgb(max_input_1[i + 8 + j + 1].fill_colors[0])]
										colors['list'] = np.array(colors['list'])[:,2]
										max_color_index['list'] += [np.argmax(colors['list'])]

								k = 0
								
							# Filter
							if(True):
								max_list['list'] = []

								for i in [0, 16, 32, 48]:
									for j in [0, 2, 4, 6]:
										if(max_color_index['list'][k] == 0):
											max_input_1[i + j + 1].opacities = [0.1]
											max_input_1[i + 8 + j].opacities = [0.1]
											max_input_1[i + 8 + j + 1].opacities = [0.1]
											max_list['list'] += [max_input_1[i + j]]
										if(max_color_index['list'][k] == 1):
											max_input_1[i + j].opacities = [0.1]
											max_input_1[i + 8 + j].opacities = [0.1]
											max_input_1[i + 8 + j + 1].opacities = [0.1]
											max_list['list'] += [max_input_1[i + j + 1]]
										if(max_color_index['list'][k] == 2):
											max_input_1[i + j].opacities = [0.1]
											max_input_1[i + j + 1].opacities = [0.1]
											max_input_1[i + 8 + j + 1].opacities = [0.1]
											max_list['list'] += [max_input_1[i + 8 + j]]
										if(max_color_index['list'][k] == 3):
											max_input_1[i + j].opacities = [0.1]
											max_input_1[i + j + 1].opacities = [0.1]
											max_input_1[i + 8 + j].opacities = [0.1]
											max_list['list'] += [max_input_1[i + 8 + j + 1]]
										k = k + 1
					return
				if(frame == 3):
					if(True):         
						# Labels
						if(True):
							max_step_labels['filter'].default_opacities = [0]
							max_step_labels['filter'].opacity = [0]
							
							max_step_labels['pool'].default_opacities = [0.9]
							max_step_labels['pool'].opacity = [0.9]
							
							max_step_labels['output'].default_opacities = [0]
							max_step_labels['output'].opacity = [0]
						
						# Pooling
						if(True):
							size = 2
							input1_x = 16
							input1_y = 6
							square_x, square_y = create_square(size)

							for i in range(8):
								for j in range(8):
									if(max_input_1[8*i + j].opacities[0] == 0.1):
										max_input_1[8*i + j].opacities = [0]

							for i in range(4):
								for j in range(4):
									max_list['list'][4*i + j].x = square_x + input1_x + j*size
									max_list['list'][4*i + j].y = square_y + input1_y + (3-i)*size
					return
				if(frame == 4):
					if(True):                   
					   # Labels
						if(True):
							max_step_labels['pool'].default_opacities = [0]
							max_step_labels['pool'].opacity = [0]
							
							max_step_labels['output'].default_opacities = [0.9]
							max_step_labels['output'].opacity = [0.9]
						 
						# Output
						if(True):
							size = 2
							input1_x = 16
							input1_y = 6
							square_x, square_y = create_square(size)

							for i in range(4):
								for j in range(4):
									max_list['list'][4*i + j].x = square_x + 50 + j*size
									max_list['list'][4*i + j].y = square_y + input1_y + (3-i)*size
									
		# Buttons
		if(True):
			max_frame = {'value' : 0}

			def max_prev_button_press(*args):
				if(max_frame['value'] > 0):
					max_frame['value'] += -1

				max_anim(max_frame['value'])

				if(max_frame['value'] == 0):
					max_prev_button.disabled = True

				if(max_frame['value'] == 3):
					max_next_button.disabled = False  

			def max_next_button_press(*args):
				if(max_frame['value'] < 4):
					max_frame['value'] += 1

				max_anim(max_frame['value'])

				if(max_frame['value'] == 1):
					max_prev_button.disabled = False

				if(max_frame['value'] == 4):
					max_next_button.disabled = True

			def max_shuffle_press(*args):
				max_frame['value'] = 0
				max_prev_button.disabled = True
				max_next_button.disabled = False
				max_anim(0)
				for i in range(8):
					for j in range(8):
						max_input_1[8*i + j].fill_colors = [rgb2hex(0, 0, np.random.randint(0, 255))]
						
			max_prev_button = widgets.Button(description = 'Previous',
											 disabled = True)
			max_next_button = widgets.Button(description = 'Next',
											 disabled = False)
			max_shuffle_button = widgets.Button(description = 'Shuffle Colors',
												disabled = False)

			max_shuffle_button.on_click(max_shuffle_press)
			max_prev_button.on_click(max_prev_button_press)
			max_next_button.on_click(max_next_button_press)
			max_buttons = widgets.HBox([max_prev_button, max_next_button])

		# Display
		if(True):
			max_anim(0)
			display(max_fig, max_buttons, max_shuffle_button)
			
			
			
			
			
			
def average_pool():
	### Average Pooling
	if(True):
		# Functions
		if(True):
			def create_square(size = 1, starting_x = 0, starting_y = 0):
				x = np.array([starting_x, starting_x + size, starting_x + size, starting_x, starting_x])
				y = np.array([starting_y, starting_y, starting_y + size, starting_y + size, starting_y])
				return x,y

			def square_mark(scale_x, scale_y, size = 1, x = 0, y = 0, fill_color = 'red', color = 'red'):
				square_x, square_y = create_square(size = size)
				mark = bq.Lines(x = square_x + x, y = square_y + y,
								scales = {'x' : scale_x, 'y': scale_y},
								fill = 'inside',
								opacities = [1],
								fill_opacities = [1],
								colors = [color],
								fill_colors = [fill_color])
				return mark
			
			def rgb2hex(red = 0, green = 0, blue = 0):
				return '#{:02x}{:02x}{:02x}'.format(red, green, blue)
			
			def hex2rgb(hex):
				hex = hex.lstrip('#')
				hlen = len(hex)
				return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

		# Scales and Axes
		if(True):
			avg_x_sc = plt.LinearScale(min = 0, max = 40)
			avg_y_sc = plt.LinearScale(min = 0, max = 20)

			avg_ax_x = bq.Axis(scale= avg_x_sc)
			avg_ax_y = bq.Axis(scale= avg_y_sc, orientation = 'vertical')
		
		# Inputs
		if(True):
			size = 1.5
			square_x, square_y = create_square(size = size)
			input_x = 14
			input_y = 2
			# Input 1
			if(True):
				input1_x = input_x
				input1_y = input_y
				avg_input_1 = []
				for i in range(8):
					for j in range(8):
						avg_input_1 += [square_mark(x = input1_x + j*size, y = input1_y + (7-i)*size,
													size = size,
													scale_x = avg_x_sc, scale_y = avg_y_sc,
													fill_color = rgb2hex(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), color = 'steelblue')]
				
		# Step Labels
		if(True):
			avg_step_labels = {
				'input' : plt.Label(text = ['Step 0 : Collection of Inputs'],
									 x = [8.2],
									 y = [18],
									 colors = ['steelblue'],
									 scales = {'x': avg_x_sc, 'y': avg_y_sc},
									 default_size = 30,
									 default_opacities = [0.9],
									 update_on_move = True,
									 font_weight = 'bolder'),

				'split' : plt.Label(text = ['Step 1 : Split'],
										x = [14.6],
										y = [18],
										colors = ['steelblue'],
										scales = {'x': avg_x_sc, 'y': avg_y_sc},
										default_size = 30,
										default_opacities = [0.9],
										update_on_move = True,
										font_weight = 'bolder'),

				'filter' : plt.Label(text = ['Step 2 : Averaging'],
									 x = [12.2],
									 y = [18],
									 colors = ['steelblue'],
									 scales = {'x': avg_x_sc, 'y': avg_y_sc},
									 default_size = 30,
									 default_opacities = [0.9],
									 update_on_move = True,
									 font_weight = 'bolder'),

				'pool' : plt.Label(text = ['Step 3: Pooling'],
									x = [13.5],
									y = [18],
									colors = ['steelblue'],
									scales = {'x': avg_x_sc, 'y': avg_y_sc},
									default_size = 30,
									default_opacities = [0.9],
									update_on_move = True,
									font_weight = 'bolder'),

				'output' : plt.Label(text = ['Step 4 : Output'],
										x = [13.6],
										y = [18],
										colors = ['steelblue'],
										scales = {'x': avg_x_sc, 'y': avg_y_sc},
										default_size = 30,
										default_opacities = [0.9],
										update_on_move = True,
										font_weight = 'bolder')
			}
			avg_step_labels_list = []

			for label in avg_step_labels.keys():
				avg_step_labels_list += [avg_step_labels[label]]
			
		# Figure
		if(True):
			avg_fig = plt.Figure(marks = avg_input_1 + avg_step_labels_list,
								 #axes = [avg_ax_x, avg_ax_y],
								 background_style = {'fill': 'white'},
								 title = "Average-Pooling Layer - Forward Pipeline",
								 animation_duration = 1000)
			avg_fig.layout.height = '475px'
			avg_fig.layout.width = '800px'
		
		# Animation
		if(True):   
			avg_list = {'list' : []}
			avg_color_index = {'list' : []}
			colors = {'list' : []}
			def avg_anim(frame):
				if(frame == 0):
					if(True):
						# Labels
						if(True):
							for label in avg_step_labels_list:
								label.default_opacities = [0]
								label.opacity = [0]
							avg_step_labels['input'].default_opacities = [0.9]
							avg_step_labels['input'].opacity = [0.9]
						
						# Inputs
						if(True):
							size = 1.5
							input1_x = 14
							input1_y = 2
							square_x, square_y = create_square(size)
							for i in range(8):
								for j in range(8):
									avg_input_1[8*i + j].x = square_x + input1_x + j*size
									avg_input_1[8*i + j].y = square_y + input1_y + (7-i)*size
									avg_input_1[8*i + j].opacities = [1]
					return
				if(frame == 1):
					if(True):
						# Labels
						if(True):
							avg_step_labels['input'].default_opacities = [0]
							avg_step_labels['input'].opacity = [0]
							
							avg_step_labels['split'].default_opacities = [0.9]
							avg_step_labels['split'].opacity = [0.9]
							
							avg_step_labels['filter'].default_opacities = [0]
							avg_step_labels['filter'].opacity = [0]
						
						
						# Split
						if(True):
							size = 1.5
							input1_x = 13.25
							input1_y = 2
							square_x, square_y = create_square(size)

							for i in range(8):
								for j in range(8):
									avg_input_1[8*i + j].x = square_x + np.linspace(input1_x-2, input1_x + 8*size + 2, 11)[j + j//2]
									avg_input_1[8*i + j].y = square_y + np.linspace(input1_y-2, input1_y + 8*size + 2, 12)[10-(i + i//2)]

									avg_input_1[8*i + j].opacities = [1]
					return
				if(frame == 2):
					if(True):
						# Frame Return
						if(True):
							size = 1.5
							input1_x = 13.25
							input1_y = 2
							square_x, square_y = create_square(size)
							avg_color_index['list'] = []
							avg_x = []
							avg_y = []
							for i in range(8):
								for j in range(8):
									avg_x += [square_x + np.linspace(input1_x-2, input1_x + 8*size + 2, 11)[j + j//2]]
									avg_y += [square_y + np.linspace(input1_y-2, input1_y + 8*size + 2, 12)[10-(i + i//2)]]

						# Labels
						if(True):
							avg_step_labels['split'].default_opacities = [0]
							avg_step_labels['split'].opacity = [0]

							avg_step_labels['filter'].default_opacities = [0.9]
							avg_step_labels['filter'].opacity = [0.9]

							avg_step_labels['pool'].default_opacities = [0]
							avg_step_labels['pool'].opacity = [0]

						# Filtering
						if(True):
							# Filter
							if(True):
								for i in [0, 16, 32, 48]:
									for j in [0, 2, 4, 6]:
										pool_avg_x = (avg_x[i + j][0] + avg_x[i + j + 1][0])/2
										pool_avg_y = (avg_y[i + j][0] + avg_y[i + 8 + j][0])/2

										avg_input_1[i + j].x = square_x + pool_avg_x
										avg_input_1[i + j].y = square_y + pool_avg_y

										avg_input_1[i + j + 1].x = square_x + pool_avg_x
										avg_input_1[i + j + 1].y = square_y + pool_avg_y
										avg_input_1[i + j + 1].opacities = [0.25]

										avg_input_1[i + 8 + j].x = square_x + pool_avg_x
										avg_input_1[i + 8 + j].y = square_y + pool_avg_y
										avg_input_1[i + 8 + j].opacities = [0.5]

										avg_input_1[i + 8 + j + 1].x = square_x + pool_avg_x
										avg_input_1[i + 8 + j + 1].y = square_y + pool_avg_y
										avg_input_1[i + 8 + j + 1].opacities = [0.25]

					return
				if(frame == 3):
					if(True):         
						# Labels
						if(True):
							avg_step_labels['filter'].default_opacities = [0]
							avg_step_labels['filter'].opacity = [0]
							
							avg_step_labels['pool'].default_opacities = [0.9]
							avg_step_labels['pool'].opacity = [0.9]
							
							avg_step_labels['output'].default_opacities = [0]
							avg_step_labels['output'].opacity = [0]
						
						# Pooling
						if(True):
							size = 2
							input1_x = 16
							input1_y = 6
							square_x, square_y = create_square(size)
							

							for i in [0, 16, 32, 48]:
								for j in [0, 2, 4, 6]:

									avg_input_1[i + j].x = square_x + input1_x + (j/2)*size
									avg_input_1[i + j].y = square_y + input1_y + (3-i/16)*size
									
									avg_input_1[i + j + 1].x = square_x + input1_x + (j/2)*size
									avg_input_1[i + j + 1].y = square_y + input1_y + (3-i/16)*size

									avg_input_1[i + 8 + j].x = square_x + input1_x + (j/2)*size
									avg_input_1[i + 8 + j].y = square_y + input1_y + (3-i/16)*size

									avg_input_1[i + 8 + j + 1].x = square_x + input1_x + (j/2)*size
									avg_input_1[i + 8 + j + 1].y = square_y + input1_y + (3-i/16)*size
					return
				if(frame == 4):
					if(True):                   
					   # Labels
						if(True):
							avg_step_labels['pool'].default_opacities = [0]
							avg_step_labels['pool'].opacity = [0]
							
							avg_step_labels['output'].default_opacities = [0.9]
							avg_step_labels['output'].opacity = [0.9]
						 
						# Output
						if(True):
							size = 2
							input1_x = 16
							input1_y = 6
							square_x, square_y = create_square(size)
							

							for i in [0, 16, 32, 48]:
								for j in [0, 2, 4, 6]:

									avg_input_1[i + j].x = square_x + 50 + (j/2)*size
									avg_input_1[i + j].y = square_y + input1_y + (3-i/16)*size
									
									avg_input_1[i + j + 1].x = square_x + 50 + (j/2)*size
									avg_input_1[i + j + 1].y = square_y + input1_y + (3-i/16)*size

									avg_input_1[i + 8 + j].x = square_x + 50 + (j/2)*size
									avg_input_1[i + 8 + j].y = square_y + input1_y + (3-i/16)*size

									avg_input_1[i + 8 + j + 1].x = square_x + 50 + (j/2)*size
									avg_input_1[i + 8 + j + 1].y = square_y + input1_y + (3-i/16)*size
									
		# Buttons
		if(True):
			avg_frame = {'value' : 0}

			def avg_prev_button_press(*args):
				if(avg_frame['value'] > 0):
					avg_frame['value'] += -1

				avg_anim(avg_frame['value'])

				if(avg_frame['value'] == 0):
					avg_prev_button.disabled = True

				if(avg_frame['value'] == 3):
					avg_next_button.disabled = False  

			def avg_next_button_press(*args):
				if(avg_frame['value'] < 4):
					avg_frame['value'] += 1

				avg_anim(avg_frame['value'])

				if(avg_frame['value'] == 1):
					avg_prev_button.disabled = False

				if(avg_frame['value'] == 4):
					avg_next_button.disabled = True

			def avg_shuffle_press(*args):
				avg_frame['value'] = 0
				avg_prev_button.disabled = True
				avg_next_button.disabled = False
				avg_anim(0)
				for i in range(8):
					for j in range(8):
						avg_input_1[8*i + j].fill_colors = [rgb2hex(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))]
						
			avg_prev_button = widgets.Button(description = 'Previous',
											 disabled = True)
			avg_next_button = widgets.Button(description = 'Next',
											 disabled = False)
			avg_shuffle_button = widgets.Button(description = 'Shuffle Colors',
												disabled = False)

			avg_shuffle_button.on_click(avg_shuffle_press)
			avg_prev_button.on_click(avg_prev_button_press)
			avg_next_button.on_click(avg_next_button_press)
			avg_buttons = widgets.HBox([avg_prev_button, avg_next_button])

		# Display
		if(True):
			avg_anim(0)
			display(avg_fig, avg_buttons, avg_shuffle_button)
			
			
			
			
			
			
			
			
			
def show_dropout():
	# Dropout Layer
	if(True):
		# Functions
		if(True):
			import ipywidgets as widgets
			import bqplot as bq
			import bqplot.pyplot as plt
			from IPython.display import display
			import numpy as np

			def create_square(size = 1, starting_x = 0, starting_y = 0):
				x = np.array([starting_x, starting_x + size, starting_x + size, starting_x, starting_x])
				y = np.array([starting_y, starting_y, starting_y + size, starting_y + size, starting_y])
				return x,y

		# Scales and Axes
		if(True):
			drop_x_sc = plt.LinearScale(min = 0, max = 40)
			drop_y_sc = plt.LinearScale(min = 0, max = 20)

			drop_ax_x = bq.Axis(scale= drop_x_sc)
			drop_ax_y = bq.Axis(scale= drop_y_sc, orientation = 'vertical')

		# Labels
		if(True):
			prev_layer_label = plt.Label(text = ['Previous Layer'], x = [-1], y = [20.1],
										 scales = {'x': drop_x_sc, 'y' : drop_y_sc},
										 font_weight = 'bolder',
										 update_on_move = True,
										 colors = ["steelblue"])

			drop_layer_label = plt.Label(text = ['Dense Layer'], x = [16.9], y = [20],
										 scales = {'x': drop_x_sc, 'y' : drop_y_sc},
										 font_weight = 'bolder',
										 update_on_move = True,
										 colors = ["red"])

			next_layer_label = plt.Label(text = ['Another Dense Layer'], x = [30], y = [20],
										 scales = {'x': drop_x_sc, 'y' : drop_y_sc},
										 font_weight = 'bolder',
										 update_on_move = True,
										 colors = ["green"])

			drop_inputs_label = plt.Label(text = ['Collection of inputs'],
									 x = [15], y = [5],
									 scales = {'x': drop_x_sc, 'y': drop_y_sc},
									 #default_size = 20,
									 #default_opacities = [0.9],
									 font_weight = 'bolder',
									 #update_on_move = True,
									 colors = ['steelblue'])  

			drop_labels = {
				'inputs' : plt.Label(text = ['Step 0: Collection of inputs'],
									 x = [8.5], y = [2],
									 scales = {'x': drop_x_sc, 'y': drop_y_sc},
									 default_size = 30,
									 default_opacities = [0.9],
									 font_weight = 'bolder',
									 update_on_move = True,
									 colors = ['steelblue']),  

				'dotproduct1' : plt.Label(text = ['Step 1: Dot Products'],
										  x = [10.8], y = [2],
										  scales = {'x': drop_x_sc, 'y': drop_y_sc},
										  default_size = 30,
										  default_opacities = [0.9],
										  font_weight = 'bolder',
										  update_on_move = True,
										  colors = ['steelblue']),
				
				'dropout1' : plt.Label(text = ['Step 2: Dropout (p = 4/5)'],
										  x = [8.8], y = [2],
										  scales = {'x': drop_x_sc, 'y': drop_y_sc},
										  default_size = 30,
										  default_opacities = [0.9],
										  font_weight = 'bolder',
										  update_on_move = True,
										  colors = ['steelblue']),

				'activation1' : plt.Label(text = ['Step 3: Activations'],
										  x = [11.8], y = [2],
										  scales = {'x': drop_x_sc, 'y': drop_y_sc},
										  default_size = 30,
										  default_opacities = [0.9],
										  font_weight = 'bolder',
										  update_on_move = True,
										  colors = ['steelblue']),

				'dotproduct2' : plt.Label(text = ['Step 4: Dot Products'],
										  x = [7], y = [2],
										  scales = {'x': drop_x_sc, 'y': drop_y_sc},
										  default_size = 30,
										  default_opacities = [0.9],
										  font_weight = 'bolder',
										  update_on_move = True,
										  colors = ['steelblue']),
		
				'dropout2' : plt.Label(text = ['Step 5: Dropout (p = 2/3)'],
										  x = [5], y = [2],
										  scales = {'x': drop_x_sc, 'y': drop_y_sc},
										  default_size = 30,
										  default_opacities = [0.9],
										  font_weight = 'bolder',
										  update_on_move = True,
										  colors = ['steelblue']),

				'activation2' : plt.Label(text = ['Step 6: Activations'],
										  x = [11.8], y = [2],
										  scales = {'x': drop_x_sc, 'y': drop_y_sc},
										  default_size = 30,
										  default_opacities = [0.9],
										  font_weight = 'bolder',
										  update_on_move = True,
										  colors = ['steelblue']),
							}
			for x in drop_labels.keys():
				drop_labels[x].default_opacities = [0]
				drop_labels[x].opacity = [0]

		# Inputs
		if(True):
			size = 1
			square_x, square_y = create_square(size = size)
			# Inputs 1
			if(True):
				drop_input_1_1 = bq.Lines(x = square_x + 1.5, y = square_y + 17.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				drop_input_1_2 = bq.Lines(x = square_x + 1.5, y = square_y + 13.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				drop_input_1_3 = bq.Lines(x = square_x + 1.5, y = square_y + 9.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				drop_input_1_4 = bq.Lines(x = square_x + 1.5, y = square_y + 5.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				drop_input_1_5 = bq.Lines(x = square_x + 1.5, y = square_y + 1.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

			# Inputs 2
			if(True):
				drop_input_2_1 = bq.Lines(x = square_x + 1.5, y = square_y + 17.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				drop_input_2_2 = bq.Lines(x = square_x + 1.5, y = square_y + 13.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				drop_input_2_3 = bq.Lines(x = square_x + 1.5, y = square_y + 9.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				drop_input_2_4 = bq.Lines(x = square_x + 1.5, y = square_y + 5.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				drop_input_2_5 = bq.Lines(x = square_x + 1.5, y = square_y + 1.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

			# Inputs 3
			if(True):
				drop_input_3_1 = bq.Lines(x = square_x + 1.5, y = square_y + 17.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				drop_input_3_2 = bq.Lines(x = square_x + 1.5, y = square_y + 13.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				drop_input_3_3 = bq.Lines(x = square_x + 1.5, y = square_y + 9.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				drop_input_3_4 = bq.Lines(x = square_x + 1.5, y = square_y + 5.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

				drop_input_3_5 = bq.Lines(x = square_x + 1.5, y = square_y + 1.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										fill_colors = ['steelblue'])

		# Weights
		if(True):
			size = 1
			square_x, square_y = create_square(size = size)
			# Weights 1
			if(True):
				drop_weight_1_1 = bq.Lines(x = square_x + 19.5, y = square_y + 13.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				drop_weight_1_2 = bq.Lines(x = square_x + 19.5, y = square_y + 13.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				drop_weight_1_3 = bq.Lines(x = square_x + 19.5, y = square_y + 13.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				drop_weight_1_4= bq.Lines(x = square_x + 19.5, y = square_y + 13.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				drop_weight_1_5 = bq.Lines(x = square_x + 19.5, y = square_y + 13.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

			# Weights 2
			if(True):
				drop_weight_2_1 = bq.Lines(x = square_x + 19.5, y = square_y + 9.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				drop_weight_2_2 = bq.Lines(x = square_x + 19.5, y = square_y + 9.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				drop_weight_2_3 = bq.Lines(x = square_x + 19.5, y = square_y + 9.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				drop_weight_2_4= bq.Lines(x = square_x + 19.5, y = square_y + 9.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				drop_weight_2_5 = bq.Lines(x = square_x + 19.5, y = square_y + 9.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

			# Weights 3
			if(True):
				drop_weight_3_1 = bq.Lines(x = square_x + 19.5, y = square_y + 5.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				drop_weight_3_2 = bq.Lines(x = square_x + 19.5, y = square_y + 5.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				drop_weight_3_3 = bq.Lines(x = square_x + 19.5, y = square_y + 5.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				drop_weight_3_4= bq.Lines(x = square_x + 19.5, y = square_y + 5.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

				drop_weight_3_5 = bq.Lines(x = square_x + 19.5, y = square_y + 5.5,
										scales = {'x': drop_x_sc, 'y': drop_y_sc},
										fill = 'inside',
										colors = ['red'],
										fill_colors = ['red'])

			# Weights 4
			if(True):
				drop_weight_4_1 = bq.Lines(x = square_x + 37.5, y = square_y + 17,
											scales = {'x': drop_x_sc, 'y': drop_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				drop_weight_4_2 = bq.Lines(x = square_x + 37.5, y = square_y + 17,
											scales = {'x': drop_x_sc, 'y': drop_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				drop_weight_4_3 = bq.Lines(x = square_x + 37.5, y = square_y + 17,
											scales = {'x': drop_x_sc, 'y': drop_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

			# Weights 5
			if(True):
				drop_weight_5_1 = bq.Lines(x = square_x + 37.5, y = square_y + 12,
											scales = {'x': drop_x_sc, 'y': drop_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				drop_weight_5_2 = bq.Lines(x = square_x + 37.5, y = square_y + 12,
											scales = {'x': drop_x_sc, 'y': drop_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				drop_weight_5_3 = bq.Lines(x = square_x + 37.5, y = square_y + 12,
											scales = {'x': drop_x_sc, 'y': drop_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

			# Weights 6
			if(True):
				drop_weight_6_1 = bq.Lines(x = square_x + 37.5, y = square_y + 7,
											scales = {'x': drop_x_sc, 'y': drop_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				drop_weight_6_2 = bq.Lines(x = square_x + 37.5, y = square_y + 7,
											scales = {'x': drop_x_sc, 'y': drop_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				drop_weight_6_3 = bq.Lines(x = square_x + 37.5, y = square_y + 7,
											scales = {'x': drop_x_sc, 'y': drop_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

			# Weights 7
			if(True):
				drop_weight_7_1 = bq.Lines(x = square_x + 37.5, y = square_y + 2,
											scales = {'x': drop_x_sc, 'y': drop_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				drop_weight_7_2 = bq.Lines(x = square_x + 37.5, y = square_y + 2,
											scales = {'x': drop_x_sc, 'y': drop_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

				drop_weight_7_3 = bq.Lines(x = square_x + 37.5, y = square_y + 2,
											scales = {'x': drop_x_sc, 'y': drop_y_sc},
											fill = 'inside',
											colors = ['green'],
											fill_colors = ['green'])

		# Scatterplot of Perceptrons
		if(True):
			drop_perceptrons = plt.Scatter(x = [2, 2, 2, 2, 2,
											   20, 20, 20,
											   38, 38, 38, 38],
										  y = [2, 6, 10, 14, 18,
											   6, 10, 14,
											   2.5, 7.5, 12.5, 17.5],
										  size = [900, 900 , 900, 900, 900],
										  default_size = 900,
										  colors = ['steelblue','steelblue','steelblue','steelblue','steelblue',
													'red', 'red','red',
													'green', 'green', 'green', 'green'],
										  scales = {'x': drop_x_sc, 'y': drop_y_sc})

		# Figure
		if(True):
			drop_fig = bq.Figure(marks = [drop_input_1_1, drop_input_1_2, drop_input_1_3,
										   drop_input_1_5, drop_input_1_4,
										   drop_input_2_1, drop_input_2_2, drop_input_2_3,
										   drop_input_2_5, drop_input_2_4,
										   drop_input_3_1, drop_input_3_2, drop_input_3_3,
										   drop_input_3_5, drop_input_3_4,
										   drop_weight_1_1, drop_weight_1_2, drop_weight_1_3,
										   drop_weight_1_4, drop_weight_1_5,
										   drop_weight_2_1, drop_weight_2_2, drop_weight_2_3,
										   drop_weight_2_4, drop_weight_2_5,
										   drop_weight_3_1, drop_weight_3_2, drop_weight_3_3,
										   drop_weight_3_4, drop_weight_3_5,
										   drop_weight_4_1, drop_weight_4_2, drop_weight_4_3,
										   drop_weight_5_1, drop_weight_5_2, drop_weight_5_3,
										   drop_weight_6_1, drop_weight_6_2, drop_weight_6_3,
										   drop_weight_7_1, drop_weight_7_2, drop_weight_7_3,
										   drop_perceptrons,
										   prev_layer_label, drop_layer_label, next_layer_label,
										   drop_labels['inputs'], drop_labels['dotproduct1'],
										   drop_labels['dropout1'], drop_labels['activation1'],
										   drop_labels['dotproduct2'], drop_labels['dropout2'],
										   drop_labels['activation2']],
								  #axes = [drop_ax_x, drop_ax_y],
								  animation_duration = 1000,
								  title = 'Dense Layers with Dropout - Forward Pipeline',
								  background_style = {'fill': 'white'})

			drop_fig.layout.height = '475px'
			drop_fig.layout.width = '800px'
		
		# Animation
		if(True):
			def drop_anim(frame):
				if(frame == 0):
					# Labels
					if(True):
						for x in drop_labels.keys():
							drop_labels[x].default_opacities = [0]
							drop_labels[x].opacity = [0]

					# Initialisation
					if(True):
						square_x, square_y = create_square(size = 1)

						# Inputs 1
						if(True):
							drop_input_1_1.x = square_x + 1.5
							drop_input_1_1.y = square_y + 17.5

							drop_input_1_2.x = square_x + 1.5
							drop_input_1_2.y = square_y + 13.5

							drop_input_1_3.x = square_x + 1.5
							drop_input_1_3.y = square_y + 9.5

							drop_input_1_4.x = square_x + 1.5
							drop_input_1_4.y = square_y + 5.5

							drop_input_1_5.x = square_x + 1.5
							drop_input_1_5.y = square_y + 1.5

						# Inputs 2
						if(True):
							drop_input_2_1.x = square_x + 1.5
							drop_input_2_1.y = square_y + 17.5

							drop_input_2_2.x = square_x + 1.5
							drop_input_2_2.y = square_y + 13.5

							drop_input_2_3.x = square_x + 1.5
							drop_input_2_3.y = square_y + 9.5

							drop_input_2_4.x = square_x + 1.5
							drop_input_2_4.y = square_y + 5.5

							drop_input_2_5.x = square_x + 1.5
							drop_input_2_5.y = square_y + 1.5

						# Inputs 3
						if(True):
							drop_input_3_1.x = square_x + 1.5
							drop_input_3_1.y = square_y + 17.5

							drop_input_3_2.x = square_x + 1.5
							drop_input_3_2.y = square_y + 13.5

							drop_input_3_3.x = square_x + 1.5
							drop_input_3_3.y = square_y + 9.5

							drop_input_3_4.x = square_x + 1.5
							drop_input_3_4.y = square_y + 5.5

							drop_input_3_5.x = square_x + 1.5
							drop_input_3_5.y = square_y + 1.5

						# Weights 1   
						if(True):
							drop_weight_1_1.x = square_x + 19.5
							drop_weight_1_1.y = square_y  + 13.5

							drop_weight_1_2.x = square_x + 19.5
							drop_weight_1_2.y = square_y  + 13.5

							drop_weight_1_3.x = square_x + 19.5
							drop_weight_1_3.y = square_y  + 13.5

							drop_weight_1_4.x = square_x + 19.5
							drop_weight_1_4.y = square_y  + 13.5

							drop_weight_1_5.x = square_x + 19.5
							drop_weight_1_5.y = square_y  + 13.5

						# Weights 2  
						if(True):
							drop_weight_2_1.x = square_x + 19.5
							drop_weight_2_1.y = square_y  + 9.5

							drop_weight_2_2.x = square_x + 19.5
							drop_weight_2_2.y = square_y  + 9.5

							drop_weight_2_3.x = square_x + 19.5
							drop_weight_2_3.y = square_y  + 9.5

							drop_weight_2_4.x = square_x + 19.5
							drop_weight_2_4.y = square_y  + 9.5

							drop_weight_2_5.x = square_x + 19.5
							drop_weight_2_5.y = square_y  + 9.5

						# Weights 3   
						if(True):
							drop_weight_3_1.x = square_x + 19.5
							drop_weight_3_1.y = square_y  + 5.5

							drop_weight_3_2.x = square_x + 19.5
							drop_weight_3_2.y = square_y  + 5.5

							drop_weight_3_3.x = square_x + 19.5
							drop_weight_3_3.y = square_y  + 5.5

							drop_weight_3_4.x = square_x + 19.5
							drop_weight_3_4.y = square_y  + 5.5

							drop_weight_3_5.x = square_x + 19.5
							drop_weight_3_5.y = square_y  + 5.5

						# Weights 4   
						if(True):
							drop_weight_4_1.x = square_x + 37.5
							drop_weight_4_1.y = square_y  + 17

							drop_weight_4_2.x = square_x + 37.5
							drop_weight_4_2.y = square_y  + 17

							drop_weight_4_3.x = square_x + 37.5
							drop_weight_4_3.y = square_y  + 17

						# Weights 5   
						if(True):
							drop_weight_5_1.x = square_x + 37.5
							drop_weight_5_1.y = square_y  + 12

							drop_weight_5_2.x = square_x + 37.5
							drop_weight_5_2.y = square_y  + 12

							drop_weight_5_3.x = square_x + 37.5
							drop_weight_5_3.y = square_y  + 12

						# Weights 6   
						if(True):
							drop_weight_6_1.x = square_x + 37.5
							drop_weight_6_1.y = square_y  + 7

							drop_weight_6_2.x = square_x + 37.5
							drop_weight_6_2.y = square_y  + 7

							drop_weight_6_3.x = square_x + 37.5
							drop_weight_6_3.y = square_y  + 7

						# Weights 7   
						if(True):
							drop_weight_7_1.x = square_x + 37.5
							drop_weight_7_1.y = square_y  + 2

							drop_weight_7_2.x = square_x + 37.5
							drop_weight_7_2.y = square_y  + 2

							drop_weight_7_3.x = square_x + 37.5
							drop_weight_7_3.y = square_y  + 2


						drop_fig.animation_duration = 1000
					return

				if(frame == 1):
					if(True):
						square_x, square_y = create_square(size = 1)
						# Inputs 1
						if(True):
							drop_input_1_1.x = square_x + 6
							drop_input_1_1.y = square_y + 17.5

							drop_input_1_2.x = square_x + 6
							drop_input_1_2.y = square_y + 13.5

							drop_input_1_3.x = square_x + 6
							drop_input_1_3.y = square_y + 9.5

							drop_input_1_4.x = square_x + 6
							drop_input_1_4.y = square_y + 5.5

							drop_input_1_5.x = square_x + 6
							drop_input_1_5.y = square_y + 1.5

						# Inputs 2
						if(True):
							drop_input_2_1.x = square_x + 6
							drop_input_2_1.y = square_y + 17.5

							drop_input_2_2.x = square_x + 6
							drop_input_2_2.y = square_y + 13.5

							drop_input_2_3.x = square_x + 6
							drop_input_2_3.y = square_y + 9.5

							drop_input_2_4.x = square_x + 6
							drop_input_2_4.y = square_y + 5.5

							drop_input_2_5.x = square_x + 6
							drop_input_2_5.y = square_y + 1.5

						# Inputs 3
						if(True):
							drop_input_3_1.x = square_x + 6
							drop_input_3_1.y = square_y + 17.5

							drop_input_3_2.x = square_x + 6
							drop_input_3_2.y = square_y + 13.5

							drop_input_3_3.x = square_x + 6
							drop_input_3_3.y = square_y + 9.5

							drop_input_3_4.x = square_x + 6
							drop_input_3_4.y = square_y + 5.5

							drop_input_3_5.x = square_x + 6
							drop_input_3_5.y = square_y + 1.5

						# Weights 1   
						if(True):
							drop_weight_1_1.x = square_x + 19.5
							drop_weight_1_1.y = square_y  + 13.5

							drop_weight_1_2.x = square_x + 19.5
							drop_weight_1_2.y = square_y  + 13.5

							drop_weight_1_3.x = square_x + 19.5
							drop_weight_1_3.y = square_y  + 13.5

							drop_weight_1_4.x = square_x + 19.5
							drop_weight_1_4.y = square_y  + 13.5

							drop_weight_1_5.x = square_x + 19.5
							drop_weight_1_5.y = square_y  + 13.5

						# Weights 2  
						if(True):
							drop_weight_2_1.x = square_x + 19.5
							drop_weight_2_1.y = square_y  + 9.5

							drop_weight_2_2.x = square_x + 19.5
							drop_weight_2_2.y = square_y  + 9.5

							drop_weight_2_3.x = square_x + 19.5
							drop_weight_2_3.y = square_y  + 9.5

							drop_weight_2_4.x = square_x + 19.5
							drop_weight_2_4.y = square_y  + 9.5

							drop_weight_2_5.x = square_x + 19.5
							drop_weight_2_5.y = square_y  + 9.5

						# Weights 3   
						if(True):
							drop_weight_3_1.x = square_x + 19.5
							drop_weight_3_1.y = square_y  + 5.5

							drop_weight_3_2.x = square_x + 19.5
							drop_weight_3_2.y = square_y  + 5.5

							drop_weight_3_3.x = square_x + 19.5
							drop_weight_3_3.y = square_y  + 5.5

							drop_weight_3_4.x = square_x + 19.5
							drop_weight_3_4.y = square_y  + 5.5

							drop_weight_3_5.x = square_x + 19.5
							drop_weight_3_5.y = square_y  + 5.5

						# Weights 4   
						if(True):
							drop_weight_4_1.x = square_x + 37.5
							drop_weight_4_1.y = square_y  + 17

							drop_weight_4_2.x = square_x + 37.5
							drop_weight_4_2.y = square_y  + 17

							drop_weight_4_3.x = square_x + 37.5
							drop_weight_4_3.y = square_y  + 17

						# Weights 5   
						if(True):
							drop_weight_5_1.x = square_x + 37.5
							drop_weight_5_1.y = square_y  + 12

							drop_weight_5_2.x = square_x + 37.5
							drop_weight_5_2.y = square_y  + 12

							drop_weight_5_3.x = square_x + 37.5
							drop_weight_5_3.y = square_y  + 12

						# Weights 6   
						if(True):
							drop_weight_6_1.x = square_x + 37.5
							drop_weight_6_1.y = square_y  + 7

							drop_weight_6_2.x = square_x + 37.5
							drop_weight_6_2.y = square_y  + 7

							drop_weight_6_3.x = square_x + 37.5
							drop_weight_6_3.y = square_y  + 7

						# Weights 7   
						if(True):
							drop_weight_7_1.x = square_x + 37.5
							drop_weight_7_1.y = square_y  + 2

							drop_weight_7_2.x = square_x + 37.5
							drop_weight_7_2.y = square_y  + 2

							drop_weight_7_3.x = square_x + 37.5
							drop_weight_7_3.y = square_y  + 2

						# Label
						if(True):
							drop_labels['dotproduct1'].default_opacities = [0]
							drop_labels['dotproduct1'].opacity = [0]

							drop_labels['inputs'].default_opacities = [0.9]
							drop_labels['inputs'].opacity = [0.9]

					return

				if(frame == 2):
					if(True):
						square_x, square_y = create_square(size = 1)

						drop_input_1_1.x = square_x + 6
						drop_input_1_1.y = square_y + 13.5

						drop_input_1_2.x = square_x + 8.5
						drop_input_1_2.y = square_y + 13.5

						drop_input_1_3.x = square_x + 11
						drop_input_1_3.y = square_y + 13.5

						drop_input_1_4.x = square_x + 13.5
						drop_input_1_4.y = square_y + 13.5

						drop_input_1_5.x = square_x + 16
						drop_input_1_5.y = square_y + 13.5

						drop_weight_1_1.x = square_x + 7
						drop_weight_1_1.y = square_y + 13.5

						drop_weight_1_2.x = square_x + 9.5
						drop_weight_1_2.y = square_y + 13.5

						drop_weight_1_3.x = square_x + 12
						drop_weight_1_3.y = square_y + 13.5

						drop_weight_1_4.x = square_x + 14.5
						drop_weight_1_4.y = square_y + 13.5

						drop_weight_1_5.x = square_x + 17
						drop_weight_1_5.y = square_y + 13.5
					if(True):
						square_x, square_y = create_square(size = 1)

						drop_input_2_1.x = square_x + 6
						drop_input_2_1.y = square_y + 9.5

						drop_input_2_2.x = square_x + 8.5
						drop_input_2_2.y = square_y + 9.5

						drop_input_2_3.x = square_x + 11
						drop_input_2_3.y = square_y + 9.5

						drop_input_2_4.x = square_x + 13.5
						drop_input_2_4.y = square_y + 9.5

						drop_input_2_5.x = square_x + 16
						drop_input_2_5.y = square_y + 9.5

						drop_weight_2_1.x = square_x + 7
						drop_weight_2_1.y = square_y + 9.5

						drop_weight_2_2.x = square_x + 9.5
						drop_weight_2_2.y = square_y + 9.5

						drop_weight_2_3.x = square_x + 12
						drop_weight_2_3.y = square_y + 9.5

						drop_weight_2_4.x = square_x + 14.5
						drop_weight_2_4.y = square_y + 9.5

						drop_weight_2_5.x = square_x + 17
						drop_weight_2_5.y = square_y + 9.5
					if(True):
						square_x, square_y = create_square(size = 1)

						drop_input_3_1.x = square_x + 6
						drop_input_3_1.y = square_y + 5.5

						drop_input_3_2.x = square_x + 8.5
						drop_input_3_2.y = square_y + 5.5

						drop_input_3_3.x = square_x + 11
						drop_input_3_3.y = square_y + 5.5

						drop_input_3_4.x = square_x + 13.5
						drop_input_3_4.y = square_y + 5.5

						drop_input_3_5.x = square_x + 16
						drop_input_3_5.y = square_y + 5.5

						drop_weight_3_1.x = square_x + 7
						drop_weight_3_1.y = square_y + 5.5

						drop_weight_3_2.x = square_x + 9.5
						drop_weight_3_2.y = square_y + 5.5

						drop_weight_3_3.x = square_x + 12
						drop_weight_3_3.y = square_y + 5.5

						drop_weight_3_4.x = square_x + 14.5
						drop_weight_3_4.y = square_y + 5.5

						drop_weight_3_5.x = square_x + 17
						drop_weight_3_5.y = square_y + 5.5   

					# Labels 
					if(True):
						drop_labels['inputs'].default_opacities = [0]
						drop_labels['inputs'].opacity = [0]
						drop_labels['dropout1'].default_opacities = [0]
						drop_labels['dropout1'].opacity = [0]

						drop_labels['dotproduct1'].default_opacities = [0.9]
						drop_labels['dotproduct1'].opacity = [0.9]

					return

				if(frame == 3):
					if(True):
						# Frame Return:
						if(True):
							if(True):
								square_x, square_y = create_square(size = 1)

								drop_input_1_1.x = square_x + 6
								drop_input_1_1.y = square_y + 13.5

								drop_input_1_2.x = square_x + 8.5
								drop_input_1_2.y = square_y + 13.5

								drop_input_1_3.x = square_x + 11
								drop_input_1_3.y = square_y + 13.5

								drop_input_1_4.x = square_x + 13.5
								drop_input_1_4.y = square_y + 13.5

								drop_input_1_5.x = square_x + 16
								drop_input_1_5.y = square_y + 13.5

								drop_weight_1_1.x = square_x + 7
								drop_weight_1_1.y = square_y + 13.5

								drop_weight_1_2.x = square_x + 9.5
								drop_weight_1_2.y = square_y + 13.5

								drop_weight_1_3.x = square_x + 12
								drop_weight_1_3.y = square_y + 13.5

								drop_weight_1_4.x = square_x + 14.5
								drop_weight_1_4.y = square_y + 13.5

								drop_weight_1_5.x = square_x + 17
								drop_weight_1_5.y = square_y + 13.5
								
								drop_weight_1_1.opacities = [1]
								drop_weight_2_4.opacities = [1]
								drop_weight_3_5.opacities = [1]
							if(True):
								square_x, square_y = create_square(size = 1)

								drop_input_2_1.x = square_x + 6
								drop_input_2_1.y = square_y + 9.5

								drop_input_2_2.x = square_x + 8.5
								drop_input_2_2.y = square_y + 9.5

								drop_input_2_3.x = square_x + 11
								drop_input_2_3.y = square_y + 9.5

								drop_input_2_4.x = square_x + 13.5
								drop_input_2_4.y = square_y + 9.5

								drop_input_2_5.x = square_x + 16
								drop_input_2_5.y = square_y + 9.5

								drop_weight_2_1.x = square_x + 7
								drop_weight_2_1.y = square_y + 9.5

								drop_weight_2_2.x = square_x + 9.5
								drop_weight_2_2.y = square_y + 9.5

								drop_weight_2_3.x = square_x + 12
								drop_weight_2_3.y = square_y + 9.5

								drop_weight_2_4.x = square_x + 14.5
								drop_weight_2_4.y = square_y + 9.5

								drop_weight_2_5.x = square_x + 17
								drop_weight_2_5.y = square_y + 9.5
							if(True):
								square_x, square_y = create_square(size = 1)

								drop_input_3_1.x = square_x + 6
								drop_input_3_1.y = square_y + 5.5

								drop_input_3_2.x = square_x + 8.5
								drop_input_3_2.y = square_y + 5.5

								drop_input_3_3.x = square_x + 11
								drop_input_3_3.y = square_y + 5.5

								drop_input_3_4.x = square_x + 13.5
								drop_input_3_4.y = square_y + 5.5

								drop_input_3_5.x = square_x + 16
								drop_input_3_5.y = square_y + 5.5

								drop_weight_3_1.x = square_x + 7
								drop_weight_3_1.y = square_y + 5.5

								drop_weight_3_2.x = square_x + 9.5
								drop_weight_3_2.y = square_y + 5.5

								drop_weight_3_3.x = square_x + 12
								drop_weight_3_3.y = square_y + 5.5

								drop_weight_3_4.x = square_x + 14.5
								drop_weight_3_4.y = square_y + 5.5

								drop_weight_3_5.x = square_x + 17
								drop_weight_3_5.y = square_y + 5.5   


						square_x, square_y = create_square(size = 1)
						drop_input_1_1.y = square_y - 5
						drop_weight_1_1.y = square_y - 5

						drop_input_2_4.y = square_y - 5
						drop_weight_2_4.y = square_y - 5

						drop_input_3_5.y = square_y - 5
						drop_weight_3_5.y = square_y - 5
					
						# Labels 
						if(True):
							drop_labels['activation1'].default_opacities = [0]
							drop_labels['activation1'].opacity = [0]
							drop_labels['dotproduct1'].default_opacities = [0]
							drop_labels['dotproduct1'].opacity = [0]

							drop_labels['dropout1'].default_opacities = [0.9]
							drop_labels['dropout1'].opacity = [0.9]   
				
				if(frame == 4):
					if(True):
						square_x, square_y = create_square(size = 1)
						
						drop_weight_1_1.opacities = [0]
						drop_weight_2_4.opacities = [0]
						drop_weight_3_5.opacities = [0]
						# Inputs
						if(True):
							drop_input_1_1.x = square_x + 19.5
							drop_input_1_2.x = square_x + 19.5
							drop_input_1_3.x = square_x + 19.5
							drop_input_1_4.x = square_x + 19.5
							drop_input_1_5.x = square_x + 19.5

							drop_input_2_1.x = square_x + 19.5
							drop_input_2_2.x = square_x + 19.5
							drop_input_2_3.x = square_x + 19.5
							drop_input_2_4.x = square_x + 19.5
							drop_input_2_5.x = square_x + 19.5

							drop_input_3_1.x = square_x + 19.5
							drop_input_3_2.x = square_x + 19.5
							drop_input_3_3.x = square_x + 19.5
							drop_input_3_4.x = square_x + 19.5
							drop_input_3_5.x = square_x + 19.5

						# Weights
						if(True):
							square_x, square_y = create_square(size = 1)
							drop_weight_1_1.x = square_x + 22
							drop_weight_1_2.x = square_x + 22
							drop_weight_1_3.x = square_x + 22
							drop_weight_1_4.x = square_x + 22
							drop_weight_1_5.x = square_x + 19.5

							drop_weight_2_1.x = square_x + 22
							drop_weight_2_2.x = square_x + 22
							drop_weight_2_3.x = square_x + 22
							drop_weight_2_4.x = square_x + 22
							drop_weight_2_5.x = square_x + 19.5

							drop_weight_3_1.x = square_x + 22
							drop_weight_3_2.x = square_x + 22
							drop_weight_3_3.x = square_x + 22
							drop_weight_3_4.x = square_x + 22
							drop_weight_3_5.x = square_x + 19.5

							drop_weight_1_1.y = square_y + 13.5
							drop_weight_1_2.y = square_y + 13.5
							drop_weight_1_3.y = square_y + 13.5
							drop_weight_1_4.y = square_y + 13.5
							drop_weight_1_5.y = square_y + 13.5

							drop_weight_2_1.y = square_y + 9.5
							drop_weight_2_2.y = square_y + 9.5
							drop_weight_2_3.y = square_y + 9.5
							drop_weight_2_4.y = square_y + 9.5
							drop_weight_2_5.y = square_y + 9.5

							drop_weight_3_1.y = square_y + 5.5
							drop_weight_3_2.y = square_y + 5.5
							drop_weight_3_3.y = square_y + 5.5
							drop_weight_3_4.y = square_y + 5.5
							drop_weight_3_5.y = square_y + 5.5

							drop_weight_4_1.x = square_x + 37.5
							drop_weight_4_2.x = square_x + 37.5
							drop_weight_4_3.x = square_x + 37.5

							drop_weight_5_1.x = square_x + 37.5
							drop_weight_5_2.x = square_x + 37.5
							drop_weight_5_3.x = square_x + 37.5

							drop_weight_6_1.x = square_x + 37.5
							drop_weight_6_2.x = square_x + 37.5
							drop_weight_6_3.x = square_x + 37.5

							drop_weight_7_1.x = square_x + 37.5
							drop_weight_7_2.x = square_x + 37.5
							drop_weight_7_3.x = square_x + 37.5

						# Labels 
						if(True):
							drop_labels['dropout1'].default_opacities = [0]
							drop_labels['dropout1'].opacity = [0]
							drop_labels['dotproduct2'].default_opacities = [0]
							drop_labels['dotproduct2'].opacity = [0]  

							drop_labels['activation1'].default_opacities = [0.9]
							drop_labels['activation1'].opacity = [0.9]



					return

				if(frame == 5):
					if(True): 
						square_x, square_y = create_square(size = 1)

						drop_weight_1_1.opacities = [1]
						drop_weight_2_4.opacities = [1]
						drop_weight_3_5.opacities = [1]
						# Weights 1
						if(True):
							drop_weight_1_1.x = square_x + 29
							drop_weight_1_2.x = square_x + 29
							drop_weight_1_3.x = square_x + 29
							drop_weight_1_4.x = square_x + 29

							drop_weight_1_1.y = square_y + 17
							drop_weight_1_2.y = square_y + 12
							drop_weight_1_3.y = square_y + 7
							drop_weight_1_4.y = square_y + 2

						# Weights 2
						if(True):
							drop_weight_2_1.x = square_x + 31.5
							drop_weight_2_2.x = square_x + 31.5
							drop_weight_2_3.x = square_x + 31.5
							drop_weight_2_4.x = square_x + 31.5

							drop_weight_2_1.y = square_y + 17
							drop_weight_2_2.y = square_y + 12
							drop_weight_2_3.y = square_y + 7
							drop_weight_2_4.y = square_y + 2

						# Weights 3
						if(True):
							drop_weight_3_1.y = square_y + 17
							drop_weight_3_2.y = square_y + 12
							drop_weight_3_3.y = square_y + 7
							drop_weight_3_4.y = square_y + 2

							drop_weight_3_1.x = square_x + 34
							drop_weight_3_2.x = square_x + 34
							drop_weight_3_3.x = square_x + 34
							drop_weight_3_4.x = square_x + 34

						# Weights 4
						if(True):
							drop_weight_4_1.x = square_x + 30
							drop_weight_4_2.x = square_x + 32.5
							drop_weight_4_3.x = square_x + 35
							
							drop_weight_4_1.y = square_y + 17
							drop_weight_4_2.y = square_y + 17
							drop_weight_4_3.y = square_y + 17

						# Weights 5
						if(True):
							drop_weight_5_1.x = square_x + 30
							drop_weight_5_2.x = square_x + 32.5
							drop_weight_5_3.x = square_x + 35
							
							drop_weight_5_1.y = square_y + 12
							drop_weight_5_2.y = square_y + 12
							drop_weight_5_3.y = square_y + 12

						# Weights 6
						if(True):
							drop_weight_6_1.x = square_x + 30
							drop_weight_6_2.x = square_x + 32.5
							drop_weight_6_3.x = square_x + 35
							
							drop_weight_6_1.y = square_y + 7
							drop_weight_6_2.y = square_y + 7
							drop_weight_6_3.y = square_y + 7

						# Weights 7
						if(True):
							drop_weight_7_1.x = square_x + 30
							drop_weight_7_2.x = square_x + 32.5
							drop_weight_7_3.x = square_x + 35
							
							drop_weight_7_1.y = square_y + 2
							drop_weight_7_2.y = square_y + 2
							drop_weight_7_3.y = square_y + 2
						# Labels 
						if(True):
							drop_labels['activation1'].default_opacities = [0]
							drop_labels['activation1'].opacity = [0]
							drop_labels['dropout2'].default_opacities = [0]
							drop_labels['dropout2'].opacity = [0]

							drop_labels['dotproduct2'].default_opacities = [0.9]
							drop_labels['dotproduct2'].opacity = [0.9]       

					return

				if(frame == 6):
					if(True):
						# Frame Return
						if(True):
							square_x, square_y = create_square(size = 1)

							# Weights 1
							if(True):
								drop_weight_1_1.x = square_x + 29
								drop_weight_1_2.x = square_x + 29
								drop_weight_1_3.x = square_x + 29
								drop_weight_1_4.x = square_x + 29

								drop_weight_1_1.y = square_y + 17
								drop_weight_1_2.y = square_y + 12
								drop_weight_1_3.y = square_y + 7
								drop_weight_1_4.y = square_y + 2

							# Weights 2
							if(True):
								drop_weight_2_1.x = square_x + 31.5
								drop_weight_2_2.x = square_x + 31.5
								drop_weight_2_3.x = square_x + 31.5
								drop_weight_2_4.x = square_x + 31.5

								drop_weight_2_1.y = square_y + 17
								drop_weight_2_2.y = square_y + 12
								drop_weight_2_3.y = square_y + 7
								drop_weight_2_4.y = square_y + 2

							# Weights 3
							if(True):
								drop_weight_3_1.y = square_y + 17
								drop_weight_3_2.y = square_y + 12
								drop_weight_3_3.y = square_y + 7
								drop_weight_3_4.y = square_y + 2

								drop_weight_3_1.x = square_x + 34
								drop_weight_3_2.x = square_x + 34
								drop_weight_3_3.x = square_x + 34
								drop_weight_3_4.x = square_x + 34

							# Weights 4
							if(True):
								drop_weight_4_1.x = square_x + 30
								drop_weight_4_2.x = square_x + 32.5
								drop_weight_4_3.x = square_x + 35

								drop_weight_4_1.y = square_y + 17
								drop_weight_4_2.y = square_y + 17
								drop_weight_4_3.y = square_y + 17

							# Weights 5
							if(True):
								drop_weight_5_1.x = square_x + 30
								drop_weight_5_2.x = square_x + 32.5
								drop_weight_5_3.x = square_x + 35

								drop_weight_5_1.y = square_y + 12
								drop_weight_5_2.y = square_y + 12
								drop_weight_5_3.y = square_y + 12

							# Weights 6
							if(True):
								drop_weight_6_1.x = square_x + 30
								drop_weight_6_2.x = square_x + 32.5
								drop_weight_6_3.x = square_x + 35

								drop_weight_6_1.y = square_y + 7
								drop_weight_6_2.y = square_y + 7
								drop_weight_6_3.y = square_y + 7

							# Weights 7
							if(True):
								drop_weight_7_1.x = square_x + 30
								drop_weight_7_2.x = square_x + 32.5
								drop_weight_7_3.x = square_x + 35

								drop_weight_7_1.y = square_y + 2
								drop_weight_7_2.y = square_y + 2
								drop_weight_7_3.y = square_y + 2
						square_x, square_y = create_square(size = 1)
						drop_weight_1_1.y = square_y - 5
						drop_weight_4_1.y = square_y - 5

						drop_weight_2_3.y = square_y - 5
						drop_weight_5_3.y = square_y - 5

						drop_weight_3_2.y = square_y - 5
						drop_weight_6_2.y = square_y - 5

						drop_weight_7_2.y = square_y - 5
						drop_weight_2_4.y = square_y - 5
						
						 # Labels 
						if(True):
							drop_labels['activation2'].default_opacities = [0]
							drop_labels['activation2'].opacity = [0]
							drop_labels['dotproduct2'].default_opacities = [0]
							drop_labels['dotproduct2'].opacity = [0]

							drop_labels['dropout2'].default_opacities = [0.9]
							drop_labels['dropout2'].opacity = [0.9]   
							
				if(frame == 7):
					if(True): 
						square_x, square_y = create_square(size = 1)
						# Weights 1
						if(True):
							drop_weight_1_1.x = square_x + 37.5
							drop_weight_1_2.x = square_x + 37.5
							drop_weight_1_3.x = square_x + 37.5
							drop_weight_1_4.x = square_x + 37.5

						# Weights 2
						if(True):
							drop_weight_2_1.x = square_x + 37.5
							drop_weight_2_2.x = square_x + 37.5
							drop_weight_2_3.x = square_x + 37.5
							drop_weight_2_4.x = square_x + 37.5

						# Weights 3
						if(True):

							drop_weight_3_1.x = square_x + 37.5
							drop_weight_3_2.x = square_x + 37.5
							drop_weight_3_3.x = square_x + 37.5
							drop_weight_3_4.x = square_x + 37.5

						# Weights 4
						if(True):
							drop_weight_4_1.x = square_x + 40
							drop_weight_4_2.x = square_x + 40
							drop_weight_4_3.x = square_x + 40

						# Weights 5
						if(True):
							drop_weight_5_1.x = square_x + 40
							drop_weight_5_2.x = square_x + 40
							drop_weight_5_3.x = square_x + 40

						# Weights 6
						if(True):
							drop_weight_6_1.x = square_x + 40
							drop_weight_6_2.x = square_x + 40
							drop_weight_6_3.x = square_x + 40

						# Weights 7
						if(True):
							drop_weight_7_1.x = square_x + 40
							drop_weight_7_2.x = square_x + 40
							drop_weight_7_3.x = square_x + 40   

					# Labels
					if(True):
						drop_labels['dropout2'].default_opacities = [0]
						drop_labels['dropout2'].opacity = [0]

						drop_labels['activation2'].default_opacities = [0.9]
						drop_labels['activation2'].opacity = [0.9]

					return

		# Buttons
		if(True):
			drop_prev_button = widgets.Button(description = 'Previous',
											   disabled = True)
			drop_next_button = widgets.Button(description = 'Next',
											   disabled = False)
		
			drop_frame = {'value' : 0}

			def drop_prev_button_press(*args):
				if(drop_frame['value'] > 0):
					drop_frame['value'] += -1

				drop_anim(drop_frame['value'])
				if(drop_frame['value'] == 0):
					drop_prev_button.disabled = True
				if(drop_frame['value'] == 6):
					drop_next_button.disabled = False    
					
			def drop_next_button_press(*args):
				if(drop_frame['value'] < 7):
					drop_frame['value'] += 1
				drop_anim(drop_frame['value'])
				if(drop_frame['value'] == 1):
					drop_prev_button.disabled = False
				if(drop_frame['value'] == 7):
					drop_next_button.disabled = True

			drop_prev_button.on_click(drop_prev_button_press)
			drop_next_button.on_click(drop_next_button_press)
		
		# Display
		if(True):
			drop_anim(0)

			drop_buttons = widgets.HBox([drop_prev_button, drop_next_button])
			display(drop_fig,drop_buttons)
			
			
			
			
			
			
			
def show_one_operation_conv():

	# Convolution Product
	import time
	if(True):
		# Functions
		if(True):
			def square_mark(scale_x, scale_y, size = 1, x = 0, y = 0, fill_color = 'red', color = 'red'):
				square_x, square_y = create_square(size = size)
				mark = bq.Lines(x = square_x + x, y = square_y + y,
								scales = {'x' : scale_x, 'y': scale_y},
								fill = 'inside',
								opacities = [0.9, 0.9, 0.9, 0.9, 0.9],
								something = [0.9, 0.9, 0.9, 0.9, 0.9],
								colors = [color],
								fill_colors = [fill_color])
				return mark

		# Scales and Axes
		if(True):
			convprod_x_sc = plt.LinearScale(min = 0, max = 40)
			convprod_y_sc = plt.LinearScale(min = 0, max = 20)

			convprod_ax_x = bq.Axis(scale= convprod_x_sc)
			convprod_ax_y = bq.Axis(scale= convprod_y_sc, orientation = 'vertical')

		# Kernels
		if(True):
			size = 2
			square_x, square_y = create_square(size = size)
			kernel_x = 25
			kernel_y = 5
			# Kernel 1
			if(True):
				convprod_kernel_1 = []
				for i in range(3):
					for j in range(3):
						convprod_kernel_1 += [square_mark(x = kernel_x + size*j, y = kernel_y + (2-i)*size ,
														  size = size,
														  scale_x = convprod_x_sc, scale_y = convprod_y_sc,
														  fill_color = 'pink', color='red')]

		# Inputs
		if(True):
			size = 2
			square_x, square_y = create_square(size = size)
			input_x = 9
			input_y = 5
			# Input 1
			if(True):
				input1_x = input_x
				input1_y = input_y
				convprod_input_1 = []
				for i in range(3):
					for j in range(3):
						convprod_input_1 += [square_mark(x = input1_x + j*size, y = input1_y + (2-i)*size,
														 size = size,
														 scale_x = convprod_x_sc, scale_y = convprod_y_sc,
														 fill_color = '#D6EAF8', color = 'steelblue')]

		# Step Labels
		if(True):
			plus_x = np.arange(0, 8, 1)
			plus_x = 4.34 + plus_x*4.48
			plus_y = np.zeros(8) + 14.7
			convprod_step_labels = {
				'input' : plt.Label(text = ['Input Matrix'],
									x = [8.35],
									y = [3.5],
									colors = ['#3498DB'],
									scales = {'x': convprod_x_sc, 'y': convprod_y_sc},
									default_size = 22,
									update_on_move = True,
									font_weight = 'bolder'),
				
				'kernel' : plt.Label(text = ['Convolution Kernel'],
									x = [22],
									y = [3.5],
									colors = ['#C0392B'],
									scales = {'x': convprod_x_sc, 'y': convprod_y_sc},
									default_size = 22,
									update_on_move = True,
									font_weight = 'bolder'),

				
				'multiply' : plt.Label(text = ['Term by term Multiplication'],
									   x = [9.2],
									   y = [18],
									   colors = ['steelblue'],
									   scales = {'x': convprod_x_sc, 'y': convprod_y_sc},
									   default_size = 28,
									   update_on_move = True,
									   font_weight = 'bolder'),

				'sum' : plt.Label(text = ['Sum'],
								  x = [18.2],
								  y = [18],
								  colors = ['steelblue'],
								  scales = {'x': convprod_x_sc, 'y': convprod_y_sc},
								  default_size = 28,
								  update_on_move = True,
								  font_weight = 'bolder'),
				
				'plus' : plt.Label(text = ['+'] * 8,
								   x = plus_x,
								   y = plus_y,
								   x_offset = -8,
								   colors = ['steelblue'],
								   scales = {'x': convprod_x_sc, 'y': convprod_y_sc},
								   default_size = 28,
								   default_opacities = [0.9],
								   update_on_move = True,
								   font_weight = 'bolder'),
				
				'output' : plt.Label(text = ['Output'],
									 x = [17.3],
									 y = [18],
									 colors = ['steelblue'],
									 scales = {'x': convprod_x_sc, 'y': convprod_y_sc},
									 default_size = 28,
									 update_on_move = True,
									 font_weight = 'bolder')

			}
			convprod_step_labels_list = []

			for label in convprod_step_labels.keys():
				convprod_step_labels_list += [convprod_step_labels[label]]
		
		# Figure
		if(True):
			convprod_fig = bq.Figure(marks = convprod_input_1 + convprod_kernel_1 \
											+ convprod_step_labels_list \
										 ,
								 background_style = {'fill': 'white'},
								 title = 'Convolution Product',
								 #axes = [convprod_ax_x, convprod_ax_y],
								 animation_duration = 1000)

			convprod_fig.layout.height = '475px'
			convprod_fig.layout.width = '800px'
			
		# Animation
		if(True):
			convprod_reset_button = widgets.Button(description = 'Reset',
												   disabled = False)
			
			convprod_next_button = widgets.Button(description = 'Next',
												  disabled = False)
			def convprod_anim(frame):
				if(frame == 0):
					if(True):
						# Labels
						if(True):
							for label in convprod_step_labels_list:
								label.default_opacities = [0]
								label.opacity = [0]
							convprod_step_labels['input'].default_opacities = [1]
							convprod_step_labels['input'].opacity = [1]
							convprod_step_labels['kernel'].default_opacities = [1]
							convprod_step_labels['kernel'].opacity = [1]

						# Input and Kernel
						if(True):
							size = 2
							square_x, square_y = create_square(size = size)

							for i in range(3):
								for j in range(3):
									convprod_input_1[3*i + j].opacities = [1]
									convprod_kernel_1[3*i + j].opacities = [1]
							
							for i in range(3):
								for j in range(3):
									convprod_input_1[3*i + j].x = square_x + 9 + size*j
									convprod_input_1[3*i + j].y = square_y + 5 + size*i
							for i in range(3):
								for j in range(3):
									convprod_kernel_1[3*i + j].x = square_x + 25 + size*j
									convprod_kernel_1[3*i + j].y = square_y + 5  + size*i
					return
				if(frame == 1):
					if(True):
						# Label
						if(True):
							convprod_step_labels['multiply'].default_opacities = [1]
							convprod_step_labels['multiply'].opacity = [1]

						# Input and Kernel
						if(True):
							size = 1.4
							square_x, square_y = create_square(size = size)
							for i in range(3):
								for j in range(3):
									convprod_input_1[3*(2-i) + j].x = square_x + 0.7 + 3.2*size*(3*i + j)
									convprod_input_1[3*(2-i) + j].y = square_y + 14
									convprod_kernel_1[3*(2-i) + j].x = square_x + 0.7 + size + 3.2*size*(3*i + j)
									convprod_kernel_1[3*(2-i) + j].y = square_y + 14
									time.sleep(0.7)
									
						convprod_step_labels['input'].default_opacities = [0]
						convprod_step_labels['input'].opacity = [0]
						convprod_step_labels['kernel'].default_opacities = [0]
						convprod_step_labels['kernel'].opacity = [0]
					return
				if(frame == 2):
					if(True):
						# Labels
						if(True):
							convprod_step_labels['multiply'].default_opacities = [0]
							convprod_step_labels['multiply'].opacity = [0]

							time.sleep(0.5)

							convprod_step_labels['sum'].default_opacities = [1]
							convprod_step_labels['sum'].opacity = [1]

							convprod_step_labels['plus'].default_opacities = [1] * 8
							convprod_step_labels['plus'].opacity = [1] * 8

							time.sleep(2)

							convprod_step_labels['plus'].default_opacities = [0] * 8
							convprod_step_labels['plus'].opacity = [0] * 8

							time.sleep(1)

						# Input + Kernel
						if(True):
							size = 1.4
							square_x, square_y = create_square(size = size)
							for i in range(3):
								for j in range(3):
									convprod_input_1[3*i + j].x = square_x + 19.3
									convprod_input_1[3*i + j].y = square_y + 14
									convprod_kernel_1[3*i + j].x = square_x + 19.3 
									convprod_kernel_1[3*i + j].y = square_y + 14
					return
				if(frame == 3):
					if(True):
						# Labels
						if(True):
							convprod_step_labels['sum'].default_opacities = [0]
							convprod_step_labels['sum'].opacity = [0]

							time.sleep(0.5)

							convprod_step_labels['output'].default_opacities = [1]
							convprod_step_labels['output'].opacity = [1]

							time.sleep(0.5)

						# Input and Kernel
						if(True):
							size = 1.4
							square_x, square_y = create_square(size = size)
							for i in range(3):
								for j in range(3):
									convprod_input_1[3*i + j].opacities = [0]
									convprod_kernel_1[3*i + j].opacities = [0]
							
							convprod_kernel_1[-1].opacities = [1]
							
							for i in range(3):
								for j in range(3):
									convprod_input_1[3*i + j].x = square_x + 19.3
									convprod_input_1[3*i + j].y = square_y + 5.3
									convprod_kernel_1[3*i + j].x = square_x + 19.3 
									convprod_kernel_1[3*i + j].y = square_y + 5.3
					return
				
		# Buttons
		if(True):
			convprod_frame = {'value' : 0}

			def convprod_reset_button_press(*args):
				convprod_frame['value'] = 0
				convprod_anim(convprod_frame['value'])
				convprod_reset_button.disabled = True
				convprod_next_button.disabled = False

			def convprod_next_button_press(*args):
				convprod_reset_button.disabled = False
				if(convprod_frame['value'] < 3):
					convprod_frame['value'] += 1
					convprod_anim(convprod_frame['value'])
					
				if(convprod_frame['value'] == 3):
					convprod_next_button.disabled = True

			convprod_reset_button.on_click(convprod_reset_button_press)
			convprod_next_button.on_click(convprod_next_button_press)
			
		# Display
		if(True):
			convprod_anim(0)
			convprod_buttons = widgets.HBox([convprod_reset_button, convprod_next_button])
			display(convprod_fig, convprod_buttons)
        
