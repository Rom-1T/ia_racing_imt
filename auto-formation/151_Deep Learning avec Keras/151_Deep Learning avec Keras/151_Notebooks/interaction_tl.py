from IPython.display import display
import ipywidgets as widgets
from ipywidgets import Button, Layout, HTML, VBox, HBox

import bqplot.pyplot as plt
import cv2
import os
import matplotlib as mpl
from io import BytesIO
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import warnings


def show_generator():
	warnings.filterwarnings('ignore')

	box_layout = widgets.Layout(display='flex',
					flex_flow='column',
					align_items='center')
	button = widgets.Button(description = 'Generate', layout = box_layout)



			
	def button_press(*args):
		gen.image_data_generator.rotation_range = w1.value
		gen.image_data_generator.width_shift_range = w2.value
		gen.image_data_generator.height_shift_range = w3.value
	#     gen.image_data_generator.brightness_range = w4.value
		gen.image_data_generator.zoom_range = [w5.value, w5.value]
		gen.image_data_generator.horizontal_flip = w6.value=="True"
		gen.image_data_generator.vertical_flip = w7.value=="True"
		img_transform.value = arr2bytes(next(gen)[0].astype(np.uint8))


	button.on_click(button_press)


			
	w1 = widgets.FloatText(description = 'Rotation range', value = 10.0)
	w2 = widgets.FloatText(description = 'Width shift range', value = 50.0)
	w3 = widgets.FloatText(description = 'Height shift range', value = 50.0)
	# w4 = widgets.FloatText(description = 'Brightness range', value = 0.0)
	w5 = widgets.FloatText(description = 'Zoom range', value = 1.0)
	w6 = widgets.Dropdown(options=['False', 'True'],
												  value='False',
												  description='Horizontal Flip',
												  disabled = False)
	w7 = widgets.Dropdown(options=['False', 'True'],
												  value='False',
												  description='Vertical Flip',
												  disabled = False)


	kernel_grid = widgets.GridBox(children=[w1, w2 ,w3, w5, w6, w7], # , w4
								  layout=widgets.Layout(width='300 px',
																	grid_template_rows='30px 30px 30px',
																	grid_template_columns='',
																	grid_template_areas='''
																						"w1 w2 w3"
																						"w4 w5 w6"
																						"w7 w8 w9"'''))


	image = cv2.imread('python_keras_picasso_aubade.jpg')[:,:,[2,1,0]]
	def arr2bytes(p):
		with BytesIO() as buffer:
			mpl.image.imsave(buffer, p, format='jpg', vmin=0, vmax=255)
			out = buffer.getvalue()
		return out

	transformation = ImageDataGenerator(
			rotation_range = 10.0,
			width_shift_range = 50.0,
			height_shift_range = 50.0,
			zoom_range = [1, 1],
			horizontal_flip = False,
			vertical_flip= False
			)

	gen = transformation.flow(np.array([image]), batch_size=1)




	img_original = widgets.Image(value = arr2bytes(image),
									format = 'jpg',
									width="80%",
									height="80%")

	img_transform = widgets.Image(value = arr2bytes(next(gen)[0].astype(np.uint8)),
									format = 'jpg',
									width="80%",
									height="80%")


	h = HTML(value='<b>Original Image</b>')

	# Make the green box with the image widget inside it
	# Compose into a vertical box
	vb = VBox()
	vb.layout.align_items = 'center'
	vb.children = [h, img_original]

	h2 = HTML(value='<b>Image Generated</b>')

	# Make the green box with the image widget inside it
	# Compose into a vertical box
	vb2 = VBox()
	vb2.layout.align_items = 'center'
	vb2.children = [h2, img_transform]

	images_widget = widgets.HBox([vb, vb2])

	display(images_widget, kernel_grid, button)
	
	
	

from IPython.display import display
import ipywidgets as widgets
from ipywidgets import Button, Layout, VBox

def show_tl(path = './imgs'):
	prev_button = widgets.Button(description = 'Previous',
											   disabled = True, icon="backward", layout=Layout(width="80%", height="30%"))
	next_button = widgets.Button(description = 'Next',
											   disabled = False, icon="forward", layout=Layout(width="80%", height="30%"))


	dense_frame = {'value' : 0}

	def prev_button_press(*args):
		if(dense_frame['value'] > 0):
			dense_frame['value'] += -1
		with open('./imgs/'+img_path[dense_frame['value']], "rb") as f:
			img_widget.value = f.read()
		if(dense_frame['value'] == 0):
			prev_button.disabled = True
		if(dense_frame['value'] < N):
			next_button.disabled = False
			
	def next_button_press(*args):
		if(dense_frame['value'] < N):
			dense_frame['value'] += 1
		
		with open('./imgs/'+img_path[dense_frame['value']], "rb") as f:
			img_widget.value = f.read()

		if(dense_frame['value'] > 0):
			prev_button.disabled = False
		if(dense_frame['value'] == N):
			next_button.disabled = True

	import os
	img_path = sorted(os.listdir(path))
	N = len(img_path) - 1
	def path2byte(p):
		with open('./imgs/'+img_path[0], 'rb') as f:
			raw_image = f.read()
			return raw_image


	img_widget = widgets.Image(value = path2byte(img_path[dense_frame['value']]),
									format = 'jpg',
									width="90%",
									height="90%")

	vb = VBox()
	vb.layout.align_items = 'center'
	vb.children = [img_widget]

	prev_button.on_click(prev_button_press)
	next_button.on_click(next_button_press)

	buttons = widgets.HBox([prev_button, next_button])

	display(vb, buttons)
