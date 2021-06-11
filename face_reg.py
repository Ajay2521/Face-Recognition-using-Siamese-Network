# importing the neccessary libraries

# open cv for image processing
import cv2 

# used to manipulate different parts
import sys  

# used for manipulating array/matrics
import numpy as np

# used for accessing the file and folder in the machine
import os

# used for landmark's facial detector with pre-trained models, the dlib is used to estimate the location of 68 coordinates
import dlib

from imutils import face_utils

# for visulating the image
import matplotlib.pyplot as plt 

# It is a magic function that renders the figure in a notebook
# %matplotlib inline 

import tensorflow as tf

# used to load the json model file
from tensorflow.keras.models import model_from_json

# used to load the model 
from tensorflow.keras.models import load_model

# use to retrieve the faces information
detector = dlib.get_frontal_face_detector()
# print(detector)

# opening, reading and closing the json file
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()

# loading the json model using model_from_json
# model = model_from_json(loaded_model_json)

# loading the model weigth 
# model.load_weights('model.h5')

# loading the model from the folder faceNet
model = load_model('./faceNet')

def img_path_to_encoding(image_path,model):

    # reading the image
    img = cv2.imread(image_path) # value while be of whole number
    # print('\nImage data :',img)
    # print("\nImage Data :",img.shape) # image size -> (223, 223, 3) -> (n,n,nc)
    # print("\nImage :\n",plt.imshow(img))->BGR image  with size (223, 223, 3)

    return img_to_encoding(img,model)


def img_to_encoding(image, model):
  
    # resizing the image
    img = cv2.resize(image, dsize=(160, 160))
    # print('\nImage resize :',img)
    # print("\nImage resize :",img.shape) # image size -> (160, 160, 3) -> (n,n,nc)
    # print("\nImage resize :\n",plt.imshow(img))->BGR image  with size (160, 160, 3)

    # converting the img data in the form of pixcel values
    img = np.around(np.array(img) / 255.0, decimals=12) 
    # print('\nImage pixcel value :',img) 
    # print("\nImage pixcel shape :",img.shape) # (160, 160, 3) -> (n, n, nc)

    # expanding the dimension for making the image to fit for the input
    x_train = np.expand_dims(img, axis=0)
    # print('\nx_train :',x_train)
    # print("\nx_train shape :",x_train.shape) # (1, 160, 160, 3) input shape of model (ni, n, n, nc)

    # predicting the embedding of the image by passing the image to the model
    embedding = model.predict(x_train)
    # print('\nEmbedding :',embedding) # value in range of 0 to 9, + or - -> 0.18973544
    # print('\nEmbedding shape :',embedding.shape) # (1, 128) - 128 features 

    # predicting the embedding of the image by passing the image to the model
    # Euclidean distance is the shortest between the 2 points
    # ord =  Order of the norms

    embedding = embedding / np.linalg.norm(embedding, ord=2)
    # print('\nEmbedding :',embedding) # value in range or 0.00 to 0.19 -> 0.01639363
    # print('\nEmbedding shape :',embedding.shape) # (1, 128)

    return embedding

def load_database():
	# importing the database in the program as a dict datatype
	face_database = {}
	
	# listdir - used to get the list of all files and directories in the specified directory.
	for folder_name in os.listdir('static/database'):
		
		# print('\nfolder_name :',folder_name) # base folder name(user name)
		# path.join - concatenates various path components with exactly one directory separator ('/')
		
		for image_name in os.listdir(os.path.join('static/database',folder_name)): # database/folder_name
			# print('\nimage_name :',image_name) # image name with extension
			
			# splitext - used to split the path name into a pair root and extension.
			# basename - used to get the base name in specified path
			user_name = os.path.splitext(os.path.basename(image_name))[0]
			# print('\nUser name : ',user_name) # image name with out extension
			
			# img_path_to_encoding - used to get the face embedding for a image
			face_database[user_name] = img_path_to_encoding(os.path.join('static/database',folder_name,image_name), model)
	
	# print(face_database)
	return face_database

def recognize_image(imagePath):
	
	# loading the face_database
	face_database = {}
	face_database = load_database()
	
	# reading the uploaded image from ImagePath
	image = cv2.imread(imagePath)
	# print(image) -> print the image value in array [n,n,nc]
	# plt.imshow(image) -> displaying the image
	
	# converting the RGB Image into Gray scale Image
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# print(gray_image) -> print the image value in array [n,n]
	# plt.imshow(gray_image) # -> displaying the image
	
	# faces = faceClassifier.detectMultiScale(gray_image)
	# print(faces)
	
	# detecting the faces in the image, which is similar to detectMultiScale()
	# The 1 in the second argument indicates that we should upsample the image 1 time.  
	# This will make everything bigger and allow us to detect more faces.	
	faces = detector(gray_image, 1)
	# print(faces) # -> print the image value in array [(x,y)(w,h)]
	
	# adds a counter to an iterable and returns it in a form of enumerate object
	for i, d in enumerate(faces):
		
		# top, bottom, left, right = x, y, w, h
		# x = left(), y = top()
		# w = right() - x, h = bottom() - y
		# roi - region of interest
		roi_image = image[d.top():d.top() + (d.bottom() - d.top()), d.left():d.left() +  (d.right() - d.left())]
		
		# encoding the faces in new image 
		encoding = img_to_encoding(roi_image, model)
		# print("\nEncoding :\n",encoding)
		min_dist = 10
		
		for(username, encoded_image_name) in face_database.items():
			
			# calculating the Euclidean distance which is the shortest between the 2 points
			dist = np.linalg.norm(encoding - encoded_image_name)
			
			if(dist < min_dist):
				min_dist = dist
				user_name = username
				print('\nMin dist: ',min_dist,"\nUser_Name :",user_name)
		
		# if min_dist is high then it denoted the person face is not in the database
		if min_dist > 0.8:
			
			# drawing the boundary boxes for the real time detecting face
			cv2.rectangle(image, (d.left(), d.top()), (d.left() + (d.right() - d.left()), d.top() + (d.bottom() - d.top())), (0, 0, 255), 2)
			cv2.putText(image, 'Unknown', (d.left(), d.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
		
		# for less min_dist the person face is in database
		else:
			
			# drawing the boundary boxes for the real time detecting face
			cv2.rectangle(image, (d.left(), d.top()), (d.left() + (d.right() - d.left()), d.top() + (d.bottom() - d.top())), (0, 255, 0), 2)
			cv2.putText(image, user_name[:-3], (d.left(), d.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
			
		# saving the image
		cv2.imwrite('recognized_faces.jpg', image)

imagePath = './Test & Output/test.jpg'
recognize_image(imagePath)

def recognize_video():
	
	# loading the face_database
	face_database = {}
	face_database = load_database()
	
	# starting up the stream
	video_capture = cv2.VideoCapture(0)
	
	while True:
		
		# capturing the frames and reading it
		ret, frame = video_capture.read()
		# print(frame) -> print the image value in array [n,n,nc]
		# plt.imshow(frame) -> displaying the image
		
		# converting the RGB Image into Gray scale Image
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# print(gray_image) -> print the image value in array [n,n]
		# plt.imshow(gray_image) # -> displaying the image
		
		# faces = faceClassifier.detectMultiScale(gray_image)
		# print(faces)
		
		# detecting the faces in the image, which is similar to detectMultiScale()
		# The 1 in the second argument indicates that we should upsample the image 1 time.  
		# This will make everything bigger and allow us to detect more faces.	
		faces = detector(gray_frame, 1)
		# print(faces) # -> print the image value in array [(x,y)(w,h)]
		
		# adds a counter to an iterable and returns it in a form of enumerate object
		for i, d in enumerate(faces):
			
			# top, bottom, left, right = x, y, w, h
			# x = left(), y = top()
			# w = right() - x, h = bottom() - y
			# roi - region of interest
			roi_frame = frame[d.top():d.top() + (d.bottom() - d.top()), d.left():d.left() +  (d.right() - d.left())]
			
			# encoding the faces in new image 
			encoding = img_to_encoding(roi_frame, model)
			# print("\nEncoding :\n",encoding)
			min_dist = 100
			
			for(username, encoded_image_name) in face_database.items():
				
				# calculating the Euclidean distance which is the shortest between the 2 points
				dist = np.linalg.norm(encoding - encoded_image_name)
				
				if(dist < min_dist):
					min_dist = dist
					user_name = username
					print('\nMin dist: ',min_dist,"\nUser_Name :",user_name)
					
			# if min_dist is high then it denoted the person face is not in the database
			if min_dist > 0.8:
				
				# drawing the boundary boxes for the real time detecting face
				cv2.rectangle(frame, (d.left(), d.top()), (d.left() + (d.right() - d.left()), d.top() + (d.bottom() - d.top())), (0, 0, 255), 2)
				cv2.putText(frame, 'Unknown', (d.left(), d.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
				
			# for less min_dist the person face is in database
			else:
				
				# drawing the boundary boxes for the real time detecting face
				cv2.rectangle(frame, (d.left(), d.top()), (d.left() + (d.right() - d.left()), d.top() + (d.bottom() - d.top())), (0, 255, 0), 2)
				cv2.putText(frame, user_name[:-3], (d.left(), d.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
		
		# displaying the frame
		cv2.imshow('recognized_faces', frame)
		
		if(cv2.waitKey(1) & 0xFF == ord('q')):
			break
			
	video_capture.release()
	cv2.destroyAllWindows()

recognize_video()