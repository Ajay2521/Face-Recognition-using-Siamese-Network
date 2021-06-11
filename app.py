#Import necessary libraries

# return a secure version of the user input file name
from werkzeug.utils import secure_filename

# Flask - Flask is an API of Python that allows us to build up web-applications.
# flash - used to generate informative messages in the flask
# request - used to gain access
# redirect - used to returns a response object and redirects the user to another target location
# url_for - used for creating a URL to prevent the overhead of having to change URLs throughout an application
# render_template - used to generate output from a template file based on the Jinja2 engine
# Response - container for the response data returned by application route functions, plus some additional information needed to create an HTTP response
from flask import Flask, flash, request, redirect, url_for, render_template, Response

# open cv for image processing
import cv2

# used to manipulate different parts
import sys

# used for manipulating array/matrix
import numpy as np

# used for accessing the file and folder in the machine
import os

# used for landmark's facial detector with pre-trained models, the dlib is used to estimate the location of 68 coordinates
import dlib

# VideoStream - used for video stream using webcam
from imutils.video import VideoStream

# imutils - used to make basic image processing functions such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images
import imutils

import tensorflow as tf

# used to load the json model file
from tensorflow.keras.models import model_from_json

# used to load the model
from tensorflow.keras.models import load_model

# act as a primary database where the user uploaded data is store and accessed for future works
UPLOAD_FOLDER = './static/uploaded_image'

# allowing the user to upload only certain file types
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Initializing the Flask app
app = Flask(__name__)

# setting up the upload folder to the app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# for making the user input to a secret
app.secret_key = "secret-key"

# Rejecting files greater than a specific amount
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# use to retrieve the faces information
detector = dlib.get_frontal_face_detector()
# print(detector)

# opening, reading and closing the json file
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()

# loading the json model using model_from_json
# model = model_from_json(loaded_model_json)

# loading the model weight
# model.load_weights('model.h5')

# loading the model from the folder faceNet
model = load_model('./faceNet')

# function which is used to encode the image if the image is at a specific location/imagepath
def img_path_to_encoding(image_path,model):

    # reading the image
    img = cv2.imread(image_path) # value while be of whole number
    # print('\nImage data :',img)
    # print("\nImage Data :",img.shape) # image size -> (223, 223, 3) -> (n,n,nc)
    # print("\nImage :\n",plt.imshow(img))->BGR image  with size (223, 223, 3)

    return img_to_encoding(img,model)

# function which is used to encode the image if the image is already loaded and read the image data
def img_to_encoding(image, model):

    # resizing the image
    img = cv2.resize(image, (160, 160))
    # print('\nImage resize :',img)
    # print("\nImage resize :",img.shape) # image size -> (160, 160, 3) -> (n,n,nc)
    # print("\nImage resize :\n",plt.imshow(img))->BGR image  with size (160, 160, 3)

    # converting the img data in the form of pixel values
    img = np.around(np.array(img) / 255.0, decimals=12)
    # print('\nImage pixel value :',img)
    # print("\nImage pixel shape :",img.shape) # (160, 160, 3) -> (n, n, nc)

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


# function used to load the face database which load the user name and encoded details of the user face
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
				# print('\nMin dist: ',min_dist,"\nUser_Name :",user_name)

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
		cv2.imwrite('static/uploaded_image/recognized_faces.jpg', image)

# function to recgonize the user face in the real time stream
def recognize_video():

	# loading the face_database
	face_database = {}
	face_database = load_database()

	# Starting the video stream from webcam using imutils lib
	vs = VideoStream(src=0).start()

	while True:

		# capturing the frames and reading it
		frame = vs.read()
		# print(frame) -> print the image value in array [n,n,nc]
		# plt.imshow(frame) -> displaying the image

		frame = imutils.resize(frame, width = 400)

		# converting the RGB Image into Gray scale Image
		# gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# print(gray_image) -> print the image value in array [n,n]
		# plt.imshow(gray_image) # -> displaying the image

		# faces = faceClassifier.detectMultiScale(gray_image)
		# print(faces)

		# detecting the faces in the image, which is similar to detectMultiScale()
		# The 1 in the second argument indicates that we should upsample the image 1 time.
		# This will make everything bigger and allow us to detect more faces.
		faces = detector(frame, 1)
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


		# convert the image format into streaming data and assign it to memory cache
		res,buffer = cv2.imencode('.jpg',frame)

		# converting to the byte data
		frame = buffer.tobytes()

		# for continuous frame to make a stream video
		yield (b'--frame\r\n'
		b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

		# breaking up the stream
		if(cv2.waitKey(1) & 0xFF == ord('q')):
			break


	# stoping the video stream
	vs.stream.release()

	# closing all the windows
	cv2.destoryAllWindows()

# function for checking the upload image extension
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# function for face detecting and save the Face ROI(embedding) for static images
# takes 2 parameter, imagepath = Uploaded image location, username = user name
def image_data_generator(imagePath,username):

  # setting up the path for saving the image
  path = 'static/database'
  # print(path) output -> path

  # folder for the user to store user image
  directory = os.path.join(path,username)
  # print(directory)  output -> path/name

  # Creating the folder for user if the user folder not exist
  if not os.path.exists(directory):
	  # exist_ok - Checks for the presence of the folder
	  # exist_ok = 'False' - Error is raised if the target directory already exists
	  # exist_ok = 'True' - Error exceptions will be ignored
	  os.makedirs(directory, exist_ok = 'True')
	  # print("\nDirectory with the name {} is created successful".format(name))

  # reading the uploaded image
  image = cv2.imread(imagePath)
  # print(image) -> print the image value in array [n,n,nc]
  # plt.imshow(image) -> displaying the image

  # converting the RGB Image into Gray scale Image
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # print(gray_image) -> print the image value in array [n,n]
  # plt.imshow(gray_image) # -> displaying the image

  # detecting the faces in the image, which is similar to detectMultiScale()
  # faces = face_cascade.detectMultiScale(gray_image)
  # print(faces)

  # The 1 in the second argument indicates that we should upsample the image 1 time.  This will make everything bigger and allow us to detect more faces.	
  faces = detector(gray_image, 1)
  #print(faces) # -> print the image value in array [(x,y)(w,h)]

  # adds a counter to an iterable and returns it in a form of enumerate object
  for i, d in enumerate(faces):

    # top, bottom, left, right = x, y, w, h
    # x = left(), y = top()
    # w = right() - x, h = bottom() - y
    # roi - region of interest
    roi_image = image[d.top():d.top() + (d.bottom() - d.top()), d.left():d.left() +  (d.right() - d.left())]

    # saving the roi cropped images
    cv2.imwrite(directory + '/' + username +".jpg",roi_image)


# function for face detecting and save the Face ROI(embedding) for real time images
# takes 1 parameter username = user name
def video_data_generator(username):

	# setting up the path for saving the image
	path = 'static/database'
	# print(path) output -> path

	# folder for the user to store user image
	directory = os.path.join(path, username)
	# print(directory)  output -> path/name

	# Creating the folder for user if the user folder not exist
	if not os.path.exists(directory):

	  	# exist_ok - Checks for the presence of the folder
	  	# exist_ok = 'False' - Error is raised if the target directory already exists
	  	# exist_ok = 'True' - Error exceptions will be ignored
		os.makedirs(directory, exist_ok = 'True')
		# print("\nDirectory with the name {} is created successful".format(name))

	# setting up the no. of image to detect and store in the database
	number_of_images = 1
	max_number_of_images = 50

	# Starting the video stream from webcam using imutils lib
	vs = VideoStream(src=0).start()

	while number_of_images <= max_number_of_images:

		# read the frame from the threaded videostream and resize it and have width of 400
		frame = vs.read()
		frame = imutils.resize(frame, width = 400)

		# The 1 in the second argument indicates that we should upsample the image 1 time.
		# This will make everything bigger and allow us to detect more faces
		faces =  detector(frame, 1)
		#print(detector(gray_frame, 1)) # -> print the image value in array [(x,y)(w,h)]

		# adds a counter to an iterable and returns it in a form of enumerate object
		for i, d in enumerate(faces):

			# top, bottom, left, right = x, y, w, h
			# x = left(), y = top()
			# w = right() - x, h = bottom() - y
			# roi - region of interest
			roi_image = frame[d.top():d.top() + (d.bottom() - d.top()), d.left():d.left() +  (d.right() - d.left())]

			# saving the cropped image
			cv2.imwrite(os.path.join(directory, str(username+str(number_of_images)+'.jpg')), roi_image)

			# drawing the boundary boxes for the real time detecting face
			cv2.rectangle(frame, (d.left(), d.top()), (d.left() + (d.right() - d.left()), d.top() + (d.bottom() - d.top())), (0, 255, 0), 2)

			number_of_images += 1


		# convert the image format into streaming data and assign it to memory cache
		res,buffer = cv2.imencode('.jpg',frame)

		# converting to the byte data
		frame = buffer.tobytes()

		# for continuous frame to make a stream video
		yield (b'--frame\r\n'
		b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

		# breaking up the stream
		if(cv2.waitKey(1) & 0xFF == ord('q')):
			break


	# stoping the video stream
	vs.stream.release()

	# closing all the windows
	cv2.destoryAllWindows()

# function for displaying the index page
@app.route('/')
def index():
    return render_template('index.html')

# function for displaying the registration page
@app.route('/register')
def register():
    return render_template('register.html')

# function for displaying the recognition page
@app.route('/recognition')
def recognition():
    return render_template('recognition.html')

# function for displaying the image page for uploading the user image into the database
@app.route('/uploadImageData')
def uploadImageData():
    return render_template('uploadImageData.html')

# once the upload & display button has been click it invoke the following function
# for storing the uploaded image to the database
@app.route('/uploadImageData', methods=['POST'])
def upload_image_data():

	# checking the presence of file
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)

	# checking whethere the uploaded file is of allowed file type
	if file and allowed_file(file.filename):

		# storing the file name and user name
		filename = secure_filename(file.filename)
		username = request.form.get("userName")

		# saving the uploaded file
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		flash('Image successfully uploaded and displayed below')

		# redirecting to the url with some information
		return render_template('uploadImageData.html', filename=filename, username=username)
	else:
		flash('Allowed image types are png, jpg, jpeg, gif')
		return redirect(request.url)

# displaying the uploaded user image which is store
@app.route('/uploadImageData/display/<filename>')
def display_uploaded_image(filename):
	return redirect(url_for('static' , filename="uploaded_image/" + filename ))

# displaying the user face which is detected and stored in the database
@app.route('/uploadImageData/detect/<username>/<filename>')
def display_database_image(filename,username):

	# setting up the path of the image for which the face as to be detected
	imagePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

	# calling up the function to generate the data for the uploaded image
	image_data_generator(imagePath,username)

	return redirect(url_for('static', filename='database/' + username + "/" + username + ".jpg"))

# function for displaying the live page for detecting the user real time face using the webcam
@app.route('/webcamVideoData')
def webcamVideoData():
    return render_template('webcamVideoData.html')

# function for getting the user name from the form
@app.route('/webcamVideoData', methods=['POST'])
def webcam_detect():
	username = request.form.get("userName")
	return render_template('webcamVideoData.html', username=username)

# function for generating the user face data for real time stream
@app.route('/webcamVideoData/<username>')
def webcam_face_detect(username):
	return Response(video_data_generator(username), mimetype='multipart/x-mixed-replace; boundary=frame')

# function redirect to the upload page where recognition using static image can be done
@app.route('/upload')
def upload():
  return render_template('upload.html')

# once the upload & predict button has been click it invoke the following function
@app.route('/upload', methods=['POST'])
def upload_image():
	# checking for the presence of the file
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)

	# checking the uploaded file type is of allowed file type
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)

		# saving the file
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename = filename)

	else:
		flash('Allowed image types are png, jpg, jpeg, gif')
		return redirect(request.url)

# displaying the uploaded user file
@app.route('/upload/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename= 'uploaded_image/' + filename))

# displaying the predicted image
@app.route('/upload/recognized/<filename>')
def recognized_image_display(filename):
	imagePath = os.path.join('static', 'uploaded_image', filename)
	recognize_image(imagePath)
	return redirect(url_for('static', filename='uploaded_image/recognized_faces.jpg' ))


# redirect to the live page
@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/webcam')
def webcam():
    return Response(recognize_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# running the flask app
if __name__ == "__main__":
    app.run(debug=True)