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

# use to retrieve the faces information
detector = dlib.get_frontal_face_detector()
# print(detector)


# function for face detecting and save the Face ROI(embedding)
# takes 2 parameter, imagepath = Uploaded image location, name = user name
def image_data_generator(imagePath,name):

  # setting up the path for saving the image
  path = 'database'
  # print(path) output -> path
  
  # folder for the user to store user image
  directory = os.path.join(path, name)
  # print(directory)  output -> path/name

  # Creating the folder for user if the user folder not exist
  if not os.path.exists(directory):
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
    
    # top, bottom, left, rigth = x, y, w, h
    # x = left(), y = top() 
    # w = right() - x, h = bottom() - y
    # roi - region of interest	
    roi_image = gray_image[d.top():d.top() + (d.bottom() - d.top()), d.left():d.left() +  (d.right() - d.left())]
    
    # saving the roi croped images
    cv2.imwrite(directory+'/'+name+".jpg",roi_image)

imagePath = 'faceDetect.jpg'
name = input("\nEnter name of person : ")
image_data_generator(imagePath, name)

# function for face detecting and save the Face ROI(embedding) from webcam
# takes 1 parameter, name = user name
def video_data_generator(name):

  # setting up the path for saving the image
  path = 'database'
  # print(path) output -> path
  
  # folder for the user to store user image
  directory = os.path.join(path, name)
  # print(directory)  output -> path/name

  # Creating the folder for user if the user folder not exist
  if not os.path.exists(directory):
	  os.makedirs(directory, exist_ok = 'True')
	  # print("\nDirectory with the name {} is created successful".format(name))

  # starting up the webcam
  webcam = cv2.VideoCapture(0)

  number_of_images = 0
  MAX_NUMBER_OF_IMAGES = 20


  while number_of_images < MAX_NUMBER_OF_IMAGES:
	
    # reading the data from the webcam
    ret, frame = webcam.read()

    # flips a 2D array around vertical, horizontal, or both axes
    # 1 means flipping around y-axis
    frame = cv2.flip(frame, 1)

    # converting the rgb frames to gray scale frames
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # The 1 in the second argument indicates that we should upsample the image 1 time.  This will make everything bigger and allow us to detect more faces.	
    faces = detector(frame, 1)
    #print(faces) # -> print the image value in array [(x,y)(w,h)]

    # adds a counter to an iterable and returns it in a form of enumerate object
    for i, d in enumerate(faces):

      # top, bottom, left, rigth = x, y, w, h
      # x = left(), y = top() 
      # w = right() - x, h = bottom() - y
      # roi - region of interest	
      roi_image = frame[d.top():d.top() + (d.bottom() - d.top()), d.left():d.left() +  (d.right() - d.left())]

      # saving the croped image 
      cv2.imwrite(os.path.join(directory, str(name+str(number_of_images)+'.jpg')), roi_image)
		
      number_of_images += 1

      cv2.rectangle(frame, (d.left(), d.top()), (d.left() + (d.right() - d.left()), d.top() + (d.bottom() - d.top())), (0, 255, 0), 2)
  
    # displaying the video
    cv2.imshow("Webcam",frame)

    # for closing the stream
    if(cv2.waitKey(1) & 0xFF == ord('q')):
      break

  # stoping the webcam
  webcam.release()

  # closing the window
  cv2.destroyAllWindows()

name = input("\nEnter name of person : ")
video_data_generator(name)
