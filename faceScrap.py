def face_scrap(imagePath):
    #imagePath contains the image from which the face needs to be extracted
    image = cv2.imread(imagePath)
    # print(image.shape) -> result -> (194, 259, 3)
    # plt.imshow(image) # displaying the image

    # converting the image into gray scale image
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(gray.shape) -> result -> (194, 259)
    # plt.imshow(gray) # displaying the image
    
    # using haarcascade_frontalface_default.xml for classfing(detection) the face 
    faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # print(faceClassfier)
    
    # detecting the multiple faces in the image
    # detectMultiScale - Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles
    faces = faceClassifier.detectMultiScale(grayImage)
    # print(faces) display the detected images x axis, y axis and image width and height
    # print("\nNo. of Faces Found :",len(faces)) # printing the No of faces detected using detectMultiScale
    
    #saving every face which is detected in previous steps
    for (x, y, w, h) in faces:
        # drawing a rectangle on any image 
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # roi - region of intreset using slicing the y and x axis along with height and width 
        roi_image = image[y:y + h, x:x + w]
        # saving the croped image
        cv2.imwrite('./images/'+str(w) + str(h) + '_faces.jpg', roi_image)

    # saving the detected image
    cv2.imwrite('detected_faces.jpg', image)

imagePath = './faceDetect.jpg'
face_scrap(imagePath)