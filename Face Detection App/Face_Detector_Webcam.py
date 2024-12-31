#importing the open source computer vision library
import cv2
from random import randrange


# Load some pre-trained data on face frontals 
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# To Capture video from webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    # Read the current frame
    # succesful_frame_read is a boolean and frame is the actual image
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Drawing Rectanble
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (w+x,h+y), (randrange(128,256),randrange(128,256),randrange(128,256)),2)

    cv2.imshow("Bhavya Piyush Face Decetor",frame)

    # It pauses the execution of the code
    # Automatically press a key after 1 millisecond
    key = cv2.waitKey(1)

    # Upper and lower case ASCII value of Q
    if key == 81 or key == 113:
        break

# Release the video capture object
webcam.release()
cv2.destroyAllWindows()



print("Code execution is completed here! ")