#importing the open source computer vision library
import cv2
from random import randrange


# Load some pre-trained data on face frontals 
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Choose an image to detect face in
img = cv2.imread("94.55.1.jpg")

# Must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Face
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Drawing Rectanble
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x, y), (w+x,h+y), (randrange(128,256),randrange(128,256),randrange(128,256)),2)

cv2.imshow("Bhavya Piyush Face Decetor",img)
print(face_coordinates)
# It pauses the execution of the code
cv2.waitKey()





print("Code execution is completed here! ")