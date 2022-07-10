import cv2
from random import randrange

# Loding some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

#### For detecting faces in video real time

webcam = cv2.VideoCapture(0)

while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()
    # Converting image to gray scale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (randrange(256), randrange(256), randrange(256)), 5)
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

#### Release video capture object
webcam.release()

"""
#### For detecting faces in a photo real time

# Choosing an image to detect faces in
img = cv2.imread('ChrisEvans.jpeg')
# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

# print(face_coordinates)
# Draw rectangle around faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256),
                  randrange(256), randrange(256)), 2)

# Show the image
cv2.imshow('Face Detector', img)

# waits until a key is Pressed
cv2.waitKey()
"""
print("Code Executed!")
