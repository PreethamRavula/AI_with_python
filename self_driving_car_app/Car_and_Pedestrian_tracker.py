import cv2
from cv2 import COLOR_BGR2GRAY

# Realtime Tracking
video = cv2.VideoCapture('detection_video.mp4')

# Pre-trained car classifier
car_file = 'car_detector.xml'
pedestrian_file = 'haarcascade_fullbody.xml'

# Creating a Car classifier
car_tracker = cv2.CascadeClassifier(car_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_file)

while True:

    # Read the current frame
    (successful_read, frame) = video.read()

    if successful_read:
        # Convert the frame to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        break

    # Detect Cars and Pedestrians
    cars_video = car_tracker.detectMultiScale(grayscale_frame)
    pedestrian_video = pedestrian_tracker.detectMultiScale(grayscale_frame)

    # Font for writing over the rectangle
    font = cv2.FONT_HERSHEY_DUPLEX

    # Draw rectangles around cars
    for (x, y, w, h) in cars_video:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'Car', (x + 6, y - 6), font, 0.5, (0, 0, 255), 2)

    # Draw rectangles arounf pedestrians
    for (a, b, c, d) in pedestrian_video:
        cv2.rectangle(frame, (a, b), (a + c, b + d), (0, 255, 255), 2)
        cv2.putText(frame, 'Person', (a + 6, b - 6), font, 0.5, (0, 255, 255), 2)

    # Displaying the Frame
    cv2.imshow('Car and Pedestrian Detector Realtime', frame)

    # Don't autoclose
    key = cv2.waitKey(1)

    if key == 83 or key == 113:
        break


# Code for tracking a car in an Image
"""
# Importing Image
img_file = 'cars.jpeg'

# create opencv Image
img = cv2.imread(img_file)

# Converting image to gray scale
img_grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Cars
cars = car_tracker.detectMultiScale(img_grayScale)
# print(cars)

# Draw rectangles around cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)

# Display image with car spotted
cv2.imshow('Car Detector', img)

# Wait until a key is pressed
cv2.waitKey()
"""
print('Code Executed!!')
