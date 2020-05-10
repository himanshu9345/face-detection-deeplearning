import cv2
import face_recognition
import numpy as np

# loading image into np array
# this library uses HOG to detect faces in a image
image = face_recognition.load_image_file("images/people.jpg")

# getting facial feature in an image
face_landmarks_list = face_recognition.face_landmarks(image)

print("Found {} face(s) in this image.".format(len(face_landmarks_list)))

image = cv2.imread("images/people.jpg")

for face_landmarks in face_landmarks_list:
    for name, list_of_points in face_landmarks.items():
        print("Feature detected is {}".format(name))
        cv2.polylines(image, [np.array(list_of_points)] \
            , isClosed=False, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    # press q to see facial features of new face
    cv2.imshow("Facial Feature", image)
    k = cv2.waitKey(0)
# dumping ouput to a image file
cv2.imwrite("images/step2and3_output.jpg", image)
