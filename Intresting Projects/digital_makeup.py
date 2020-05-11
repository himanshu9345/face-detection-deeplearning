import face_recognition
import cv2
import numpy as np


people_image  = face_recognition.load_image_file('images/people.jpg')

face_landmarks_list = face_recognition.face_landmarks(people_image)

image = cv2.imread('images/people.jpg')

for face_landmark in face_landmarks_list:
    cv2.drawContours(image, [np.array(face_landmark['top_lip'])], 0, (255, 255, 0), -1)
    cv2.drawContours(image, [np.array(face_landmark['bottom_lip'])], 0, (255, 255, 0), -1)

    cv2.drawContours(image, [np.array(face_landmark['left_eyebrow'])], 0, (0, 0, 0), -1)
    cv2.drawContours(image, [np.array(face_landmark['right_eyebrow'])], 0, (0, 0, 0), -1)

    # press q to see facial features of new face
    # cv2.imshow("Makeup", image)
    # k = cv2.waitKey(0)
cv2.imwrite("images/output/digital_makeup_output.jpg", image)