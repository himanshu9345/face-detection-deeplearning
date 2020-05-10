import cv2
import face_recognition

# loading image into np array
image = face_recognition.load_image_file("images/people.jpg")

# getting face loactions
face_locations = face_recognition.face_locations(image)

number_of_faces = len(face_locations)

image = cv2.imread("images/people.jpg")
height, width, size = image.shape
for face_location in face_locations:
    top, right, bottom, left = face_location
    print(face_location)

    top_left = left, top
    bottom_right = right, bottom
    print(top_left, bottom_right)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
    cv2.imshow("output", image)
    k = cv2.waitKey(0)
