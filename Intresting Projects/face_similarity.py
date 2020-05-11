'''
This project will find the similar looking face of an input image
from the database of images

This could be extented as application which can file  nearest matching
celebrity form database of celebrity images

'''
import face_recognition
from pathlib import Path
import cv2


# user photo
known_image = face_recognition.load_image_file("images/test_face.jpg")

# extract encoding out of it
known_image_encoding = face_recognition.face_encodings(known_image)[0]

# print(known_image_encoding)
best_face_distance = 1.0
best_image = None

for image_path in Path("images/people").glob("*.png"):
    celebrity_image = face_recognition.load_image_file(image_path)
    celebrity_image_encoding = face_recognition.face_encodings(celebrity_image)
    # facedistance eucledien distance
    face_distance = face_recognition.face_distance(celebrity_image_encoding, known_image_encoding)[0]
    if face_distance < best_face_distance:
        best_face_distance = face_distance
        best_image = image_path
        print(best_image)
print("Best image path ", best_image)
image = cv2.imread(str(best_image))
print("Best Match distance ", best_face_distance)
# press q to see facial features of new face
cv2.imshow("Best Match", image)
k = cv2.waitKey(0)

