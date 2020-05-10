'''
Euclidean distance is define as
distance between two points in the space along the straight line. It is easy to 
calculate low computation cost.

To compare two faces are similar or not we use euclidean distance formula
to calculate similarity in the faces.

We can consider the faces are same if euclidean distance between two 
points are less than some pre set threshold. This threshold is used in previous 
step by Nerual network.
Face-recongintion library uses this threshold as 0.6.

This code will not work for the images that are samll, 
to make it work we will upsample the images and detect faces from that upsampled
image and then pass it generate face endcodings after this change system will recognise 
faces from small images.
'''

import face_recognition

# load the picture of known people

person1_image = face_recognition.load_image_file('images/images_for_recognition/person_1.jpg')
person2_image = face_recognition.load_image_file('images/images_for_recognition/person_2.jpg')
person3_image = face_recognition.load_image_file('images/images_for_recognition/person_3.jpg')

# getting 1st item from returned np array as there only be 1 image in the picture
person1_face_encodings = face_recognition.face_encodings(person1_image)[0]
person2_face_encodings = face_recognition.face_encodings(person2_image)[0]
person3_face_encodings = face_recognition.face_encodings(person3_image)[0]

# create a dataset of known images, lets make python list
known_faces_dataset = [
    person1_face_encodings,
    person2_face_encodings,
    person3_face_encodings
]

# load some unknown image and get its face encodings
unknown_image = face_recognition.load_image_file('images/images_for_recognition/unknown_7.jpg')

# resizing the images and extracting face loctions
unknown_face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=2)
# passing extracted loaction for get faceencoding of it
unknown_face_encodings = face_recognition.face_encodings(unknown_image,known_face_locations=unknown_face_locations)

#there could be multiple faces in the unknown image we have iterate thought each face encoding 
# and search though our know person datasets
for unknown_face_encoding in unknown_face_encodings:
    
    results = face_recognition.compare_faces(known_faces_dataset, unknown_face_encoding, tolerance=0.6)

    if results[0]:
        print("Found Person 1")
    elif results[1]:
        print("Found Person 2")
    elif results[2]:
        print("Found Person 3")
    else:
        print("Unknown")




