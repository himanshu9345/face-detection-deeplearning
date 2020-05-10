'''
Deep Metric Learning

How our face recongition systme know which feature are best to 
represent a face?
width of eye brows, or widht of eyes etc.
One thing that we can do is to compare all the features with all features of
other face but that will be slow.
So, how we can know which feature are important?.
We can assign this task to  deep convolustional neural network.
###############################
#####DEEP METRIC LEARNING######
###############################
This technique is called deep metric learning, where computer have to come 
up with the measurement which we know know how to measure.

To to used DML in face encoding, triplets of images are feed.
(image1,image2,image3) where image1 and 2 ar the image  of same person(alligned differnlty).
Neural network will represent each images as a set of 128 measurements and these measurments are
differnt for differnt faces. In the triplets Model will hearn how to keep measurment from two
picture same and far appart from 3rd image/person.

After going though lots of training with lots of images this model will have almost the
same merasurement for the same person and two picutre of two diffent person will have very differnt
measurements.

To do face encoding i have used pre trained model in face-recogniton library.

#####About 128 measurement from Neural network.#####
We dont know what 128 measurement(is it scalar reprenetation of length of eys lips, face etc,
may be or may be not) means since neural network suffer from
the problem of interpretability and it is very common.
'''

import face_recognition

image = face_recognition.load_image_file("images/people.jpg")

# face encodings
face_encodings = face_recognition.face_encodings(image)

# printing face encoding of first face
if face_encodings:
    first_face = face_encodings[0]
    print(first_face)