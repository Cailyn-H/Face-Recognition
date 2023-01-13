import face_recognition as fr
import cv2
import numpy as np

video = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
locations = []
encodings = []
face_name = []
encoded_face = []
recognized_names = ['your last name, your first name']
images = ['your last name, your first name.JPG']

#find images files
def names(lname_fname):
    image_name = '{}.JPG'.format(lname_fname)
    if image_name in images:
        return image_name
    else:
        images.append(image_name)
        print('Image Added')
        return image_name



#starts from here
print("Name (Lastname,Firstname): ")
input_name = input()
if input_name not in recognized_names:
    recognized_names.append(input_name)

for face_names in recognized_names:
    image = fr.load_image_file(names(face_names))
    encoded_face.append(fr.face_encodings(image)[0])


print("ready")
while True:
    # ret is boolean for all frame. Even though there is no frame, instead of throwing error, it will throw nothing
    ret,frame = video.read()

    resizedFrame = cv2.resize(frame, (0,0),fx=0.1, fy=0.1)
    convertedRGB = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)
    if ret:
    #A list of tuples of found face locations in css (top, right, bottom, left) order
        locations = fr.face_locations(convertedRGB)
    #A list of 128-dimensional face encodings (one for each face in the image)
        encodings = fr.face_encodings(convertedRGB, locations)

        face_name = []
        for faces in encodings:
            name = "UNKNOWN"
            matching = fr.compare_faces(encoded_face, faces)
            similarity = fr.face_distance(encoded_face, faces)
            matching_index = np.argmin(similarity)

            if matching[matching_index]:
                name = recognized_names[matching_index]
            face_name.append(name)

    ret = not ret
    # Display the results
    for (top, right, bottom, left), name in zip(locations, face_name):
            top *=10
            right *=10
            bottom *=10
            left *=10

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 5)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 248, 220), 3)

    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
            video.release()
            cv2.destroyAllWindows()




