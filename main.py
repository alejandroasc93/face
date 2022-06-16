# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2 as cv
from deepface import DeepFace
from retinaface import RetinaFace


def search_face():
    original_image = cv.imread('C:\\Users\\Alejandro\\PycharmProjects\\faces\\img\\img.png')
    # faces = RetinaFace.extract_faces('C:\\Users\\Alejandro\\PycharmProjects\\faces\\img\\img.png')
    # emotions = []
    # for face in faces:
    #     obj = DeepFace.analyze(face, detector_backend='skip', actions=['emotion'])
    #     emotions.append(obj['dominant_emotion'])

    grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    detected_faces = face_cascade.detectMultiScale(grayscale_image)

    for face in detected_faces:
        obj = DeepFace.analyze(face,  actions=['emotion'])
        (column, row, width, height) = face
        cv.rectangle(
            original_image,
            (column, row),
            (column + width, row + height),
            (0, 255, 0),
            2
        )
        cv.putText(original_image, obj['dominant_emotion'], (column, row-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    # result = DeepFace.analyze(original_image, ['age', 'gender', 'race', 'emotion'])
    # print(result)
    cv.imshow('Image', original_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    search_face()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
