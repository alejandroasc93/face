import threading
import os
from pathlib import Path

import cv2
from deepface import DeepFace
from retinaface import RetinaFace

interval = 50
tmp_dir = f"{os.path.join(Path(os.path.abspath(__file__)).parent, 'tmp')}"
output_file = f"{os.path.join(os.path.join(Path(os.path.abspath(__file__)).parent, 'output'), 'result.txt')}"


if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)

if not os.path.exists(os.path.join(Path(os.path.abspath(__file__)).parent, 'output')):
    os.mkdir(os.path.join(Path(os.path.abspath(__file__)).parent, 'output'))



def search_face():
    face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier()
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print("Error loading xml file")

    cap = cv2.VideoCapture(0)
    c = 0
    while cap.isOpened():
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        c += 1
        if c % interval == 0:
            filename = f"{tmp_dir}\\{c}.jpg"
            cv2.imwrite(filename, frame)
            hilo = threading.Thread(target=search_emotion, args=(filename,))
            hilo.start()
            # search_emotion(filename)
        cv2.imshow('cap', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def search_emotion(file):
    print(file)
    faces = RetinaFace.extract_faces(file)
    for face in faces:
        obj = DeepFace.analyze(face, detector_backend='skip', actions=['emotion'])
        with open(output_file, 'a') as f:
            f.write(obj['dominant_emotion'] + '\n')
            f.close()
        os.remove(file)


if __name__ == '__main__':
    search_face()
