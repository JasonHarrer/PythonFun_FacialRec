#!/usr/bin/env python

import cv2
import sys


def main():
    cascadePath = sys.argv[1]
    faceCascade = cv2.CascadeClassifier(cascadePath)

    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = faceCascade.detectMultiScale(
                    gray,
                    1.1,
                    5,
                    minSize=(30, 30)
                )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
