import cv2
import sys


def main():
    imagePath = sys.argv[1]
    cascadePath = sys.argv[2]
    scaleFactor = float(sys.argv[3])
    minNeighbors = int(sys.argv[4])

    faceCascade = cv2.CascadeClassifier(cascadePath)
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor,
                minNeighbors,
                minSize=(30, 30)
            )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
