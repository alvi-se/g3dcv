"""
Author: Alvise Favero (888851@stud.unive.it)
Academic Year: 2025/2026
Course: Geometric and 3D Computer Vision
"""

import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture("./LaserScanner_project_data/data/cup1.mp4")

    while cap.isOpened():
        ret, frame = cap.read()

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Thresholding with random numbers (first attempts)
        _, thresholded = cv2.threshold(grayscale, 10, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, hierarchy = cv2.findContours(
            thresholded, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_CCOMP
        )
        img = cv2.drawContours(frame, contours, -1, (0, 255, 75), 2)

        cv2.imshow("frame", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
