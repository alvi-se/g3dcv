"""
Author: Alvise Favero (888851@stud.unive.it)
Academic Year: 2025/2026
Course: Geometric and 3D Computer Vision
"""

import cv2
import numpy as np

INNER_RECT = [[0.01, 0.01, 0.],
              [0.01, 0.14, 0.],
              [0.24, 0.14, 0.],
              [0.24, 0.01, 0.]
              ]

OUTER_RECT = [[0., 0., 0.],
              [0, 0.15, 0.],
              [0.25, 0.15, 0.],
              [0.25, 0., 0.]
              ]

RECT = INNER_RECT + OUTER_RECT


def sort_points(points):
    if len(points) != 4:
        raise Exception("Can only sort quadrilaterals")
    # Contours are 3-dimensional array of dimensions (4, 1, 2)
    points = points.reshape(4, 2)
    # Highest value is the bottom right vertex, lowest value is the top left
    d1 = sorted(points, key=lambda p: p[0] + p[1])
    # Highest value is the bottom left vertex, lowest value is the top right
    d2 = sorted(points, key=lambda p: p[0] - p[1])

    # "Clockwise" order of vertices
    return np.array([d1[0], d2[0], d1[-1], d2[-1]])


def main():
    cap = cv2.VideoCapture("./LaserScanner_project_data/data/puppet.mp4")
    
    k = np.loadtxt("./LaserScanner_project_data/calibration/K.txt")
    dist = np.loadtxt("./LaserScanner_project_data/calibration/dist.txt")

    while cap.isOpened():
        ret, frame = cap.read()

        # Compensate camera distortion
        frame = cv2.undistort(frame, k, dist)

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Experimented a bit with the threshold and epsilon for approximation
        _, thresholded = cv2.threshold(grayscale, 20, 255, cv2.THRESH_BINARY)
        # thresholded = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15)

        # Find contours
        contours, hierarchy = cv2.findContours(
            # thresholded, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_CCOMP
            thresholded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        rectangular_contours = []

        # for i, c in zip(np.arange(len(contours)), contours):
        for c in contours:
            c1 = cv2.approxPolyDP(c, 15, True)
            # Keep only contours with 4 vertices
            if len(c1) == 4:
                c1 = sort_points(c1)
                rectangular_contours.append(c1)

        
        img = cv2.drawContours(frame, [rectangular_contours[0], rectangular_contours[1]], -1, (0, 255, 75), 2)
        img = cv2.drawContours(img, [rectangular_contours[2], rectangular_contours[3]], -1, (0, 0, 255), 2)
        
        a = cv2.solvePnP(
                np.array(RECT).astype(np.float32),
                np.concatenate([
                    rectangular_contours[0].astype(np.float32),
                    rectangular_contours[1].astype(np.float32)
                    ]),
                k, None)

        cv2.imshow("frame", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
