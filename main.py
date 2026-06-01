"""
Author: Alvise Favero (888851@stud.unive.it)
Academic Year: 2025/2026
Course: Geometric and 3D Computer Vision
"""

from collections.abc import Sequence
from dataclasses import dataclass
import cv2
import numpy as np
import open3d as o3d

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


K = np.loadtxt("./LaserScanner_project_data/calibration/K.txt")
DIST = np.loadtxt("./LaserScanner_project_data/calibration/dist.txt")


@dataclass
class Plane3D:
    tvec: cv2.typing.MatLike
    rmat: cv2.typing.MatLike
    normal: cv2.typing.MatLike


@dataclass
class Ray3D:
    point: cv2.typing.MatLike
    direction: cv2.typing.MatLike

    @classmethod
    def backproject(cls, point2d: cv2.typing.MatLike, k: cv2.typing.MatLike):
        point2d_homogeneous = np.append(point2d, 1).reshape(3, 1)
        p = np.zeros((3, 1))
        a = np.linalg.inv(k) @ point2d_homogeneous
        direction = a / np.linalg.norm(a)

        return Ray3D(p, direction)


def intersect_plane_ray(plane: Plane3D, ray: Ray3D) -> cv2.typing.MatLike:
    z = ((plane.tvec - ray.point).T @ plane.normal) / (ray.direction.T @ plane.normal)
    return ray.point + (z * ray.direction)


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


def find_rectangles(contours: Sequence[cv2.typing.MatLike], hierarchy):
    rectangular_contours = []

    # for i, c in zip(np.arange(len(contours)), contours):
    for c in contours:
        c1 = cv2.approxPolyDP(c, 15, True)
        # Keep only contours with 4 vertices
        if len(c1) == 4:
            c1 = sort_points(c1)
            # solvePnP requires float32 so we convert it here,
            # to make code more readable in the next lines
            c1 = c1.astype(np.float32)
            rectangular_contours.append(c1)

    # TODO: use hierarchy for more robust filtering and detection
    # This is really bad because it assumes the position of the rectangles
    # in the array. By using hierarchy it would be better: search
    # for the rectangles that have another rectangle inside
    r1 = np.concatenate([rectangular_contours[0], rectangular_contours[2]])
    r2 = np.concatenate([rectangular_contours[1], rectangular_contours[3]])

    return [r1, r2]


def fit_plane(points: list[cv2.typing.MatLike]) -> Plane3D:
    # Compute mean of each coordinate
    mean = np.mean(points, axis=0)
    y = points - mean
    cov_mat = sum([col @ col.T for col in y])

    eigvals, eigvects = np.linalg.eig(cov_mat)
    # Find index of minimum eigenvalue
    normal_i = np.argmin(eigvals)
    # Get associated eigenvector
    normal = eigvects[:, normal_i].reshape((3, 1))
    # Shh I'm not setting rmat
    return Plane3D(tvec=mean, rmat=np.zeros((3, 3)), normal=normal)


def extract_base_planes(frame: cv2.typing.MatLike) -> tuple[Plane3D, Plane3D]:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Experimented a bit with the threshold and epsilon for approximation
    _, thresholded = cv2.threshold(grayscale, 20, 255, cv2.THRESH_BINARY)
    # thresholded = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15)

    # Find contours
    contours, hierarchy = cv2.findContours(
        # thresholded, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_CCOMP
        thresholded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    rectangles = find_rectangles(contours, hierarchy)


    # FIXME currently broken, don't know why
    # Green color for the first rectangle
    # cv2.drawContours(frame, rectangles, -1, (0, 255, 0), 2)
    # Red color for the second rectangle
    # cv2.drawContours(frame, rectangles[1], -1, (0, 0, 255), 2)

    r1_succ, r1_rvec, r1_tvec = cv2.solvePnP(
            np.array(RECT).astype(np.float32),
            rectangles[0],
            # None is the camera distortion: we have already fixed it
            # previously
            K, None)

    r2_succ, r2_rvec, r2_tvec = cv2.solvePnP(
            np.array(RECT).astype(np.float32),
            rectangles[1],
            K, None)


    r1_rmat, _ = cv2.Rodrigues(r1_rvec)
    r2_rmat, _ = cv2.Rodrigues(r2_rvec)


    r1_normal =  r1_rmat @ np.array([0, 0, 1])
    r2_normal =  r2_rmat @ np.array([0, 0, 1])
    np.matrix

    return (
            Plane3D(r1_tvec, r1_rmat, r1_normal),
            Plane3D(r2_tvec, r2_rmat, r2_normal)
            )


def main():
    cap = cv2.VideoCapture("./LaserScanner_project_data/data/cup1.mp4")
    
    # Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(10, 3))

    # 3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    is_camera_resetted = False
    keep_running = True
    while cap.isOpened() and keep_running:
        ret, frame = cap.read()
        if not ret:
            return

        # Compensate camera distortion
        frame = cv2.undistort(frame, K, DIST)

        r1, r2 = extract_base_planes(frame)


        # Sanity check: reproject points back to the image
        r1_reprojection, _ = cv2.projectPoints(
            np.array(RECT, dtype=np.float32), 
            r1.rmat,
            r1.tvec,
            K, 
            np.zeros((4, 1), dtype=np.float32)
        )

        r2_reprojection, _ = cv2.projectPoints(
            np.array(RECT, dtype=np.float32), 
            r2.rmat,
            r2.tvec,
            K, 
            np.zeros((4, 1), dtype=np.float32)
        )

        for p in np.concatenate([r1_reprojection, r2_reprojection]):
            px, py = int(p[0][0]), int(p[0][1])
            cv2.circle(frame, (px, py), 5, (255, 0, 0), -1)


        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Lost 30 minutes here because I didn't know OpenCV used
        # HSV with H in [0, 179] and not [0, 360]...
        red_lower = np.array([165, 35, 80])
        red_upper = np.array([180, 255, 255])
        laser_mask = cv2.inRange(hsv, red_lower, red_upper)
        # This gives points (y, x) instead of (x, y)
        laser_y, laser_x = np.where(laser_mask > 0)

        laser_points_r1 = []
        laser_points_r2 = []
        # These are to be intersected with the laser plane
        laser_points_out = []

        for px, py in zip(laser_x, laser_y):
            # I don't know what would happen if I used all 4 vertices
            # (both inner and outer rectangle) and I don't want to discover
            # So I'm just limiting it to the inner square
            # (first 4 points, r1_reprojection[:4])
            if cv2.pointPolygonTest(r1_reprojection[:4], (px.item(), py.item()), False) == 1:
                # This used to be a check to see if the right points were
                # detected. Now instead the same points are reprojected
                # (see sanity check in lines below, after having found their
                # 3d coordinates)
                # cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)

                # Save points to be fitted in plane
                laser_points_r1.append((px, py))
            elif cv2.pointPolygonTest(r2_reprojection[:4], (px.item(), py.item()), False) == 1:
                # Same as above
                # cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)
                laser_points_r2.append((px, py))
            else:
                # If the laser is not in the rectangles, color it with a
                # dark red
                cv2.circle(frame, (px, py), 2, (0, 0, 160), -1)

                laser_points_out.append((px, py))

        # 3D points of the laser intersected on plane r1
        points_r1 = []
        for p in laser_points_r1:
            # Backproject laser points in a 3D ray
            ray = Ray3D.backproject(p, K)
            # Intersect the 3D ray with the plane found before
            p = intersect_plane_ray(r1, ray)
            points_r1.append(p)


        # 3D points of the laser intersected on plane r2
        points_r2 = []
        for p in laser_points_r2:
            ray = Ray3D.backproject(p, K)
            p = intersect_plane_ray(r2, ray)
            points_r2.append(p)

        # Yes, this array could have been filled directly in the two for loops
        # instead of filling two separate ones. But you never know, it might
        # be useful to keep points separate
        laser_points_3d = points_r1 + points_r2

        # Sanity check: reproject and draw laser points on the image
        # with a brighter red than points not intersecting
        if len(laser_points_3d) != 0:
            p_repr, _ = cv2.projectPoints(
                np.array(laser_points_3d, dtype=np.float32),
                # No need to translate or rotate the points
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                K,
                None
            )

            for p in p_repr:
                px, py = int(p[0][0]), int(p[0][1])
                cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)


            laser_plane = fit_plane(laser_points_3d)

            obj_points = []
            for p in laser_points_out:
                ray = Ray3D.backproject(p, K)
                p3d = intersect_plane_ray(laser_plane, ray)
                obj_points.append(p3d)
            obj_points = np.array(obj_points).reshape(len(obj_points), 3)

            pcd.points.extend(obj_points)

            # Visualize new points
            vis.update_geometry(pcd)
            keep_running = vis.poll_events()
            if not is_camera_resetted:
                is_camera_resetted = True
                vis.reset_view_point()
            vis.update_renderer()


        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
