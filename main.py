"""
Author: Alvise Favero (888851@stud.unive.it)
Academic Year: 2025/2026
Course: Geometric and 3D Computer Vision
"""

from collections.abc import Sequence
from dataclasses import dataclass
import logging
import time
import cv2
import numpy as np
import open3d as o3d
import os
import sys

logging.basicConfig(level=logging.INFO)


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

    # "Counter clockwise" order of vertices, where the first is the upper left
    return np.array([d1[0], d2[0], d1[-1], d2[-1]]).reshape((4, 1, 2))


def find_rectangles(contours: Sequence[cv2.typing.MatLike], hierarchy) -> tuple[cv2.typing.MatLike]:
    filtered_contours = {}
    rectangular_contours = []

    for c, i in zip(contours, range(len(contours))):
        c1 = cv2.approxPolyDP(c, 15, True)
        # Keep only contours with 4 vertices
        if len(c1) == 4:
            c1 = sort_points(c1)
            # solvePnP requires float32 so we convert it here,
            # to make code more readable in the next lines
            c1 = c1.astype(np.float32)
            filtered_contours[i] = c1

    for k, v in filtered_contours.items():
        # Hierarchy is [next, prev, child, parent]
        # So hierarchy[2] != -1 means "if contour has child contour"
        if hierarchy[0, k, 2] != -1:
            # First add the child contour (inner rectangle)
            rectangular_contours.append(filtered_contours[hierarchy[0, k, 2]])
            rectangular_contours.append(v)

    r1 = np.concatenate([rectangular_contours[0], rectangular_contours[1]])
    r2 = np.concatenate([rectangular_contours[2], rectangular_contours[3]])

    return (r1, r2)



def extract_base_planes(frame: cv2.typing.MatLike, thresholded: cv2.typing.MatLike) -> tuple[Plane3D, Plane3D, cv2.typing.MatLike, cv2.typing.MatLike]:
    # Find contours
    contours, hierarchy = cv2.findContours(
        # thresholded, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_CCOMP
        thresholded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    rectangles = find_rectangles(contours, hierarchy)


    # Green color for the first rectangle
    cv2.drawContours(frame, [
            rectangles[0][0:4].astype(np.int32),
            rectangles[0][4:8].astype(np.int32),
        ]
        , -1, (0, 255, 0), 2)
    # Red color for the second rectangle
    cv2.drawContours(frame, [
            rectangles[1][0:4].astype(np.int32),
            rectangles[1][4:8].astype(np.int32),
        ]
        , -1, (0, 0, 255), 2)

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

    return (
            Plane3D(r1_tvec, r1_rmat, r1_normal),
            Plane3D(r2_tvec, r2_rmat, r2_normal),
            *rectangles
            )


def fit_plane(points: list[cv2.typing.MatLike]) -> Plane3D:
    # Compute mean of each coordinate
    mean = np.mean(points, axis=0)
    y = points - mean
    cov_mat = sum([col @ col.T for col in y])

    # Optimize with eigh: https://stackoverflow.com/questions/45434989/numpy-difference-between-linalg-eig-and-linalg-eigh
    eigvals, eigvects = np.linalg.eigh(cov_mat)
    # Find index of minimum eigenvalue
    normal_i = np.argmin(eigvals)
    # Get associated eigenvector
    normal = eigvects[:, normal_i].reshape((3, 1))
    # Shh I'm not setting rmat
    return Plane3D(tvec=mean, rmat=np.zeros((3, 3)), normal=normal)


def add_label(img, text):
    """
    Only function in the project that is AI generated. It's just to
    add a label under the images, for visualization purposes.
    """
    # If the image is 1-channel (Grayscale or Threshold), convert it to BGR
    # Otherwise, it cannot be concatenated with color images
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Add a constant black border only at the bottom (40 pixels high)
    banner_height = 40
    bordered_img = cv2.copyMakeBorder(
        img,
        top=0, bottom=banner_height, left=0, right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0] # Black
    )

    # Calculate text position inside the black banner
    # (X=15 pixels from left, Y=total height minus 12 pixels from bottom)
    text_position = (15, bordered_img.shape[0] - 12)

    # Render the text
    cv2.putText(
        bordered_img,
        text,
        text_position,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(255, 255, 255), # White
        thickness=1,
        lineType=cv2.LINE_AA # Antialiasing for crisp text rendering
    )

    return bordered_img



def main():
    start_time = time.time()

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    
    # Workaround to make Open3D work on Wayland
    # Wayland is not supported apparently, so the idea is to force it
    # to use X11, so that the window will be run on xWayland
    os.environ["XDG_SESSION_TYPE"] = "x11"

    # Open3D PointCloud
    pcd = o3d.geometry.PointCloud()

    # 3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    window_name = "Laser Scanner - Realtime Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frames = 0
    skipped_frames = 0

    is_camera_resetted = False
    keep_running = True
    while cap.isOpened() and keep_running:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1

        width = len(frame[0])
        height = len(frame)

        # Compensate camera distortion
        frame = cv2.undistort(frame, K, DIST)
        # Save for later visualization
        original_frame = frame.copy()

        # This would be the best one, but it destroys performance
        # frame = cv2.bilateralFilter(frame, 15, 20, 20)
        frame = cv2.GaussianBlur(frame, (9, 9), 2)

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # grayscale = cv2.bilateralFilter(grayscale, 15, 20, 20)

        # After experimenting also with adaptive thresholding, in the end the
        # static one is the one that works best for this specific project:
        # the rectangles are the elements with the lowest intensity in the image
        # so we can put a very low threshold to detect only those.
        # Using adaptive thresholding works best for all contours of the image,
        # but it becomes more difficult to filter and find the rectangles
        _, thresholded = cv2.threshold(grayscale, 20, 255, cv2.THRESH_BINARY_INV)
        # _, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # thresholded = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61, 4)

        try:
            r1, r2, contours_r1, contours_r2 = extract_base_planes(frame, thresholded)
        except:
            skipped_frames += 1
            logging.warning(f"Failed to detect rectangles, frame skipped. Currently skipped: {skipped_frames}")
            continue



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
        red_lower = np.array([160, 35, 70])
        red_upper = np.array([182, 255, 255])
        laser_mask = cv2.inRange(hsv, red_lower, red_upper)
        # This gives points (y, x) instead of (x, y)
        laser_y, laser_x = np.where(laser_mask > 0)

        laser_points_r1 = []
        laser_points_r2 = []
        # These are to be intersected with the laser plane
        laser_points_out = []


        # 4----------7
        # |  0----3  |
        # |  |    |  |
        # |  1----2  |
        # 5----------6      <- }
        #                      } Draw a polygon using these edges and
        # 4----------7      <- } test if the point is inside
        # |  0----3  |
        # |  |    |  |
        # |  1----2  |
        # 5----------6
        detection_area = np.concatenate([
            np.array([r1_reprojection[7], r1_reprojection[4]]),
            r2_reprojection[5:7],
        ])

        # Draw the area
        frame = cv2.drawContours(frame, [detection_area.astype(np.int32)], -1, (255, 255, 255), 2)

        for px, py in zip(laser_x, laser_y):
            # I don't know what would happen if I used all 8 vertices
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

            elif cv2.pointPolygonTest(detection_area, (px.item(), py.item()), False) == 1:
                # light green
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

                laser_points_out.append((px, py))
            else:
                # If the laser is not in the rectangles, color it with a
                # dark red
                cv2.circle(frame, (px, py), 2, (0, 0, 160), -1)


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

            # Fit the laser points into a 3D plane
            laser_plane = fit_plane(laser_points_3d)

            obj_points = []
            for p in laser_points_out:
                ray = Ray3D.backproject(p, K)
                p3d = intersect_plane_ray(laser_plane, ray)
                obj_points.append(p3d)
            obj_points = np.array(obj_points).reshape(len(obj_points), 3)

            # CV and CG use different coordinate system, so we transform
            # OpenCV coordinates to OpenGL (used by Open3D)
            T = np.eye(3)
            T[1, 1] = -1  # Flip Y axis
            T[2, 2] = -1  # Flip Z axis

            obj_points = obj_points @ T


            # Reset camera on the first extracted points
            if not is_camera_resetted:
                pcd.points = o3d.utility.Vector3dVector(obj_points)
                vis.add_geometry(pcd)
                is_camera_resetted = True
                vis.reset_view_point()

            else:
                pcd.points.extend(obj_points)

            # Visualize new points in the Open3D visualizer
            vis.update_geometry(pcd)
            keep_running = vis.poll_events()
            vis.update_renderer()


        # Put labels to each image
        lbl_original = add_label(original_frame, "1. Original Input (Undistorted)")
        lbl_features = add_label(frame, "2. Detected features")
        lbl_gray     = add_label(grayscale, "3. Grayscale")
        lbl_thresh   = add_label(thresholded, "4. Binary Threshold")

        upper_row = np.hstack((lbl_original, lbl_features))
        lower_row = np.hstack((lbl_gray, lbl_thresh))

        grid = np.vstack((upper_row, lower_row))

        cv2.imshow("Laser Scanner - Realtime Monitor", grid)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            keep_running = False

    # Computation time
    end_time = time.time()
    computation_time = end_time - start_time

    # Video duration in seconds
    fps = cap.get(cv2.CAP_PROP_FPS)
    seconds = round(frames / fps)

    logging.info(f"Took {computation_time} seconds, video duration until now is {seconds}. Extraction takes {computation_time / seconds}x time")

    # I don't want it to write partial 3D objects
    if keep_running:
        ply_file = os.path.join(
                os.path.dirname(video_path),
                os.path.basename(video_path).split('.')[0] + '.ply'
                )
        logging.info(f"Writing to file {ply_file}")
        o3d.io.write_point_cloud(ply_file, pcd)
    else:
        logging.info("Not writing file due to interruption")


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
