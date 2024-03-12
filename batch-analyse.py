import glob
import json
import os
import re
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from pyod.models.iforest import IForest

def analyse(experiment, series, area_bl, area_tl, area_br, area_tr, steps_v, steps_h):
    MIN_CHANGE = 5
    LIT_THRESHOLD = .45

    LASER_HEIGHT = 7.541

    # CALIB 10 - Fine-Tuned
    CAMERA_MATRIX = np.array([
        np.array([-0.1999696, -0.003075067, 0.001649046, -0.245]),
        np.array([-0.003454817, 0.1612259, -0.1182974, 3.58]),
        np.array([0.0004895178, -0.1183079, -0.1612545, 4.803]),
        np.array([0.0, 0.0, 0.0, 1.0])
    ])

    print(area_bl, area_tl, area_br, area_tr)

    VOLUME_SLICE_SIZE = 0.01

    ### Calibrate the camera intrinsics ###
    print("Calibrating...")

    # Find images
    image_series_paths = glob.glob("./calibration/*.png")
    images = list(map(lambda p: cv2.imread(p, cv2.IMREAD_GRAYSCALE), image_series_paths))

    # Define Checkerboard meta
    checkerboard_dims = (7, 5)  # Internal corners per (cols, rows)

    # Define optimization criteria for subpixel-search
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare outputs
    objpoints = []  # Creating vector to store vectors of 3D points for each checkerboard image
    imgpoints = []  # Creating vector to store vectors of 2D points for each checkerboard image
    objp = np.zeros((1, checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
    i = 0
    for image in images:
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
            image, checkerboard_dims,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    ret, camera_intrinsics_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, images[0].shape[::-1],
                                                                         None, None)

    # Also store the output size of the camera since it stays constant
    height, width = images[0].shape[:2]

    ### Load the image series ###
    print("Preparing images...")

    # Find images
    image_series_paths = glob.glob(f"image-series/{experiment}/{series}/*.png")

    # Convert to dict for further processing
    image_series = list(map(lambda p: {"path": p, "name": os.path.basename(p)}, sorted(image_series_paths)))

    # Filter all non-scan images
    image_series = list(filter(
        lambda p: re.match("^(?P<dir>h|v)-(?P<num>[0-9]*)-(c(?P<cid>[0-9]*))?.png$", p["name"]), image_series
    ))

    series_h_count = 0
    series_v_count = 0

    # Load data
    for path in image_series:
        match = re.match("^(?P<dir>h|v)-(?P<num>[0-9]*)-(c(?P<cid>[0-9]*))?.png$", path["name"])
        path["dir"] = match.group("dir")
        if path["dir"] == "h":
            series_h_count += 1
        else:
            series_v_count += 1
        path["num"] = int(match.group("num"))

    # Prepare camera matrix for undistortion
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_intrinsics_mtx, dist, (width, height), 1, (width, height))

    # Load and undistort images
    for path in image_series:
        path["image"] = cv2.imread(path["path"], cv2.IMREAD_GRAYSCALE)
        path["image"] = cv2.undistort(path["image"], camera_intrinsics_mtx, dist, None, newcameramtx)

    # Squash the series into a 3d structure for easier processing
    image_series_raw = np.array(list(map(lambda i: i["image"], image_series)))

    ### Determine series characteristics ###
    print("Determining series characteristics...")
    series_max = image_series_raw.max(axis=0)
    series_min = image_series_raw.min(axis=0)
    series_range = np.subtract(series_max, series_min)
    series_thresh = np.add(series_min, (series_range * LIT_THRESHOLD))

    ### Determine which pixels are lit per image ###
    print("Determining lit pixels...")
    # Store the results in these series (-1 for unlit, 1 for lit, 0 for not projected on)
    series_h = np.zeros((height, width, series_h_count))
    series_v = np.zeros((height, width, series_v_count))

    for x in range(width):
        for y in range(height):
            if series_range[y][x] < MIN_CHANGE:
                continue

            for image_def in image_series:
                # Determine if image point is lit
                value = image_def["image"][y][x]
                lit = value > series_thresh[y][x]

                if image_def["dir"] == "h":
                    series_h[y][x][image_def["num"]] = 1 if lit else -1
                else:
                    series_v[y][x][image_def["num"]] = 1 if lit else -1

    ### Decode grey code
    print("Decoding grey code...")

    # Store the recovered coordinates. Zero means that there are no coordinates for the position
    recovered_h_coords = np.zeros((height, width))
    recovered_v_coords = np.zeros((height, width))

    # Prepare code to decode series
    def decode_greycode(series_array):
        raw = 0

        for elem in series_array:
            raw = raw << 1
            if elem == 1:
                raw += 1

        h0 = raw ^ (raw >> 8)
        h1 = h0 ^ (h0 >> 4)
        h2 = h1 ^ (h1 >> 2)
        return h2 ^ (h2 >> 1)

    for x in range(width):
        for y in range(height):
            h = series_h[y][x]
            v = series_v[y][x]

            # If the range was smaller than MIN_CHANGE before, this position won't contain valid coordinates; skip
            if series_range[y][x] < MIN_CHANGE:
                continue

            h_coord = decode_greycode(h)
            v_coord = decode_greycode(v)

            if h_coord > steps_h or v_coord > steps_v:
                continue

            recovered_h_coords[y][x] = h_coord
            recovered_v_coords[y][x] = v_coord

    ### Construct rays ###
    print("Constructing rays...")

    ## Prepare generally valid data beforehand
    # Calculate how change happens
    v_change = (area_br - area_bl) / steps_h
    h_change = (area_tl - area_bl) / steps_v

    # Determine the laser position
    laser_position = np.array([0, 0, LASER_HEIGHT])

    # Also alias some camera properties for more clear code
    camera_position = CAMERA_MATRIX[:3, 3]
    camera_position_mtx = CAMERA_MATRIX
    camera_rotation_mtx = CAMERA_MATRIX[:3, :3]

    # Find distance between camera and laser
    camera_laser_distance = np.sqrt(np.sum((camera_position - laser_position) ** 2))

    # Find origin-based vectors from camera to laser and vice versa
    laser_to_camera = camera_position - laser_position
    camera_to_laser = laser_position - camera_position

    def vect_norm(v):
        return np.sqrt(v.dot(v))

    # Prepare
    pointcloud_raw = []

    # Debug Visualisations: Prepare a greycode decode visualisation
    image = np.zeros((height, width, 3), np.uint8)

    for x in range(width):
        for y in range(height):
            # As before, this will never yield coordinates, so skip
            if series_range[y][x] < MIN_CHANGE:
                continue

            h_coord = recovered_h_coords[y][x]
            v_coord = recovered_v_coords[y][x]

            ## Calculate the rays emitting from the laser

            # Calculate where the point would have hit the ground
            # ToDo: Check if unity implementation has first coord be one. If not, subtract one
            point_on_ground_2d = area_bl + (v_change * v_coord) + (h_change * h_coord)
            point_on_ground = np.array([*point_on_ground_2d, 0])

            # Subtract the laser position to get a direction vector originating at the origin
            laser_ray_raw = np.subtract(point_on_ground, laser_position)

            # Norm the laser ray to have unit length
            laser_ray_raw_length = vect_norm(laser_ray_raw)
            laser_ray = laser_ray_raw / laser_ray_raw_length

            ## Calculate the rays emitting from the camera

            # Transform the image-plane-point into the world
            point_in_image = [x, y, 1]
            point_in_camera = np.dot(np.linalg.inv(camera_intrinsics_mtx), point_in_image)
            point_in_world_raw = np.dot(camera_position_mtx, np.array([*point_in_camera, 1]))
            point_in_world = point_in_world_raw[:3] / point_in_world_raw[3]

            # Subtract the camera position to get a direction vector originating at the origin
            camera_ray_raw = np.subtract(point_in_world, camera_position)  # ToDo: Turn?

            # Norm the camera ray to have unit length
            camera_ray_raw_length = vect_norm(camera_ray_raw)
            camera_ray = camera_ray_raw / camera_ray_raw_length

            ## Triangulate the point

            # Find angles
            alpha = np.arccos(
                np.dot(laser_ray, laser_to_camera) /
                (vect_norm(laser_ray) * vect_norm(laser_to_camera))
            )
            beta = np.arccos(
                np.dot(camera_ray, camera_to_laser) /
                (vect_norm(camera_ray) * vect_norm(camera_to_laser))
            )
            gamma = np.pi - alpha - beta

            # Infer the lengths
            camera_ray_length = (camera_laser_distance / np.sin(gamma)) * np.sin(alpha)
            laser_ray_length = (camera_laser_distance / np.sin(gamma)) * np.sin(beta)

            # Compute the resulting point
            point = ((camera_ray * camera_ray_length) + camera_position)

            # Also add the point to the point cloud, if it is above the ground
            if point[2] > 0:
                pointcloud_raw.append(point)

    ### Clean up pointcloud
    def correction_DB(pointcloud_raw):
        pointcloud_raw = np.array(pointcloud_raw)

        clustering = DBSCAN(eps=0.02, min_samples=10).fit(np.array(pointcloud_raw))
        labels = clustering.labels_

        return pointcloud_raw[labels != -1]

    def correction_ifor(pointcloud_raw):
        clf = IForest(contamination=0.1)
        clf.fit(pointcloud_raw)
        y_test_pred = clf.predict(pointcloud_raw)
        return pointcloud_raw[y_test_pred == 0]

    def merge(*fs):
        def f(poi):
            for f in fs:
                poi=f(poi)
            return poi
        return f

    correction=merge(correction_DB,correction_ifor)

    #labels=correction(pointcloud_raw)
    #pointcloud = np.array(pointcloud_raw)[labels != -1]
    pointcloud = correction(pointcloud_raw)

    ### Calculate volume ###
    print("Calculate volume...")

    # Prepare Convex Hull calculation
    def convex_hull(points):
        if len(points) < 3:
            return 0

        hull = ConvexHull(points)

        return hull.points, hull.area

    # Sort the pointcloud for efficient post-processing
    pointcloud_top_down = sorted(pointcloud, key=lambda x: x[2], reverse=True)

    ## Look for reasons the process should be aborted

    # If there are not enough points
    if len(pointcloud) < 3:
        raise Exception("Unable to calculate volume: Not enough points in the pointcloud")

    # If the pointcloud is obviously too tall
    pointcloud_height = pointcloud_top_down[0][2]

    if pointcloud_height >= LASER_HEIGHT:
        raise Exception("Unable to calculate volume: Object higher than emitter")

    ## Process slices, accumulating volumeslices = []
    volume = 0
    existing_points = []
    last_area = 0
    idx = 0

    for i in range(int(np.ceil(pointcloud_height / VOLUME_SLICE_SIZE)), 0, -1):
        lower_bound = (i - 1) * VOLUME_SLICE_SIZE

        # print(f"Processing lower_bound {str(lower_bound)}")

        while len(pointcloud_top_down) > idx and pointcloud_top_down[idx][2] > lower_bound:
            existing_points.append([pointcloud_top_down[idx][0], pointcloud_top_down[idx][1], 0])
            existing_points.append([pointcloud_top_down[idx][0], pointcloud_top_down[idx][1], 0.000000001])
            idx += 1

        if len(existing_points) >= 4:
            try:
                points, area = convex_hull(existing_points)
                last_area = (area / 2)
                existing_points = list(points)
            except:
                print("WARN: Convex hull errored")

        volume += (last_area * VOLUME_SLICE_SIZE)

    return volume


EXP_BASE_PATH = "./test-data"

EXP_LOG_FORMAT = '^Bounds: \((?P<bl_x>-?\d*.\d*), -?\d*.\d*, (?P<bl_y>-?\d*.\d*)\);\((?P<tl_x>-?\d*.\d*), -?\d*.\d*, ' \
                 '(?P<tl_y>-?\d*.\d*)\);\((?P<br_x>-?\d*.\d*), -?\d.\d, (?P<br_y>-?\d*.\d*)\);\((?P<tr_x>-?\d*.\d*), ' \
                 '-?\d*.\d*, (?P<tr_y>-?\d*.\d*)\)\nWidth: (?P<v>-?\d*)\nHeight: (?P<h>-?\d*)\nLaser Line Width: (' \
                 '?P<llw>-?\d*,\d*)\n?$'

V_OFFSET = 2.75
H_OFFSET = 2.9

print("Starting batch processing...")

experiment_paths = glob.glob(f"{EXP_BASE_PATH}/*/experiment-*.txt")
experiments = list(
    map(lambda p: re.match(f'{EXP_BASE_PATH}/(?P<experiment>.*)/.*', p).group('experiment'), experiment_paths))

# experiments=["pvfjt"]

print(f"Found {len(experiments)} experiments")

for experiment in experiments:
    raw_exp_log = open(f"{EXP_BASE_PATH}/{experiment}/experiment-logsdoc-{experiment}.txt")
    exp_log = re.match(EXP_LOG_FORMAT, raw_exp_log.read())
    raw_exp_log.close()

    runs_paths = glob.glob(f"{EXP_BASE_PATH}/{experiment}/*/*.png")
    runs = list(set(
        list(map(
            lambda p: re.match(f'{EXP_BASE_PATH}/{experiment}/(?P<run>.*)/.*', p).group('run'), runs_paths
        ))
    ))

    # runs = ["8"]

    print(f"Processing {experiment} with {len(runs)} runs...")

    area_bl = np.array([float(exp_log.group("bl_x")) - V_OFFSET, float(exp_log.group("bl_y")) - H_OFFSET])
    area_tl = np.array([float(exp_log.group("tl_x")) - V_OFFSET, float(exp_log.group("tl_y")) - H_OFFSET])
    area_br = np.array([float(exp_log.group("br_x")) - V_OFFSET, float(exp_log.group("br_y")) - H_OFFSET])
    area_tr = np.array([float(exp_log.group("tr_x")) - V_OFFSET, float(exp_log.group("tr_y")) - H_OFFSET])

    steps_v = float(exp_log.group("v"))
    steps_h = float(exp_log.group("h"))

    outfile = open(f"{EXP_BASE_PATH}/{experiment}/measured-volumes.txt", "a")

    runs.sort(key=lambda a: int(a))

    results = []

    for run in runs:
        v = analyse(experiment, run, area_bl, area_tl, area_br, area_tr, steps_v, steps_h)
        results.append({"run": run, "result": v})
        outfile.write(f"{run} result: {str(v)}\n")
        outfile.flush()

    outfile.close()

    textable = open(f"{EXP_BASE_PATH}/table-{experiment}.tex", "w")
    textable.write("\\begin{table}\n")
    textable.write("\\centering\n")
    textable.write("\\begin{tabular}{lrrrr}\n")
    textable.write("Nummer & Volumen [$\\text{cm}^3$] \\\\\n")

    for result in results:
        textable.write(f"{result['run']} & {str(result['result'] * 1000000)} \\\\\n")

    textable.write("\\end{tabular}\n")
    textable.write("\\caption{Automatisiert ermittelte Volumina für Experiment " + experiment + "}\n")
    textable.write(f"\\label{f'tab:raw-exp-{experiment}'}\n")
    textable.write("\\end{table}\n")

    textable.close()

print("Done")
