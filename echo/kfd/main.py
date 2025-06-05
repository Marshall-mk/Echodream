import os
import numpy as np
import cv2
import re
from PIL import Image
import csv
import random

interval = 5
mode = "echo"  # or CAMUS
filter_small_block_threshold = 300 // 8  # or 300 for CAMUS
segment_threshold = 101  # or 499 for CAMUS

# mode = "CAMUS"  # or CAMUS
# filter_small_block_threshold = 300  # or 300 for CAMUS
# segment_threshold = 499  # or 499 for CAMUS


def filter_small_block(
    image=None, filter_small_block_threshold=filter_small_block_threshold
):
    """
    Removes small connected components in a binary image that are below a given area threshold.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        image, connectivity=4
    )
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] <= filter_small_block_threshold:
            image[labels == label] = 0
    return image


def smooth_frames(frames, kernal_size=1):
    """
    Smooths a sequence of frames using a moving average filter.
    """
    frame_out = []
    for i in range(len(frames)):
        num_left = i - max(0, i - kernal_size)
        num_right = min(i + kernal_size + 1, len(frames)) - i - 1
        real_num = min(num_left, num_right)
        current_frames = frames[
            max(0, i - real_num) : min(i + 1 + real_num, len(frames))
        ]
        frame = np.mean(current_frames, axis=0).astype(np.uint8)
        frame_out.append(frame)
    return frame_out


def two_stage_segmentation(
    image_folder,
    image_out_folder,
    patient,
    threshold=3000,
    border_area=None,
    max_frames=None,
):
    """
    Performs two-stage segmentation on images for a patient, including adaptive thresholding and small block filtering.
    """
    seg_frames = []
    images_folder = os.path.join(image_folder, patient)
    png_files = sorted(
        [f for f in os.listdir(images_folder) if f.endswith((".jpg", ".png"))],
        key=lambda x: int(re.findall(r"-?\d+", x)[0]),
    )

    # Limit to first max_frames if specified
    if max_frames is not None:
        png_files = png_files[:max_frames]

    org_frames = [
        cv2.cvtColor(
            cv2.imread(os.path.join(images_folder, f)).astype(np.uint8),
            cv2.COLOR_BGR2GRAY,
        )
        for f in png_files
    ]
    if mode == "CAMUS":
        border_area = remove_border_area(np.array(org_frames))
    for i, png_file in enumerate(png_files):
        frame = org_frames[i]
        if mode == "echo":
            segment_threshold = 101
            thresh_cat_frame = cv2.adaptiveThreshold(
                frame,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                segment_threshold,
                0,
            )
        else:
            segment_threshold = 499
            thresh_cat_frame = cv2.adaptiveThreshold(
                frame,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                segment_threshold,
                0,
            )
        for subfolder in ["pure_seg", "seg-merge"]:
            folder_path = f"{image_out_folder}/{patient}/{subfolder}"
            os.makedirs(folder_path, exist_ok=True)
        cv2.imwrite(
            f"{image_out_folder}/{patient}/pure_seg/{png_file}", thresh_cat_frame
        )
        thresh_cat_frame[border_area] = 255
        thresh_cat_frame = filter_small_block(thresh_cat_frame)
        cv2.imwrite(
            f"{image_out_folder}/{patient}/seg-merge/{png_file}", thresh_cat_frame
        )
        seg_frames.append(thresh_cat_frame)
    return org_frames, seg_frames, border_area


def remove_border_area(image_frames, threshold=100):
    """
    Identifies the border area in a stack of image frames by thresholding the sum across frames.
    """
    gray = np.clip(np.sum(image_frames, axis=0), 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
    height, width = gray.shape
    corner_points = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    component_mask = np.zeros_like(gray)
    for x, y in corner_points:
        label = labels[y, x]
        component_mask[labels == label] = 255
    return component_mask == 255


def get_intersection_area(seg_frames, image_out_folder, patient, mode):
    """
    Computes the intersection area across segmented frames.
    """
    if mode == "echo":
        count_times = seg_frames.sum(axis=0)
        nums = len(seg_frames)
        intersection = count_times <= 0.2 * nums
    else:
        intersection = seg_frames.sum(axis=0)
        intersection = intersection <= np.quantile(intersection, 0.01)
    intersection_complement_image = Image.fromarray(
        255 - np.uint8(intersection) * 255, mode="L"
    )
    os.makedirs(f"{image_out_folder}/{patient}", exist_ok=True)
    intersection_complement_image.save(
        f"{image_out_folder}/{patient}/intersection_complement.png"
    )
    return intersection if mode == "CAMUS" else (count_times, intersection)


def select_k_points(image, k):
    """
    Selects k evenly spaced points along the skeleton of a binary region.
    """
    region = np.where(image, 255, 0).astype(np.uint8)
    skeleton = cv2.ximgproc.thinning(region)
    y, x = np.where(skeleton == 255)
    skeleton_points = np.column_stack((x, y))
    if len(skeleton_points) >= k:
        selected_points_indices = np.linspace(0, len(skeleton_points) - 1, k, dtype=int)
        return skeleton_points[selected_points_indices]
    return skeleton_points


def find_nearest_pixel_to_centroid(region, offsetx, offsety):
    """
    Finds the pixel in a region closest to its centroid.
    """
    y, x = np.where(region == 1)
    if len(x) == 0 or len(y) == 0:
        return None
    centroid = (np.mean(x) + offsetx, np.mean(y) + offsety)
    return min(
        zip(x + offsetx, y + offsety),
        key=lambda p: (p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2,
    )


def select_4_points(image):
    """
    Selects one representative point from each quarter of a connected region.
    """
    region = np.where(image, 1, 0)
    y, x = np.where(region == 1)
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
    quarters = [
        region[min_y:center_y, min_x:center_x],
        region[min_y:center_y, center_x:max_x],
        region[center_y:max_y, min_x:center_x],
        region[center_y:max_y, center_x:max_x],
    ]
    offsets = [
        (min_x, min_y),
        (center_x, min_y),
        (min_x, center_y),
        (center_x, center_y),
    ]
    centers = [
        find_nearest_pixel_to_centroid(q.copy(), ox, oy)
        for q, (ox, oy) in zip(quarters, offsets)
    ]
    centers = [center for center in centers if center is not None]
    return centers


def select_2_points(image, intersection_times=None):
    """
    Selects one or two representative points from a region, optionally using intersection times.
    """
    if mode == "echo":
        smallest_area = intersection_times == np.min(intersection_times)
        region = np.where(image, 1, 0) * smallest_area
    else:
        region = np.where(image, 1, 0)
    y, x = np.where(region == 1)
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
    quarters = [
        region[min_y:center_y, min_x:max_x],
        region[center_y:max_y, min_x:max_x],
    ]
    centers = []
    centers.append(find_nearest_pixel_to_centroid(quarters[0].copy(), min_x, min_y))
    if mode == "CAMUS":
        centers.append(
            find_nearest_pixel_to_centroid(quarters[1].copy(), min_x, center_y)
        )
    centers = [center for center in centers if center is not None]
    return centers


def erode(image, kernel_size=3, iterations=1):
    """
    Applies morphological erosion to an image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


def get_center_point(image, image_out_folder, patient, intersection_times=None):
    """
    Finds center points in the largest connected regions of a binary image.
    """
    h, w = image.shape
    image = (image * 255).astype(np.uint8)
    if mode == "echo":
        image = erode(image, kernel_size=3, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )
    mask = image > 0
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    labels[~mask] = 0
    center_list = {
        (centroids[label][0], centroids[label][1], label): int(
            np.count_nonzero(labels == label)
        )
        for label in range(1, num_labels)
        if np.count_nonzero(labels == label) > 0
    }
    if not center_list:
        return None
    center_list = sorted(center_list.items(), key=lambda x: x[1], reverse=True)
    first_center = center_list[0]
    if mode == "echo":
        center_list = [first_center] + [
            center for center in center_list[1:] if center[1] > 100 // 4
        ]
    else:
        center_list = [first_center] + [
            center for center in center_list[1:] if center[1] > 800 // 4
        ]
    min_center = float("inf")
    center_point = None
    cv2.circle(image_bgr, (int(w // 2), 0), 1, (0, 0, 255), -1)
    for (x, y, label), _ in center_list:
        dist = abs(w // 2 - x) + abs(0 - y)
        if dist < min_center:
            min_center = dist
            if mode == "echo":
                center_point = select_2_points(labels == label, intersection_times)
            else:
                center_point = select_2_points(labels == label)
            center_point = np.array(center_point)
            # print(f"ceter_point_shape: {center_point.shape}")
    os.makedirs(f"{image_out_folder}/{patient}", exist_ok=True)
    cv2.imwrite(f"{image_out_folder}/{patient}/with_rect.png", image_bgr)
    return center_point


def find_first_white_pixel_distance(point, angle, binary_image):
    """
    Finds the distance from a point to the first white pixel along a given angle in a binary image.
    """
    angle_rad = np.deg2rad(angle)
    height, width = binary_image.shape
    dx, dy = np.cos(angle_rad), -np.sin(angle_rad)
    x, y = point
    while 0 <= x < width and 0 <= y < height:
        if binary_image[int(y), int(x)] > 0:
            distance = np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
            return distance, int(x), int(y)
        x += dx
        y += dy
    return np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2), int(x), int(y)


def get_all_angle_distance(
    points, input_image_path, image_out_folder, patient, points_and_angles=None
):
    """
    Measures distances from given points to the edge along multiple angles in a binary image.
    """
    dis = []
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    image0 = image.copy()
    if points_and_angles is None:
        points_and_angles = {
            point: [j for j in range(0, 360) if j % 5 == 0] for point in points
        }
    for point in points:
        selected_numbers = points_and_angles[point]
        for j1 in selected_numbers:
            for j in range(j1, j1 + 5):
                result = find_first_white_pixel_distance(point, j, image0)
                if result is not None:
                    first_white_pixel_distance, dst_x, dst_y = result
                else:
                    print("This degree can not find a suitable edge.")
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.circle(image, (dst_x, dst_y), 1, (0, 0, 255), -1)
                cv2.circle(image, point, 1, (0, 0, 255), -1)
                cv2.line(image, (dst_x, dst_y), point, (0, 0, 255))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                dis.append(first_white_pixel_distance)
    # print(f"len(dis): {len(dis)}")
    last_part = os.path.basename(input_image_path)
    folder_path = f"{image_out_folder}/{patient}/seg-with-line"
    os.makedirs(folder_path, exist_ok=True)
    cv2.imwrite(f"{folder_path}/{last_part}", image)
    return dis


def sort_key(s):
    """
    Sorting key for filenames, extracting the trailing number.
    """
    return int(s.split(".")[0].split("_")[-1])


def get_change_array(
    image_folder,
    image_out_folder,
    patient,
    num_sample=48,
    mask_border=None,
    max_frames=None,
):
    """
    Computes the change array of edge distances across frames for a patient.
    """
    org_frames, seg_frames, border_area = two_stage_segmentation(
        image_folder,
        image_out_folder,
        patient,
        border_area=mask_border,
        max_frames=max_frames,
    )
    seg_frames = np.array(seg_frames)
    if mode == "echo":
        intersection_times, intersection = get_intersection_area(
            seg_frames, image_out_folder, patient, mode
        )
    else:
        intersection = get_intersection_area(
            seg_frames, image_out_folder, patient, mode
        )
    intersection = intersection * (1 - border_area)
    if mode == "echo":
        cur_dst_points = get_center_point(
            intersection, image_out_folder, patient, intersection_times
        )
    else:
        cur_dst_points = get_center_point(intersection, image_out_folder, patient)
    if cur_dst_points is None or len(cur_dst_points) == 0:
        print(f"Warning: No center points found for patient {patient}")
        return np.array([]), 0, 0
    input_image_path = f"{image_out_folder}/{patient}/seg-merge"
    file_name = sorted(os.listdir(input_image_path), key=sort_key)
    possible_numbers = [j for j in range(0, 360) if j % interval == 0]
    cur_dst_points = [(int(point[0]), int(point[1])) for point in cur_dst_points]
    num_points = len(cur_dst_points)
    points_and_angles = {
        cur_dst_point: random.sample(possible_numbers, num_sample)
        for cur_dst_point in cur_dst_points
    }
    res = []
    for j in file_name:
        cur_path = os.path.join(input_image_path, j)
        dis = get_all_angle_distance(
            cur_dst_points, cur_path, image_out_folder, patient, points_and_angles
        )
        res.append(dis)
    original_array = np.array(res)
    change_array = original_array[1:, :] - original_array[:-1, :]
    return change_array, num_sample, num_points


def make_bigger_interval(change_array, nums=2):
    """
    Aggregates the change array over a larger interval by summing over previous frames.
    """
    change_array = np.pad(
        change_array,
        ((nums - 1, 0), (0, 0), (0, 0)),
        "constant",
        constant_values=(0, 0),
    )
    change_array_2 = np.zeros_like(change_array)
    for i in range(nums - 1, change_array.shape[0]):
        change_array_2[i, :] = np.sum(change_array[i - nums + 1 : i + 1, :], axis=0)
    return change_array_2[nums - 1 :, :]


def get_boarder_mask(image_folder, threshold=30, times=300):
    """
    Computes a mask for the border area by thresholding the maximum pixel value across images.
    """
    max_value = None
    count = 0
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith((".jpg", ".png")):
                if count >= times:
                    break
                count += 1
                img = cv2.cvtColor(
                    cv2.imread(os.path.join(root, file)).astype(np.uint8),
                    cv2.COLOR_BGR2GRAY,
                )
                max_value = img if max_value is None else np.maximum(max_value, img)
    return max_value < threshold


def Get_ED_and_ES(
    image_folder, image_out_folder, num_sample=48, num_points=2, max_frames=None
):
    """
    Main processing function to compute and save edge dynamics for all patients.
    """
    totoal_patients = sorted(os.listdir(image_folder))
    processed_patients = set()
    if os.path.exists(output_list_csv):
        with open(output_list_csv, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and len(row) > 0:
                    processed_patients.add(row[0])
    cashed_change_array = {}
    if mode == "echo":
        boarder_mask = get_boarder_mask(image_folder)
    if os.path.exists(cashed_change_array_csv):
        with open(cashed_change_array_csv, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                cashed_change_array[row[0]] = (
                    np.array(row[4:]).astype(np.float32),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                )
    for patient in totoal_patients:
        if patient in processed_patients:
            print(f"Skipping already processed patient: {patient}")
            continue
        if patient in cashed_change_array:
            change_array, num_sample, num_points, num_frames = cashed_change_array[
                patient
            ]
            change_array = np.array(change_array)
        else:
            if mode == "echo":
                change_array, num_sample, num_points = get_change_array(
                    image_folder,
                    image_out_folder,
                    patient,
                    num_sample,
                    boarder_mask,
                    max_frames,
                )
            else:
                change_array, num_sample, num_points = get_change_array(
                    image_folder,
                    image_out_folder,
                    patient,
                    num_sample,
                    max_frames=max_frames,
                )
            num_frames = change_array.shape[0]
            with open(cashed_change_array_csv, "a", newline="") as file:
                writer = csv.writer(file)
                change_array = change_array.reshape(-1)
                your_list = change_array.tolist()
                your_list.insert(0, num_frames)
                your_list.insert(0, num_points)
                your_list.insert(0, num_sample)
                your_list.insert(0, patient)
                writer.writerow(your_list)
        if num_points == 0 or num_sample == 0 or len(change_array) == 0:
            print(f"Warning: Invalid dimensions for patient {patient}. Skipping.")
            continue
        median_interval = 5
        possibility = 0.2 * median_interval
        change_array = np.reshape(
            change_array, (num_frames, num_sample * num_points, -1)
        )
        if mode == "echo":
            major_changed_second_index = abs(change_array) > 10
            major_changed_second_index = np.any(major_changed_second_index, axis=(0, 2))
            change_array[:, major_changed_second_index, :] = 0
            change_array[abs(change_array) > 5 // 2] = 0
        change_array[change_array > 0] = 1
        change_array[change_array < 0] = -1
        change_array = np.sum(change_array, axis=2)
        change_array[abs(change_array) < possibility] = 0
        change_array[change_array <= -possibility] = -1
        change_array[change_array >= possibility] = 1
        change_array = np.sum(change_array, axis=1) / (
            np.sum(change_array != 0, axis=1) + 1e-6
        )
        res = change_array.tolist()
        string_value = patient
        with open(output_list_csv, "a", newline="") as file:
            writer = csv.writer(file)
            length = len(res) + 1
            your_list = res
            your_list.insert(0, string_value)
            your_list.insert(1, length)
            writer.writerow(your_list)
    return 1


if __name__ == "__main__":
    # Main entry point for running the edge dynamics extraction pipeline.
    image_folder = "/nfs/usrhome/khmuhammad/EchoPath/datasets/CardiacASD/jpg"
    image_out_folder = "/home/khmuhammad/Echo-Dream/experiments/phase"
    cashed_change_array_csv = (
        "/home/khmuhammad/Echo-Dream/experiments/cashed_change_array.csv"
    )
    output_list_csv = "/home/khmuhammad/Echo-Dream/experiments/CardiacASD.csv"

    # Set max_frames to limit processing to first x frames per patient (None for all frames)
    max_frames = 64  # Change this number or set to None to process all frames

    ultra_final = Get_ED_and_ES(image_folder, image_out_folder, max_frames=max_frames)
