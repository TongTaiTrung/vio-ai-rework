import cv2
import mediapipe as mp
import numpy as np
from .create_report import export_excel
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Initialize MediaPipe pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define joint triplets for angle calculation
JOINTS = {
    "left_elbow":  (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    "right_elbow": (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    "left_shoulder": (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    "right_shoulder": (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    "left_knee":  (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    "right_knee": (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    "left_hip":   (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    "right_hip":  (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
}

# Define body parts for grouped angle calculation
PARTS = {
    "arms": [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    ],
    "legs": [
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    ],
    "torso": [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    ]
}

# Weights for each body part in scoring
WEIGHTS = {"arms": 0.35, "legs": 0.35, "torso": 0.30}

def calculate_angle_vectorized(points_a, points_b, points_c):
    """
    Vectorized calculation of angles between three sets of points.
    Args:
        points_a, points_b, points_c: np.ndarray of shape (N, 2)
    Returns:
        angles: np.ndarray of shape (N,)
    """
    ba = points_a - points_b
    bc = points_c - points_b
    cosine = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angles

def calculate_angle(a, b, c):
    """
    Calculate the angle at point b formed by points a, b, c.
    Args:
        a, b, c: list or np.ndarray of shape (2,)
    Returns:
        angle: float
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def flip_detection(sample_angles, student_angles):
    """
    Detect if the student's pose is flipped horizontally compared to the sample.
    Args:
        sample_angles: dict of np.ndarray
        student_angles: dict of np.ndarray
    Returns:
        should_flip: bool
    """
    sample_indices = np.linspace(0, len(sample_angles) - 1, min(10, len(sample_angles)), dtype=int)

    normal_diff = 0
    flipped_diff = 0

    for part in PARTS.keys():
        if len(sample_angles[part]) == 0 or len(student_angles[part]) == 0:
            continue

        for idx in sample_indices:
            if idx < len(student_angles[part]):
                normal_diff += np.mean(np.abs(sample_angles[part][idx] - student_angles[part][idx]))

                if part in ['arms', 'legs']:
                    flipped_student = student_angles[part][idx].copy()
                    flipped_student[0], flipped_student[1] = flipped_student[1], flipped_student[0]
                    flipped_diff += np.mean(np.abs(sample_angles[part][idx] - flipped_student))
                else:
                    flipped_diff += np.mean(np.abs(sample_angles[part][idx] - student_angles[part][idx]))

    return flipped_diff < normal_diff

def extract_joint_angles_optimized(video_path, skip_frames=1, max_frames=None):
    """
    Extract joint and part angles from a video using MediaPipe Pose.
    Args:
        video_path: str, path to video file
        skip_frames: int, number of frames to skip between processing
        max_frames: int or None, maximum frames to process
    Returns:
        part_angles: dict of np.ndarray
        joint_angles: dict of np.ndarray
        frames: list of np.ndarray (drawn frames)
        orientations: list of float (shoulder orientation angles)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open video: {video_path}")

    part_angles = {part: [] for part in PARTS}
    joint_angles = {joint: [] for joint in JOINTS}
    frames = []
    orientations = []
    frame_count = 0
    processed_frames = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue
            if max_frames and processed_frames >= max_frames:
                break

            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Calculate shoulder orientation angle
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                orientation = np.degrees(np.arctan2(
                    right_shoulder[1] - left_shoulder[1],
                    right_shoulder[0] - left_shoulder[0]
                ))
                orientations.append(orientation)

                # Calculate angles for each body part
                for part, triplets in PARTS.items():
                    angles = []
                    for a, b, c in triplets:
                        p1 = [landmarks[a].x, landmarks[a].y]
                        p2 = [landmarks[b].x, landmarks[b].y]
                        p3 = [landmarks[c].x, landmarks[c].y]
                        angles.append(calculate_angle(p1, p2, p3))
                    part_angles[part].append(angles)

                # Calculate angles for each joint
                for joint, (a, b, c) in JOINTS.items():
                    p1 = [landmarks[a].x, landmarks[a].y]
                    p2 = [landmarks[b].x, landmarks[b].y]
                    p3 = [landmarks[c].x, landmarks[c].y]
                    joint_angles[joint].append(calculate_angle(p1, p2, p3))

                # Store up to 100 frames with drawn landmarks
                if len(frames) < 100:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    frames.append(frame)

            frame_count += 1
            processed_frames += 1

    cap.release()

    # Convert lists to numpy arrays for easier processing
    for part in PARTS:
        part_angles[part] = np.array(part_angles[part])
    for joint in JOINTS:
        joint_angles[joint] = np.array(joint_angles[joint])

    return part_angles, joint_angles, frames, orientations

def score_with_dtw_fast(sample_part, student_part):
    """
    Score similarity between sample and student part angles using FastDTW.
    Args:
        sample_part: np.ndarray
        student_part: np.ndarray
    Returns:
        score: float (0-100)
    """
    if len(sample_part) == 0 or len(student_part) == 0:
        return 0

    min_len = min(len(sample_part), len(student_part))
    max_len = max(len(sample_part), len(student_part))

    radius = max(1, int(0.05 * max_len))  # Reduced from 0.1

    length_ratio = min_len / max_len
    if length_ratio < 0.5:
        length_penalty = length_ratio
    else:
        length_penalty = 1.0

    try:
        # Subsample if sequences are too long
        if max_len > 200:
            sample_indices = np.linspace(0, len(sample_part) - 1, 200, dtype=int)
            student_indices = np.linspace(0, len(student_part) - 1, 200, dtype=int)
            sample_subset = sample_part[sample_indices]
            student_subset = student_part[student_indices]
        else:
            sample_subset = sample_part
            student_subset = student_part

        distance, path = fastdtw(sample_subset, student_subset,
                               dist=euclidean, radius=radius)

        normalized_distance = distance / len(sample_subset)
        score = max(0, 100 * np.exp(-normalized_distance * 0.05) * length_penalty)
        return score
    except:
        return 0

def show_videos_with_info(sample_frames, student_frames, sample_orientations, student_orientations):
    """
    Display sample and student frames side by side with orientation info.
    Args:
        sample_frames: list of np.ndarray
        student_frames: list of np.ndarray
        sample_orientations: list of float
        student_orientations: list of float
    """
    length = min(len(sample_frames), len(student_frames))
    for i in range(length):
        left = cv2.resize(sample_frames[i], (480, 360))
        right = cv2.resize(student_frames[i], (480, 360))
        if i < len(sample_orientations):
            cv2.putText(left, f"Angle: {sample_orientations[i]:.1f}°",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if i < len(student_orientations):
            cv2.putText(right, f"Angle: {student_orientations[i]:.1f}°",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        combined = np.hstack((left, right))
        cv2.imshow("eww", combined)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
def analyze(sample_video, student_videos, student_code, skip_frames=1, max_frames=300):
    print(sample_video, student_videos)
    sample_parts, sample_joints, sample_frames, sample_orient = extract_joint_angles_optimized(
        sample_video, skip_frames, max_frames
    )

    results = []

    for student_video in student_videos:
        student_parts, student_joints, student_frames, student_orient = extract_joint_angles_optimized(
            student_video, skip_frames, max_frames
        )

        should_flip = flip_detection(sample_parts, student_parts)
        if should_flip:
            for part in ['arms', 'legs']:
                if len(student_parts[part]) > 0:
                    student_parts[part][:, [0, 1]] = student_parts[part][:, [1, 0]]

        part_scores = {}
        total_score = 0
        is_dam = sample_video.lower().startswith("dam")

        for part in PARTS:
            if is_dam and part in ["legs", "torso"]:
                score = 100.0
            else:
                score = score_with_dtw_fast(sample_parts[part], student_parts[part])
            part_scores[part] = score

            if (is_dam):
                if part == 'torso': total_score += score*0.5
                elif part == 'arms': total_score += score*0.5
            else:
                total_score += score * WEIGHTS[part]

        joint_summary = {}
        for joint in JOINTS:
            joint_summary[joint] = {
                "sample_avg": float(np.mean(sample_joints[joint])) if len(sample_joints[joint]) else 0,
                "student_avg": float(np.mean(student_joints[joint])) if len(student_joints[joint]) else 0
            }

        student_completed = (len(student_frames) > 50 and np.mean(list(part_scores.values())) > 30)
        if student_completed:
            total_score += 30.0
        total_score = min(total_score, 100.0)

        urls = export_excel(joint_summary, total_score, student_code)

        results.append({
            "student_code": student_code,
            "student_orient": student_orient,
            "sample_orient": sample_orient,
            "score": f"{total_score:.2f}",
            "sample_frame": len(sample_frames),
            "student_frame": len(student_frames),
            "avg_angle_sample": f"{np.mean(sample_orient):.1f}" if sample_orient else "0",
            "avg_angle_student": f"{np.mean(student_orient):.1f}" if student_orient else "0",
            "used_flip": bool(should_flip),
            "part_score": {k: float(v) for k, v in part_scores.items()},
            "joint_summary": joint_summary,
            "chart_image": urls['chart_image'],
            "excel_bytes": urls['excel_bytes'],
        })

    return results