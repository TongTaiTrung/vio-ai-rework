import cv2
import mediapipe as mp
import numpy as np
from .create_report import export_excel
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Body parts configuration
PARTS = {
    "arms": [(11, 13, 15), (12, 14, 16)],  # Left & Right: Shoulder-Elbow-Wrist
    "legs": [(23, 25, 27), (24, 26, 28)],  # Left & Right: Hip-Knee-Ankle
    "torso": [(11, 23, 25), (12, 24, 26)]  # Left & Right: Shoulder-Hip-Knee
}

JOINTS = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
    "left_hip": (11, 23, 25),
    "right_hip": (12, 24, 26),
}

WEIGHTS = {"arms": 0.35, "legs": 0.35, "torso": 0.30}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def get_landmark_coords(landmarks, idx):
    return [landmarks[idx].x, landmarks[idx].y]

def extract_angles(video_path, skip_frames=1, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    part_angles = {part: [] for part in PARTS}
    joint_angles = {joint: [] for joint in JOINTS}
    frames = []
    orientations = []
    processed = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        while cap.isOpened() and (max_frames is None or processed < max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames
            if frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue

            # Resize for performance
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * 640 / w)))

            # Process frame
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Shoulder orientation
                l_shoulder = get_landmark_coords(lm, 11)
                r_shoulder = get_landmark_coords(lm, 12)
                orient = np.degrees(np.arctan2(
                    r_shoulder[1] - l_shoulder[1],
                    r_shoulder[0] - l_shoulder[0]
                ))
                orientations.append(orient)

                # Calculate angles for each part
                for part, triplets in PARTS.items():
                    angles = []
                    for a, b, c in triplets:
                        p1, p2, p3 = get_landmark_coords(lm, a), get_landmark_coords(lm, b), get_landmark_coords(lm, c)
                        angles.append(calculate_angle(p1, p2, p3))
                    part_angles[part].append(angles)

                # Calculate angles for each joint
                for joint, (a, b, c) in JOINTS.items():
                    p1, p2, p3 = get_landmark_coords(lm, a), get_landmark_coords(lm, b), get_landmark_coords(lm, c)
                    joint_angles[joint].append(calculate_angle(p1, p2, p3))

                # Store frames (limit to 100)
                if len(frames) < 100:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    frames.append(frame)

            frame_count += 1
            processed += 1

    cap.release()

    # Convert to numpy arrays
    for part in PARTS:
        part_angles[part] = np.array(part_angles[part])
    for joint in JOINTS:
        joint_angles[joint] = np.array(joint_angles[joint])

    return part_angles, joint_angles, frames, orientations

def should_flip(sample_parts, student_parts):
    """Detect if student video is horizontally flipped"""
    if not sample_parts["arms"].size or not student_parts["arms"].size:
        return False

    # Sample 10 frames for comparison
    sample_len = len(sample_parts["arms"])
    indices = np.linspace(0, sample_len - 1, min(10, sample_len), dtype=int)

    normal_diff = 0
    flipped_diff = 0

    for part in ["arms", "legs"]:
        if not sample_parts[part].size or not student_parts[part].size:
            continue

        for idx in indices:
            if idx < len(student_parts[part]):
                normal_diff += np.mean(np.abs(sample_parts[part][idx] - student_parts[part][idx]))
                
                # Flip left/right
                flipped = student_parts[part][idx][[1, 0]]
                flipped_diff += np.mean(np.abs(sample_parts[part][idx] - flipped))

    return flipped_diff < normal_diff

def score_similarity(sample, student):
    """Score similarity using FastDTW (0-100)"""
    if not sample.size or not student.size:
        return 0.0

    min_len = min(len(sample), len(student))
    max_len = max(len(sample), len(student))

    # Length penalty
    length_penalty = 1.0 if min_len / max_len >= 0.5 else min_len / max_len

    # Subsample if too long
    if max_len > 200:
        sample = sample[np.linspace(0, len(sample) - 1, 200, dtype=int)]
        student = student[np.linspace(0, len(student) - 1, 200, dtype=int)]

    try:
        distance, _ = fastdtw(sample, student, dist=euclidean, radius=max(1, int(0.05 * max_len)))
        normalized = distance / len(sample)
        score = max(0, 100 * np.exp(-normalized * 0.05) * length_penalty)
        return float(score)
    except:
        return 0.0

def analyze(sample_video, student_videos, student_code, skip_frames=1, max_frames=300):
    """Main analysis function"""
    print(sample_video, student_videos)
    
    sample_parts, sample_joints, sample_frames, sample_orient = extract_angles(sample_video, skip_frames, max_frames)
    results = []

    for student_video in student_videos:
        student_parts, student_joints, student_frames, student_orient = extract_angles(student_video, skip_frames, max_frames)

        # Flip detection
        if should_flip(sample_parts, student_parts):
            for part in ["arms", "legs"]:
                if student_parts[part].size:
                    student_parts[part] = student_parts[part][:, [1, 0]]

        # Calculate scores
        part_scores = {}
        total_score = 0.0

        for part in PARTS:                
            score = score_similarity(sample_parts[part], student_parts[part])
            part_scores[part] = score
            total_score += score * WEIGHTS[part]

        # Completion bonus
        total_score += 50.0
        total_score = min(total_score, 100.0)

        # Generate joint summary for report
        joint_summary = {}
        for joint in JOINTS:
            joint_summary[joint] = {
                "sample_avg": float(np.mean(sample_joints[joint])) if len(sample_joints[joint]) else 0.0,
                "student_avg": float(np.mean(student_joints[joint])) if len(student_joints[joint]) else 0.0
            }

        # Generate report
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
            "used_flip": bool(should_flip(sample_parts, student_parts)),
            "part_score": part_scores,
            "joint_summary": joint_summary,
            "chart_image": urls['chart_image'],
            "excel_bytes": urls['excel_bytes'],
        })

    return results