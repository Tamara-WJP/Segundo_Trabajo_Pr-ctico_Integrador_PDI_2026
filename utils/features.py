import numpy as np

from config import LARGO_SECUENCIA
from utils.landmarks import HAND_LANDMARKS, HAND_VALUES, POSE_LANDMARKS, POSE_VALUES, TOTAL_KEYPOINTS


FEATURE_VERSION = "relative_pose_hands_temporal_v2"

TEMPORAL_STATIC_MOTION_THRESHOLD = 0.015
TEMPORAL_ACTIVE_SCORE_FRACTION = 0.25
TEMPORAL_MIN_ACTIVE_SCORE = 0.012
TEMPORAL_CONTEXT_FRAMES = 2
TEMPORAL_MIN_SEGMENT_FRAMES = 8

POSE_NOSE = 0
POSE_LEFT_EYE = 2
POSE_RIGHT_EYE = 5
POSE_MOUTH_LEFT = 9
POSE_MOUTH_RIGHT = 10
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16

HAND_WRIST = 0
HAND_INDEX_MCP = 5
HAND_MIDDLE_MCP = 9

ARM_POSE_INDEXES = [
    POSE_LEFT_ELBOW,
    POSE_RIGHT_ELBOW,
    POSE_LEFT_WRIST,
    POSE_RIGHT_WRIST,
]

LEFT_HAND_OFFSET = POSE_VALUES
RIGHT_HAND_OFFSET = POSE_VALUES + HAND_VALUES


def _split_sequence(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = sequence[:, :POSE_VALUES].reshape(LARGO_SECUENCIA, POSE_LANDMARKS, 4)
    left_hand = sequence[:, LEFT_HAND_OFFSET:RIGHT_HAND_OFFSET].reshape(
        LARGO_SECUENCIA,
        HAND_LANDMARKS,
        3,
    )
    right_hand = sequence[:, RIGHT_HAND_OFFSET:].reshape(LARGO_SECUENCIA, HAND_LANDMARKS, 3)
    return pose, left_hand, right_hand


def _smooth_scores(scores: np.ndarray) -> np.ndarray:
    if len(scores) < 3:
        return scores

    padded = np.pad(scores, (1, 1), mode="edge")
    return ((padded[:-2] + padded[1:-1] + padded[2:]) / 3.0).astype(np.float32)


def _motion_scores(sequence: np.ndarray) -> np.ndarray:
    normalized = normalize_sequence(sequence)
    pose, left_hand, right_hand = _split_sequence(normalized)
    raw_pose, raw_left_hand, raw_right_hand = _split_sequence(sequence)
    scores = np.zeros(LARGO_SECUENCIA, dtype=np.float32)

    for frame_index in range(1, LARGO_SECUENCIA):
        deltas = []

        for hand, raw_hand in (
            (left_hand, raw_left_hand),
            (right_hand, raw_right_hand),
        ):
            visible_now = np.any(np.abs(raw_hand[frame_index]) > 1e-6, axis=1)
            visible_prev = np.any(np.abs(raw_hand[frame_index - 1]) > 1e-6, axis=1)
            visible = visible_now & visible_prev

            if np.count_nonzero(visible) >= 3:
                deltas.append(
                    np.linalg.norm(
                        hand[frame_index, visible, :2]
                        - hand[frame_index - 1, visible, :2],
                        axis=1,
                    )
                )

        visible_pose = (
            (raw_pose[frame_index, ARM_POSE_INDEXES, 3] > 0.2)
            & (raw_pose[frame_index - 1, ARM_POSE_INDEXES, 3] > 0.2)
        )
        if np.any(visible_pose):
            deltas.append(
                np.linalg.norm(
                    pose[frame_index, ARM_POSE_INDEXES][visible_pose, :2]
                    - pose[frame_index - 1, ARM_POSE_INDEXES][visible_pose, :2],
                    axis=1,
                )
            )

        if deltas:
            scores[frame_index] = float(np.percentile(np.concatenate(deltas), 75))

    return _smooth_scores(scores)


def _active_segment_bounds(sequence: np.ndarray) -> tuple[int, int]:
    scores = _motion_scores(sequence)
    max_score = float(scores.max())

    if max_score < TEMPORAL_STATIC_MOTION_THRESHOLD:
        return 0, LARGO_SECUENCIA - 1

    threshold = max(TEMPORAL_MIN_ACTIVE_SCORE, max_score * TEMPORAL_ACTIVE_SCORE_FRACTION)
    active_frames = np.flatnonzero(scores >= threshold)

    if len(active_frames) == 0:
        return 0, LARGO_SECUENCIA - 1

    start = max(0, int(active_frames[0]) - TEMPORAL_CONTEXT_FRAMES)
    end = min(LARGO_SECUENCIA - 1, int(active_frames[-1]) + TEMPORAL_CONTEXT_FRAMES)

    if end - start + 1 < TEMPORAL_MIN_SEGMENT_FRAMES:
        center = (start + end) // 2
        half_width = TEMPORAL_MIN_SEGMENT_FRAMES // 2
        start = max(0, center - half_width)
        end = min(LARGO_SECUENCIA - 1, start + TEMPORAL_MIN_SEGMENT_FRAMES - 1)
        start = max(0, end - TEMPORAL_MIN_SEGMENT_FRAMES + 1)

    return start, end


def _resample_frames(sequence: np.ndarray, target_length: int = LARGO_SECUENCIA) -> np.ndarray:
    if len(sequence) == target_length:
        return sequence.astype(np.float32, copy=True)

    if len(sequence) == 1:
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)

    source_positions = np.linspace(0.0, 1.0, num=len(sequence), dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    resampled = np.empty((target_length, sequence.shape[1]), dtype=np.float32)

    for column_index in range(sequence.shape[1]):
        resampled[:, column_index] = np.interp(
            target_positions,
            source_positions,
            sequence[:, column_index],
        )

    return resampled


def temporal_normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Recorta el movimiento activo de la seña y lo reescala a 30 frames."""
    start, end = _active_segment_bounds(sequence)
    return _resample_frames(sequence[start : end + 1])


def _pose_anchor_and_scale(pose_frame: np.ndarray) -> tuple[np.ndarray, float, bool]:
    pose_xyz = pose_frame[:, :3]
    pose_present = bool(np.any(np.abs(pose_xyz) > 0))

    if not pose_present:
        return np.array([0.5, 0.5, 0.0], dtype=np.float32), 1.0, False

    anchor = pose_xyz[POSE_NOSE].astype(np.float32)
    if not np.any(np.abs(anchor) > 0):
        visible_points = pose_xyz[np.any(np.abs(pose_xyz) > 0, axis=1)]
        anchor = visible_points.mean(axis=0).astype(np.float32)

    left_shoulder = pose_xyz[POSE_LEFT_SHOULDER, :2]
    right_shoulder = pose_xyz[POSE_RIGHT_SHOULDER, :2]
    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
    scale = shoulder_width if shoulder_width > 1e-5 else 1.0

    return anchor, scale, True


def _normalize_hand(hand_frame: np.ndarray, anchor: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.zeros_like(hand_frame, dtype=np.float32)
    mask = np.any(np.abs(hand_frame) > 0, axis=1)

    if np.any(mask):
        normalized[mask] = (hand_frame[mask] - anchor) / scale

    return normalized


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Normaliza landmarks respecto de la cara y el tamano aproximado del cuerpo."""
    pose, left_hand, right_hand = _split_sequence(sequence)
    normalized_pose = np.zeros_like(pose, dtype=np.float32)
    normalized_left = np.zeros_like(left_hand, dtype=np.float32)
    normalized_right = np.zeros_like(right_hand, dtype=np.float32)

    for frame_index in range(LARGO_SECUENCIA):
        anchor, scale, pose_present = _pose_anchor_and_scale(pose[frame_index])

        if pose_present:
            normalized_pose[frame_index, :, :3] = (pose[frame_index, :, :3] - anchor) / scale
            normalized_pose[frame_index, :, 3] = pose[frame_index, :, 3]

        normalized_left[frame_index] = _normalize_hand(left_hand[frame_index], anchor, scale)
        normalized_right[frame_index] = _normalize_hand(right_hand[frame_index], anchor, scale)

    return np.concatenate(
        [
            normalized_pose.reshape(LARGO_SECUENCIA, POSE_VALUES),
            normalized_left.reshape(LARGO_SECUENCIA, HAND_VALUES),
            normalized_right.reshape(LARGO_SECUENCIA, HAND_VALUES),
        ],
        axis=1,
    )


def _mean_point(points: np.ndarray, indexes: list[int]) -> np.ndarray:
    selected = points[indexes]
    valid = selected[np.any(np.abs(selected) > 0, axis=1)]

    if len(valid) == 0:
        return np.zeros(3, dtype=np.float32)

    return valid.mean(axis=0).astype(np.float32)


def _relative_point_features(point: np.ndarray, target: np.ndarray, scale: float) -> list[float]:
    if not np.any(np.abs(point) > 0) or not np.any(np.abs(target) > 0):
        return [0.0, 0.0, 0.0, 0.0]

    vector = (point - target) / scale
    distance_xy = float(np.linalg.norm(vector[:2]))
    return [float(vector[0]), float(vector[1]), float(vector[2]), distance_xy]


def hand_face_features(sequence: np.ndarray) -> np.ndarray:
    """Genera distancias explicitas entre manos, cara y hombros."""
    pose, left_hand, right_hand = _split_sequence(sequence)
    feature_rows = []

    for frame_index in range(LARGO_SECUENCIA):
        pose_xyz = pose[frame_index, :, :3]
        _, scale, pose_present = _pose_anchor_and_scale(pose[frame_index])

        if pose_present:
            targets = [
                pose_xyz[POSE_NOSE],
                _mean_point(pose_xyz, [POSE_MOUTH_LEFT, POSE_MOUTH_RIGHT]),
                _mean_point(pose_xyz, [POSE_LEFT_EYE, POSE_RIGHT_EYE]),
                _mean_point(pose_xyz, [POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER]),
            ]
        else:
            targets = [np.zeros(3, dtype=np.float32) for _ in range(4)]

        row = []
        for hand in (left_hand[frame_index], right_hand[frame_index]):
            hand_mask = np.any(np.abs(hand) > 0, axis=1)
            hand_present = float(np.any(hand_mask))
            wrist = hand[HAND_WRIST]
            index_mcp = hand[HAND_INDEX_MCP]
            middle_mcp = hand[HAND_MIDDLE_MCP]
            center = hand[hand_mask].mean(axis=0) if np.any(hand_mask) else np.zeros(3, dtype=np.float32)

            row.append(hand_present)
            for point in (wrist, index_mcp, middle_mcp, center):
                for target in targets:
                    row.extend(_relative_point_features(point, target, scale))

        feature_rows.append(row)

    return np.array(feature_rows, dtype=np.float32)


def build_feature_vector(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    expected_shape = (LARGO_SECUENCIA, TOTAL_KEYPOINTS)

    if sequence.shape != expected_shape:
        raise ValueError(
            f"La secuencia debe tener forma {expected_shape}, "
            f"pero se recibio {sequence.shape}."
        )

    sequence = temporal_normalize_sequence(sequence)
    normalized = normalize_sequence(sequence)
    deltas = np.diff(normalized, axis=0, prepend=normalized[:1])
    relative = hand_face_features(sequence)

    return np.concatenate(
        [
            normalized.reshape(-1),
            deltas.reshape(-1),
            relative.reshape(-1),
        ],
    ).astype(np.float32)


def build_feature_matrix(sequences: np.ndarray) -> np.ndarray:
    return np.array([build_feature_vector(sequence) for sequence in sequences], dtype=np.float32)
