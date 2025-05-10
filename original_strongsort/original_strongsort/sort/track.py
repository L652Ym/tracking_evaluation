import cv2
import numpy as np
import time
import signal
from contextlib import contextmanager

from strongsort.sort.kalman_filter import KalmanFilter

# Import MMPose modules
try:
    from mmpose.apis import init_model, inference_topdown
    from mmpose.utils import register_all_modules

    register_all_modules()
    MMPOSE_AVAILABLE = True
    print("MMPose modules imported successfully")
except ImportError:
    print("Warning: MMPose not available or version incompatible")
    MMPOSE_AVAILABLE = False

# Define pose_model at module level
try:
    print("Initializing pose estimation model...")
    POSE_CONFIG = 'C:/Users/lmy/.conda/envs/yolov8/Lib/site-packages/mmpose/.mim/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_aic-coco-256x192.py'
    POSE_CHECKPOINT = 'C:/Users/lmy/.conda/envs/yolov8/Lib/site-packages/mmpose/.mim/configs/body_2d_keypoint/rtmpose/coco/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth'

    import os

    if os.path.exists(POSE_CONFIG) and os.path.exists(POSE_CHECKPOINT) and MMPOSE_AVAILABLE:
        try:
            # Try initializing with custom options to disable test-time augmentation
            from mmengine.registry import RUNNERS
            from mmengine.config import Config

            cfg = Config.fromfile(POSE_CONFIG)

            # Disable flip test which might be causing problems
            if 'model' in cfg and 'test_cfg' in cfg.model:
                cfg.model.test_cfg.flip_test = False
                print("Disabled flip test in pose model configuration")

            # Initialize model with modified config
            pose_model = init_model(cfg, POSE_CHECKPOINT, device='cpu')
            print("Pose estimation model loaded successfully with custom config!")
        except Exception as e:
            print(f"Custom config loading failed, falling back to standard loading: {e}")
            pose_model = init_model(POSE_CONFIG, POSE_CHECKPOINT, device='cpu')
            print("Pose estimation model loaded successfully with standard method!")
    else:
        print("Pose model config or checkpoint not found")
        pose_model = None
except Exception as e:
    print(f"Error initializing pose model: {e}")
    pose_model = None


# Timeout helper for pose estimation
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    """

    def __init__(self, mean, covariance, track_id, class_id, conf, n_init, max_age, ema_alpha, feature=None, im0=None):
        self.track_id = track_id
        self.class_id = int(class_id)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_alpha = ema_alpha

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            feature /= np.linalg.norm(feature)
            self.features.append(feature)

        self.conf = conf
        self._n_init = n_init
        self._max_age = max_age

        # Initialize Kalman filter
        self.kf = KalmanFilter()
        self.mean, self.covariance = mean, covariance  # Use provided mean and covariance

        # For class_id=0 (human), check pose estimation
        if im0 is not None and self.class_id == 0:  # Only filter humans (class_id=0)
            # Get current detection box in tlwh format
            tlwh = self.to_tlwh()
            x1, y1, w, h = int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])

            # Boundary check to ensure crop area is valid
            if im0 is not None:
                H, W = im0.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x1 + w)
                y2 = min(H, y1 + h)

                # Only perform pose estimation when detection area is valid
                if x2 > x1 and y2 > y1:
                    # Try different approaches to get pose estimation to work
                    self._try_pose_estimation(im0, x1, y1, x2, y2, h)

    def _try_pose_estimation(self, im0, x1, y1, x2, y2, bbox_height):
        """Try multiple approaches for pose estimation with robust error handling"""
        try:
            # Check if pose model is available
            if not MMPOSE_AVAILABLE or pose_model is None:
                print(f"ID {self.track_id}: Pose model not available, skipping pose check")
                return

            # Try different bbox formats
            formats_to_try = [
                # Format 1: Simple array [x1, y1, x2, y2]
                lambda: np.array([x1, y1, x2, y2]),

                # Format 2: Array with score [[x1, y1, x2, y2, score]]
                lambda: np.array([[x1, y1, x2, y2, 1.0]]),

                # Format 3: Dict format with bbox, center, scale, rotation
                lambda: [dict(
                    bbox=np.array([x1, y1, x2 - x1, y2 - y1]),  # [x1, y1, w, h]
                    center=np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
                    scale=np.array([x2 - x1, y2 - y1]) / 200.0,
                    rotation=0
                )]
            ]

            # Try each format with timeout
            pose_results = None

            for format_func in formats_to_try:
                try:
                    # Set a timeout to prevent hanging
                    with time_limit(2):  # 2 second timeout
                        person_bbox = format_func()
                        pose_results = inference_topdown(pose_model, im0, person_bbox)

                        # If we get valid results, break the loop
                        if pose_results and len(pose_results) > 0:
                            break

                except TimeoutError:
                    print(f"ID {self.track_id}: Pose estimation timed out")
                    continue
                except Exception as e:
                    print(f"ID {self.track_id}: Format attempt failed: {e}")
                    continue

            # Process results if we got any
            if pose_results and len(pose_results) > 0:
                keypoints = pose_results[0].get('keypoints', None)

                if keypoints is not None:
                    # Check if it's a running pose
                    if not self._is_running_pose(keypoints, bbox_height):
                        # Not a running pose, mark for deletion
                        self.state = TrackState.Deleted
                        print(f"ID {self.track_id}: Marked as non-runner for deletion")
                else:
                    print(f"ID {self.track_id}: No keypoints detected")
                    self._fallback_pose_check(bbox_height)
            else:
                print(f"ID {self.track_id}: Pose estimation failed, using fallback")
                self._fallback_pose_check(bbox_height)

        except Exception as e:
            import traceback
            print(f"ID {self.track_id}: Pose estimation error: {e}")
            traceback.print_exc()
            self._fallback_pose_check(bbox_height)

    def _fallback_pose_check(self, bbox_height):
        """Use simple heuristics when pose estimation fails"""

        # Get current bounding box
        tlwh = self.to_tlwh()
        w, h = tlwh[2], tlwh[3]

        # Simple aspect ratio check - runners typically have taller bounding boxes
        aspect_ratio = h / w if w > 0 else 0

        # Very low aspect ratio often indicates a motorcycle rider (wide)
        if aspect_ratio < 1.3:
            print(f"ID {self.track_id}: Low aspect ratio ({aspect_ratio:.2f}), likely not a runner")
            self.state = TrackState.Deleted
            return

        # Check motion pattern using Kalman filter covariance
        # Motorcycle riders often have more stable height compared to runners
        height_variance = self.covariance[3, 3] if hasattr(self, 'covariance') else 0

        if height_variance < 5 and self.age > 5:
            print(f"ID {self.track_id}: Stable height variance ({height_variance:.2f}), likely not a runner")
            self.state = TrackState.Deleted
            return

        print(f"ID {self.track_id}: Fallback check passed, keeping as potential runner")

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get kf estimated current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The predicted kf bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def ECC(self, src, dst, warp_mode=cv2.MOTION_EUCLIDEAN, eps=1e-5, max_iter=100, scale=0.1, align=False):
        """Compute the warp matrix from src to dst.
        Parameters
        ----------
        src : ndarray
            An NxM matrix of source img(BGR or Gray), it must be the same format as dst.
        dst : ndarray
            An NxM matrix of target img(BGR or Gray).
        warp_mode: flags of opencv
            translation: cv2.MOTION_TRANSLATION
            rotated and shifted: cv2.MOTION_EUCLIDEAN
            affine(shift,rotated,shear): cv2.MOTION_AFFINE
            homography(3d): cv2.MOTION_HOMOGRAPHY
        eps: float
            the threshold of the increment in the correlation coefficient between two iterations
        max_iter: int
            the number of iterations.
        scale: float or [int, int]
            scale_ratio: float
            scale_size: [W, H]
        align: bool
            whether to warp affine or perspective transforms to the source image
        Returns
        -------
        warp matrix : ndarray
            Returns the warp matrix from src to dst.
            if motion models is homography, the warp matrix will be 3x3, otherwise 2x3
        src_aligned: ndarray
            aligned source image of gray
        """

        # BGR2GRAY
        if src.ndim == 3:
            # Convert images to grayscale
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        # make the imgs smaller to speed up
        if scale is not None:
            if isinstance(scale, float) or isinstance(scale, int):
                if scale != 1:
                    src_r = cv2.resize(src, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    dst_r = cv2.resize(dst, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    scale = [scale, scale]
                else:
                    src_r, dst_r = src, dst
                    scale = None
            else:
                if scale[0] != src.shape[1] and scale[1] != src.shape[0]:
                    src_r = cv2.resize(src, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                    dst_r = cv2.resize(dst, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                    scale = [scale[0] / src.shape[1], scale[1] / src.shape[0]]
                else:
                    src_r, dst_r = src, dst
                    scale = None
        else:
            src_r, dst_r = src, dst

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        try:
            (cc, warp_matrix) = cv2.findTransformECC(src_r, dst_r, warp_matrix, warp_mode, criteria, None, 1)
        except cv2.error as e:
            return None, None

        if scale is not None:
            warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
            warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]

        if align:
            sz = src.shape
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                # Use warpPerspective for Homography
                src_aligned = cv2.warpPerspective(src, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
            else:
                # Use warpAffine for Translation, Euclidean and Affine
                src_aligned = cv2.warpAffine(src, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
            return warp_matrix, src_aligned
        else:
            return warp_matrix, None

    def get_matrix(self, matrix):
        eye = np.eye(3)
        dist = np.linalg.norm(eye - matrix)
        if dist < 100:
            return matrix
        else:
            return eye

    def camera_update(self, previous_frame, next_frame):
        warp_matrix, src_aligned = self.ECC(previous_frame, next_frame)
        if warp_matrix is None and src_aligned is None:
            return
        [a, b] = warp_matrix
        warp_matrix = np.array([a, b, [0, 0, 1]])
        warp_matrix = warp_matrix.tolist()
        matrix = self.get_matrix(warp_matrix)

        x1, y1, x2, y2 = self.to_tlbr()
        x1_, y1_, _ = matrix @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = matrix @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.mean[:4] = [cx, cy, w / h, h]

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self, kf, im0=None):
        """
        Propagate the state distribution through the Kalman filter prediction step.
        Added optional image parameter for additional pose checking.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

        # If image data is provided, perform pose check for human targets
        if im0 is not None and self.class_id == 0 and self.time_since_update <= 3:
            # Only check recently updated tracks to avoid computational burden
            self.check_pose(im0)

    def update(self, detection, class_id, conf, im0=None):
        """Perform Kalman filter measurement update step and update the feature cache."""
        # First save old state to validate pose after update
        old_state = self.state

        # Update basic information
        self.conf = conf
        self.class_id = int(class_id) if not isinstance(class_id, int) else class_id

        # Update Kalman filter state first
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, detection.to_xyah(), detection.confidence
        )

        # Update features
        feature = detection.feature / np.linalg.norm(detection.feature)
        if len(self.features) > 0:
            smooth_feat = self.ema_alpha * self.features[-1] + (1 - self.ema_alpha) * feature
            smooth_feat /= np.linalg.norm(smooth_feat)
            self.features = [smooth_feat]
        else:
            self.features = [feature]

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        # For human targets, perform pose check
        if self.class_id == 0 and im0 is not None:
            self.check_pose(im0)

            # If pose check changes state to deleted from confirmed, log it
            if old_state == TrackState.Confirmed and self.state == TrackState.Deleted:
                print(f"Confirmed track ID {self.track_id} deleted after pose check")

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    @staticmethod
    def calculate_angle(pointA, pointB, pointC):
        """Calculate angle between three points, with B as vertex"""
        BA = pointA - pointB
        BC = pointC - pointB

        cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Prevent numerical errors

        angle = np.degrees(np.arccos(cosine_angle))
        return angle

    @staticmethod
    def _is_running_pose(keypoints, bbox_height):
        """
        Use stricter standards to distinguish runners from motorcycle riders,
        enhancing detection of side-view motorcycles
        """
        # Confidence threshold
        thr = 0.3  # Slightly lower threshold to capture more keypoints

        # Keypoint indices (COCO 17-point format)
        L_HIP, R_HIP = 11, 12
        L_KNEE, R_KNEE = 13, 14
        L_ANK, R_ANK = 15, 16
        L_SHOULDER, R_SHOULDER = 5, 6
        NECK = 1  # Neck
        NOSE = 0  # Nose

        # Get keypoint coordinates and confidence
        kp = np.array(keypoints)

        # Check if there are enough valid keypoints
        valid_points = np.sum(kp[:, 2] > thr)
        if valid_points < 7:  # Increase minimum required keypoints
            print(f"Insufficient keypoints ({valid_points}/17), cannot reliably determine")
            return False  # Not enough keypoints for reliable determination

        # 1. Motorcycle rider features - Check seated posture (most significant feature)
        # Side-view motorcycle rider features: legs bent at 90 degrees, torso leaning forward

        # Check leg angle - legs on motorcycles are typically significantly bent
        if kp[L_HIP, 2] >= thr and kp[L_KNEE, 2] >= thr and kp[L_ANK, 2] >= thr:
            leg_angle = Track.calculate_angle(
                kp[L_HIP, :2], kp[L_KNEE, :2], kp[L_ANK, :2])
            # Legs on motorcycles typically bend close to 90 degrees
            if 80 < leg_angle < 110:  # Tighter angle range
                print(f"Left leg bend angle {leg_angle:.1f}° matches motorcycle rider")
                return False  # Highly bent leg, likely a motorcycle rider

        if kp[R_HIP, 2] >= thr and kp[R_KNEE, 2] >= thr and kp[R_ANK, 2] >= thr:
            leg_angle = Track.calculate_angle(
                kp[R_HIP, :2], kp[R_KNEE, :2], kp[R_ANK, :2])
            if 80 < leg_angle < 110:
                print(f"Right leg bend angle {leg_angle:.1f}° matches motorcycle rider")
                return False

        # 2. Check torso tilt angle - motorcycle riders typically lean forward more
        if kp[NECK, 2] >= thr and ((kp[L_HIP, 2] >= thr) or (kp[R_HIP, 2] >= thr)):
            # Use visible hip
            hip_x, hip_y = 0, 0
            count = 0
            if kp[L_HIP, 2] >= thr:
                hip_x += kp[L_HIP, 0]
                hip_y += kp[L_HIP, 1]
                count += 1
            if kp[R_HIP, 2] >= thr:
                hip_x += kp[R_HIP, 0]
                hip_y += kp[R_HIP, 1]
                count += 1

            if count > 0:
                hip_x /= count
                hip_y /= count

                # Calculate torso angle with vertical
                dx = kp[NECK, 0] - hip_x
                dy = kp[NECK, 1] - hip_y
                torso_angle = abs(np.degrees(np.arctan2(dx, -dy)))

                # Motorcycle riders typically lean forward more
                if torso_angle > 15:  # Lower threshold to detect forward lean
                    print(f"Torso forward lean angle {torso_angle:.1f}° matches motorcycle rider")
                    return False

        # 3. Check ankle position - from side view of motorcycle, ankle is near or below hip
        if kp[L_HIP, 2] >= thr and kp[L_ANK, 2] >= thr:
            ankle_hip_x_diff = abs(kp[L_ANK, 0] - kp[L_HIP, 0])
            relative_diff = ankle_hip_x_diff / bbox_height
            if relative_diff < 0.2:  # Stricter threshold
                print(f"Left ankle-hip horizontal distance ratio {relative_diff:.2f} matches motorcycle rider")
                return False

        if kp[R_HIP, 2] >= thr and kp[R_ANK, 2] >= thr:
            ankle_hip_x_diff = abs(kp[R_ANK, 0] - kp[R_HIP, 0])
            relative_diff = ankle_hip_x_diff / bbox_height
            if relative_diff < 0.2:
                print(f"Right ankle-hip horizontal distance ratio {relative_diff:.2f} matches motorcycle rider")
                return False

        # 4. Additional: Check knee position - motorcycle riders' knees often higher than ankles
        if kp[L_KNEE, 2] >= thr and kp[L_ANK, 2] >= thr:
            knee_ankle_y_diff = kp[L_KNEE, 1] - kp[L_ANK, 1]
            if knee_ankle_y_diff < 0:  # Knee position higher than ankle
                print("Left knee higher than ankle, matches motorcycle rider")
                return False

        if kp[R_KNEE, 2] >= thr and kp[R_ANK, 2] >= thr:
            knee_ankle_y_diff = kp[R_KNEE, 1] - kp[R_ANK, 1]
            if knee_ankle_y_diff < 0:  # Knee position higher than ankle
                print("Right knee higher than ankle, matches motorcycle rider")
                return False

        # 5. Runner-specific posture features - need to meet at least one
        runner_features_met = 0

        # Runners typically have legs positioned one in front of the other
        if kp[L_ANK, 2] >= thr and kp[R_ANK, 2] >= thr:
            ankle_x_diff = abs(kp[L_ANK, 0] - kp[R_ANK, 0])
            relative_diff = ankle_x_diff / bbox_height
            if relative_diff > 0.25:  # Runners have significant front-to-back foot distance
                print(f"Foot front-to-back distance ratio {relative_diff:.2f} matches runner")
                runner_features_met += 1

        # Runners have noticeable arm swing
        if kp[L_SHOULDER, 2] >= thr and kp[R_SHOULDER, 2] >= thr:
            shoulder_width = abs(kp[L_SHOULDER, 0] - kp[R_SHOULDER, 0])
            relative_width = shoulder_width / bbox_height
            if relative_width > 0.3:  # Runners have opened shoulders
                print(f"Shoulder width ratio {relative_width:.2f} matches runner")
                runner_features_met += 1

        # Runners typically have one leg straight and one bent
        if kp[L_KNEE, 2] >= thr and kp[R_KNEE, 2] >= thr:
            knee_height_diff = abs(kp[L_KNEE, 1] - kp[R_KNEE, 1])
            relative_diff = knee_height_diff / bbox_height
            if relative_diff > 0.1:  # Significant knee height difference
                print(f"Knee height difference ratio {relative_diff:.2f} matches runner")
                runner_features_met += 1

        # If no runner features met and all previous checks passed, do additional checks
        if runner_features_met == 0:
            # Check for additional motorcycle features

            # Check for legs parallel to ground (more obvious in side view)
            if kp[L_HIP, 2] >= thr and kp[L_ANK, 2] >= thr:
                hip_ankle_y_diff = abs(kp[L_HIP, 1] - kp[L_ANK, 1])
                relative_y_diff = hip_ankle_y_diff / bbox_height
                # On motorcycle, leg and foot height can be close
                if relative_y_diff < 0.2:
                    print(f"Left hip-ankle vertical distance ratio {relative_y_diff:.2f} matches motorcycle rider")
                    return False

            if kp[R_HIP, 2] >= thr and kp[R_ANK, 2] >= thr:
                hip_ankle_y_diff = abs(kp[R_HIP, 1] - kp[R_ANK, 1])
                relative_y_diff = hip_ankle_y_diff / bbox_height
                if relative_y_diff < 0.2:
                    print(f"Right hip-ankle vertical distance ratio {relative_y_diff:.2f} matches motorcycle rider")
                    return False

        # Print final determination
        is_runner = runner_features_met > 0
        print(f"Pose determination: {'runner' if is_runner else 'not runner'