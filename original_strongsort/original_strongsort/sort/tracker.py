# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import

import numpy as np

from strongsort.sort import iou_matching, kalman_filter, linear_assignment
from strongsort.sort.track import Track, pose_model, MMPOSE_AVAILABLE, TrackState

# Import MMPose functions if available
if MMPOSE_AVAILABLE:
    try:
        from mmpose.apis import inference_topdown

        print("Successfully imported inference_topdown in tracker.py")
    except ImportError:
        print("Warning: inference_topdown import failed in tracker.py")


class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    GATING_THRESHOLD = np.sqrt(kalman_filter.chi2inv95[4])

    def __init__(self, metric, max_iou_distance=0.9, max_age=30, n_init=3, _lambda=0, ema_alpha=0.9, mc_lambda=0.995):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self, im0=None):
        """
        将轨迹状态分布向前传播一个时间步。
        增加了可选的图像参数用于姿态检查。
        """
        for track in self.tracks:
            track.predict(self.kf, im0)

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def camera_update(self, previous_img, current_img):
        for track in self.tracks:
            track.camera_update(previous_img, current_img)

    def update(self, detections, classes, confidences, im0=None):
        """
        Perform measurement update and track management.
        """
        # Run matching cascade (appearance and motion).
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update matched tracks with assigned detections
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                detections[detection_idx],
                classes[detection_idx],
                confidences[detection_idx],
                im0  # Pass image to update for pose checking
            )

        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Initialize new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._initiate_track(
                detections[detection_idx],
                classes[detection_idx].item() if hasattr(classes[detection_idx], "item") else classes[detection_idx],
                confidences[detection_idx].item() if hasattr(confidences[detection_idx], "item") else confidences[
                    detection_idx],
                im0
            )

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update metric with features from confirmed tracks
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _full_cost_metric(self, tracks, dets, track_indices, detection_indices):
        """
        This implements the full lambda-based cost-metric. However, in doing so, it disregards
        the possibility to gate the position only which is provided by
        linear_assignment.gate_cost_matrix(). Instead, I gate by everything.
        Note that the Mahalanobis distance is itself an unnormalised metric. Given the cosine
        distance being normalised, we employ a quick and dirty normalisation based on the
        threshold: that is, we divide the positional-cost by the gating threshold, thus ensuring
        that the valid values range 0-1.
        Note also that the authors work with the squared distance. I also sqrt this, so that it
        is more intuitive in terms of values.
        """
        # Compute First the Position-based Cost Matrix
        pos_cost = np.empty([len(track_indices), len(detection_indices)])
        msrs = np.asarray([dets[i].to_xyah() for i in detection_indices])
        for row, track_idx in enumerate(track_indices):
            pos_cost[row, :] = (
                    np.sqrt(self.kf.gating_distance(tracks[track_idx].mean, tracks[track_idx].covariance, msrs, False))
                    / self.GATING_THRESHOLD
            )
        pos_gate = pos_cost > 1.0
        # Now Compute the Appearance-based Cost Matrix
        app_cost = self.metric.distance(
            np.array([dets[i].feature for i in detection_indices]),
            np.array([tracks[i].track_id for i in track_indices]),
        )
        app_gate = app_cost > self.metric.matching_threshold
        # Now combine and threshold
        cost_matrix = self._lambda * pos_cost + (1 - self._lambda) * app_cost
        cost_matrix[np.logical_or(pos_gate, app_gate)] = linear_assignment.INFTY_COST
        # Return Matrix
        return cost_matrix

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix, tracks, dets, track_indices, detection_indices
            )

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, class_id, conf, im0=None):
        """Initialize a new track for unmatched detection."""
        # Initialize Kalman filter mean and covariance for detection
        mean, covariance = self.kf.initiate(detection.to_xyah())

        print(f"尝试初始化新轨迹, 类别: {class_id}")

        # If it's a person, first check if it's a runner before creating a track
        if class_id == 0 and im0 is not None and MMPOSE_AVAILABLE and pose_model is not None:
            # Get detection box in tlwh format
            tlwh = detection.tlwh
            x1, y1, w, h = int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])

            print(f"人物检测框: [{x1},{y1},{w},{h}]")

            # Boundary checking
            H, W = im0.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)

            # Perform pose estimation if crop area is valid
            if x2 > x1 and y2 > y1:
                try:
                    person_bbox = np.array([x1, y1, x2, y2])
                    # 使用 MMPose 模型进行姿态预测
                    pose_results = inference_topdown(pose_model, im0, [person_bbox])

                    if pose_results and len(pose_results) > 0:
                        keypoints = pose_results[0].get('keypoints', None)

                        # Only create track if it has a running pose
                        if keypoints is not None:
                            # 评估是否为跑步者
                            is_runner = Track._is_running_pose(keypoints, h)
                            print(f"姿态检查结果: {'是跑步者' if is_runner else '不是跑步者'}")

                            if not is_runner:
                                print("拒绝创建非跑步者轨迹")
                                return
                            else:
                                print("确认创建跑步者轨迹")
                        else:
                            print("未检测到关键点，无法判断")
                    else:
                        print("姿态估计返回空结果")
                except Exception as e:
                    import traceback
                    print(f"姿态估计错误: {e}")
                    traceback.print_exc()

        try:
            # Create a new Track instance with new ID and detection info
            new_track = Track(
                mean,
                covariance,
                self._next_id,
                class_id,
                conf,
                self.n_init,
                self.max_age,
                self.ema_alpha,
                detection.feature,
                im0
            )
            self.tracks.append(new_track)
            print(f"成功创建轨迹 ID: {self._next_id}, 类别: {class_id}")
            self._next_id += 1
        except Exception as e:
            import traceback
            print(f"创建轨迹错误: {e}")
            traceback.print_exc()

    def filter_non_runners(self, im0):
        """
        强制检查所有轨迹，删除非跑步者。
        用于定期强制清理，例如每隔几帧调用一次。
        """
        if im0 is None:
            return

        print("开始过滤非跑步者，当前轨迹数:", len(self.tracks))
        filtered_count = 0

        for track in self.tracks:
            if track.class_id == 0 and track.is_confirmed():
                # 存储旧状态
                old_state = track.state

                # 执行姿态检查
                track.check_pose(im0)

                # 检查状态是否变化
                if old_state != track.state and track.state == TrackState.Deleted:
                    filtered_count += 1
                    print(f"轨迹 ID: {track.track_id} 被识别为非跑步者并移除")

        # 删除被标记的轨迹
        original_count = len(self.tracks)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        new_count = len(self.tracks)

        print(f"过滤完成，剩余轨迹数: {new_count}，移除轨迹数: {original_count - new_count}")