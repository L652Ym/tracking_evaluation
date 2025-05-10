import torch
import numpy as np
import cv2
from src.config import TRACKER_PARAMS


# 导入各种跟踪器实现
# 注意：这里假设您已经有这些跟踪器的实现，包括改进的StrongSORT

class ImprovedStrongSORT:
    """改进版StrongSORT跟踪器"""

    def __init__(self, model_weights, device, fp16=False,
                 max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3,
                 nn_budget=100, mc_lambda=0.995, ema_alpha=0.9):
        from strongsort.reid_multibackend import ReIDDetectMultiBackend
        from strongsort.sort.detection import Detection
        from strongsort.sort.nn_matching import NearestNeighborDistanceMetric

        # 导入修改后的跟踪器组件
        from improved_strongsort.sort.tracker import Tracker  # 改进的Tracker

        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)

        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric("cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            ema_alpha=ema_alpha,
            mc_lambda=mc_lambda
        )

        self.device = device

    def update(self, dets, ori_img):
        """更新跟踪状态"""
        if len(dets) == 0:
            self.tracker.predict()
            return np.array([])

        xyxys = dets[:, :4]
        # 继续 src/trackers.py

        confs = dets[:, 4]
        clss = dets[:, 5]

        classes = clss.cpu().numpy()
        xyxys_np = xyxys.cpu().numpy()
        confs_np = confs.cpu().numpy()
        self.height, self.width = ori_img.shape[:2]

        # 生成检测特征
        xywhs = self._xyxy_to_xywh(xyxys_np)
        features = self._get_features(xywhs, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confs_np)]

        # 更新跟踪器
        self.tracker.predict()
        self.tracker.update(detections, clss, confs)

        # 输出边界框和身份
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)

            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf

            # 添加遮挡信息（如果有）
            is_occluded = False
            if hasattr(track, 'is_occluded'):
                is_occluded = track.is_occluded

            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf, int(is_occluded)]))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def _xyxy_to_xywh(self, bbox_xyxy):
        """将 [x1, y1, x2, y2] 转换为 [x, y, w, h]，其中 xy 是中心点"""
        bbox_xywh = np.zeros_like(bbox_xyxy)
        bbox_xywh[:, 0] = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2
        bbox_xywh[:, 1] = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2
        bbox_xywh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_xywh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_xywh

    def _xywh_to_tlwh(self, bbox_xywh):
        """将 [x, y, w, h] 转换为 [top-left x, top-left y, w, h]"""
        bbox_tlwh = np.zeros_like(bbox_xywh)
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2
        bbox_tlwh[:, 2:] = bbox_xywh[:, 2:]
        return bbox_tlwh

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """将 [top-left x, top-left y, w, h] 转换为 [x1, y1, x2, y2]"""
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        """提取检测目标的外观特征"""
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features

    class OriginalStrongSORT:
        """原始StrongSORT跟踪器"""

        def __init__(self, model_weights, device, fp16=False,
                     max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3,
                     nn_budget=100, mc_lambda=0.995, ema_alpha=0.9):
            # 导入原始StrongSORT组件
            from strongsort.reid_multibackend import ReIDDetectMultiBackend
            from strongsort.sort.detection import Detection
            from strongsort.sort.nn_matching import NearestNeighborDistanceMetric
            from strongsort.sort.tracker import Tracker

            self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)

            self.max_dist = max_dist
            metric = NearestNeighborDistanceMetric("cosine", self.max_dist, nn_budget)
            self.tracker = Tracker(
                metric,
                max_iou_distance=max_iou_distance,
                max_age=max_age,
                n_init=n_init,
                ema_alpha=ema_alpha,
                mc_lambda=mc_lambda
            )

            self.device = device

        def update(self, dets, ori_img):
            """更新跟踪状态"""
            if len(dets) == 0:
                self.tracker.predict()
                return np.array([])

            xyxys = dets[:, :4]
            confs = dets[:, 4]
            clss = dets[:, 5]

            classes = clss.cpu().numpy()
            xyxys_np = xyxys.cpu().numpy()
            confs_np = confs.cpu().numpy()
            self.height, self.width = ori_img.shape[:2]

            # 生成检测特征
            xywhs = self._xyxy_to_xywh(xyxys_np)
            features = self._get_features(xywhs, ori_img)
            bbox_tlwh = self._xywh_to_tlwh(xywhs)
            detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confs_np)]

            # 更新跟踪器
            self.tracker.predict()
            self.tracker.update(detections, clss, confs)

            # 输出边界框和身份
            outputs = []
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)

                track_id = track.track_id
                class_id = track.class_id
                conf = track.conf

                outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))

            if len(outputs) > 0:
                outputs = np.stack(outputs, axis=0)
            return outputs

        # 与ImprovedStrongSORT相同的辅助方法
        def _xyxy_to_xywh(self, bbox_xyxy):
            bbox_xywh = np.zeros_like(bbox_xyxy)
            bbox_xywh[:, 0] = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2
            bbox_xywh[:, 1] = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2
            bbox_xywh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
            bbox_xywh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
            return bbox_xywh

        def _xywh_to_tlwh(self, bbox_xywh):
            bbox_tlwh = np.zeros_like(bbox_xywh)
            bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2
            bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2
            bbox_tlwh[:, 2:] = bbox_xywh[:, 2:]
            return bbox_tlwh

        def _tlwh_to_xyxy(self, bbox_tlwh):
            x, y, w, h = bbox_tlwh
            x1 = max(int(x), 0)
            x2 = min(int(x + w), self.width - 1)
            y1 = max(int(y), 0)
            y2 = min(int(y + h), self.height - 1)
            return x1, y1, x2, y2

        def _get_features(self, bbox_xywh, ori_img):
            im_crops = []
            for box in bbox_xywh:
                x1, y1, x2, y2 = self._xywh_to_xyxy(box)
                im = ori_img[y1:y2, x1:x2]
                im_crops.append(im)
            if im_crops:
                features = self.model(im_crops)
            else:
                features = np.array([])
            return features

    class DeepSORT:
        """DeepSORT跟踪器"""

        def __init__(self, model_weights, device, max_dist=0.2, max_iou_distance=0.7,
                     max_age=70, n_init=3, nn_budget=100):
            # 导入DeepSORT组件
            from deep_sort.deep_sort import DeepSort as DeepSORTTracker

            # 初始化DeepSORT跟踪器
            self.tracker = DeepSORTTracker(
                model_weights,
                device,
                max_dist=max_dist,
                max_iou_distance=max_iou_distance,
                max_age=max_age,
                n_init=n_init,
                nn_budget=nn_budget
            )

            self.height, self.width = None, None

        def update(self, dets, ori_img):
            """更新跟踪状态"""
            if len(dets) == 0:
                return np.array([])

            self.height, self.width = ori_img.shape[:2]

            # 转换检测格式
            xyxys = dets[:, :4].cpu().numpy()
            confs = dets[:, 4].cpu().numpy()
            clss = dets[:, 5].cpu().numpy()

            # 使用DeepSORT更新
            outputs = self.tracker.update(xyxys, confs, clss, ori_img)

            # 格式化输出
            if len(outputs) > 0:
                # DeepSORT输出格式: [x1, y1, x2, y2, track_id, class_id, conf]
                outputs = np.array(outputs)
            else:
                outputs = np.array([])

            return outputs

    class OCSORT:
        """OC-SORT跟踪器"""

        def __init__(self, det_thresh=0.3, max_age=70, min_hits=3,
                     iou_threshold=0.3, delta_t=3, asso_func="iou",
                     inertia=0.2, use_byte=False):
            # 导入OC-SORT组件
            from ocsort.ocsort import OCSort as OCSORTTracker

            # 初始化OC-SORT跟踪器
            self.tracker = OCSORTTracker(
                det_thresh=det_thresh,
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold,
                delta_t=delta_t,
                asso_func=asso_func,
                inertia=inertia,
                use_byte=use_byte
            )

            self.height, self.width = None, None

        def update(self, dets, ori_img):
            """更新跟踪状态"""
            if len(dets) == 0:
                return np.array([])

            self.height, self.width = ori_img.shape[:2]

            # 转换检测格式
            xyxys = dets[:, :4].cpu().numpy()
            confs = dets[:, 4].cpu().numpy()
            clss = dets[:, 5].cpu().numpy()

            # 将检测结果组合为OC-SORT所需格式: [x1, y1, x2, y2, conf, class]
            ocsort_dets = np.concatenate((xyxys, confs.reshape(-1, 1), clss.reshape(-1, 1)), axis=1)

            # 使用OC-SORT更新
            outputs = self.tracker.update(ocsort_dets)

            # 格式化输出
            if len(outputs) > 0:
                # OC-SORT输出格式: [x1, y1, x2, y2, track_id, class_id, conf]
                outputs = np.array(outputs)
            else:
                outputs = np.array([])

            return outputs