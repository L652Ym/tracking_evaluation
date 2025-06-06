import numpy as np
import torch

from strongsort.reid_multibackend import ReIDDetectMultiBackend
from strongsort.sort.detection import Detection
from strongsort.sort.nn_matching import NearestNeighborDistanceMetric
from strongsort.sort.tracker import Tracker


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


class StrongSORT(object):
    def __init__(
            self,
            model_weights,
            device,
            fp16,
            max_dist=0.2,
            max_iou_distance=0.7,
            max_age=70,
            n_init=3,
            nn_budget=100,
            mc_lambda=0.995,
            ema_alpha=0.9,
    ):

        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)

        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric("cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, dets, ori_img):
        # dets: ndarray [N,6]  (xyxy, conf, cls)
        xyxys = dets[:, :4]
        confs = dets[:, 4]
        clss = dets[:, 5]

        classes = clss.numpy()  # <-- numpy array
        xywhs = xyxy2xywh(xyxys.numpy())
        confs = confs.numpy()
        self.height, self.width = ori_img.shape[:2]

        # 生成 ReID 特征
        features = self._get_features(xywhs, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [Detection(bbox_tlwh[i], conf, features[i])
                      for i, conf in enumerate(confs)]

        # 预测并更新
        self.tracker.predict(ori_img)
        self.tracker.update(detections, classes, confs, ori_img)

        # 输出
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            outputs.append(
                np.array([x1, y1, x2, y2,
                          track.track_id,
                          track.class_id,
                          track.conf])
            )

        return np.stack(outputs, axis=0) if outputs else outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

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

    def filter_non_runners(self, im0):
        """过滤非跑步者目标"""
        if im0 is None:
            print("警告: 传入空图像，无法执行姿态过滤")
            return

        if hasattr(self.tracker, 'filter_non_runners'):
            track_count_before = len(self.tracker.tracks)
            print(f"开始过滤非跑步者，当前轨迹数: {track_count_before}")
            self.tracker.filter_non_runners(im0)
            track_count_after = len(self.tracker.tracks)
            print(f"过滤完成，剩余轨迹数: {track_count_after}，移除轨迹数: {track_count_before - track_count_after}")
        else:
            print("错误: tracker 对象没有 filter_non_runners 方法")