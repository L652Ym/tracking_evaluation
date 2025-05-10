import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import motmetrics as mm


def load_sportsmot_sequence(seq_dir):
    """加载SportsMOT序列的图像和检测结果"""
    img_dir = os.path.join(seq_dir, 'img1')
    gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
    det_file = os.path.join(seq_dir, 'det', 'det.txt')

    # 加载图像
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])

    # 加载检测结果
    detections = {}
    if os.path.exists(det_file):
        with open(det_file, 'r') as f:
            for line in f.readlines():
                data = line.strip().split(',')
                frame_id = int(data[0])
                bbox = [float(data[2]), float(data[3]),
                        float(data[2]) + float(data[4]),
                        float(data[3]) + float(data[5])]
                confidence = float(data[6])
                class_id = int(data[7]) if len(data) > 7 else 1  # 默认类别为1

                if frame_id not in detections:
                    detections[frame_id] = []

                # 按[x1, y1, x2, y2, conf, class_id]格式存储
                detections[frame_id].append(bbox + [confidence, class_id])

    # 加载真值
    gt_data = {}
    if os.path.exists(gt_file):
        with open(gt_file, 'r') as f:
            for line in f.readlines():
                data = line.strip().split(',')
                frame_id = int(data[0])
                track_id = int(data[1])
                bbox = [float(data[2]), float(data[3]),
                        float(data[2]) + float(data[4]),
                        float(data[3]) + float(data[5])]
                visibility = float(data[8]) if len(data) > 8 else 1.0

                if frame_id not in gt_data:
                    gt_data[frame_id] = []

                # 只有当visibility大于0.5时才使用
                if visibility > 0.5:
                    gt_data[frame_id].append([track_id] + bbox)

    # 获取序列信息
    seq_info = {
        'sequence_name': os.path.basename(seq_dir),
        'image_filenames': img_files,
        'num_frames': len(img_files),
        'image_size': cv2.imread(img_files[0]).shape[:2][::-1],  # (width, height)
        'detections': detections,
        'gt_data': gt_data
    }

    return seq_info


def convert_detections_to_tracker_format(detections, frame_id):
    """将检测结果转换为适合跟踪器的格式"""
    if frame_id not in detections:
        return np.array([])

    dets = np.array(detections[frame_id])
    if len(dets) == 0:
        return np.array([])

    # 转换为[x1, y1, x2, y2, conf, class]格式的tensor
    import torch
    return torch.tensor(dets)


def evaluate_mot_metrics(results, gt_data):
    """使用MOT指标评估跟踪结果"""
    acc = mm.MOTAccumulator(auto_id=True)

    # 按帧处理结果
    for frame_id in sorted(gt_data.keys()):
        # 获取真值
        gt_boxes = np.array([box[1:5] for box in gt_data[frame_id]])
        gt_ids = np.array([box[0] for box in gt_data[frame_id]])

        # 获取该帧的跟踪结果
        frame_results = [r for r in results if r[0] == frame_id]
        if len(frame_results) == 0:
            # 如果没有结果，创建空数组
            trk_boxes = np.array([])
            trk_ids = np.array([])
        else:
            trk_boxes = np.array([r[1][:4] for r in frame_results])  # 仅使用边界框
            trk_ids = np.array([r[1][4] for r in frame_results])  # 使用跟踪ID

        # 计算距离矩阵 (IoU)
        if len(gt_boxes) > 0 and len(trk_boxes) > 0:
            # 计算IoU距离
            distances = mm.distances.iou_matrix(gt_boxes, trk_boxes, max_iou=0.5)
            acc.update(gt_ids, trk_ids, distances)
        else:
            acc.update(gt_ids, trk_ids, [])

    # 计算指标
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        'mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost',
        'num_false_positives', 'num_misses', 'num_switches', 'precision', 'recall'
    ])

    return summary


def draw_tracking_results(frame, results, color=(0, 255, 0)):
    """在帧上绘制跟踪结果"""
    img = frame.copy()
    for result in results:
        # 边界框和ID
        bbox = result[:4].astype(int)
        track_id = int(result[4])

        # 绘制边界框
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # 绘制ID和其他信息
        label = f"ID:{track_id}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (bbox[0], bbox[1] - t_size[1] - 3),
                      (bbox[0] + t_size[0], bbox[1]), color, -1)
        cv2.putText(img, label, (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def create_comparison_visualization(frame, results_by_tracker, colors):
    """创建比较可视化，在一个图像中显示所有跟踪器的结果"""
    h, w = frame.shape[:2]
    trackers = list(results_by_tracker.keys())
    n_trackers = len(trackers)

    # 决定网格布局
    n_cols = min(2, n_trackers)
    n_rows = (n_trackers + n_cols - 1) // n_cols
    grid_h = h * (n_rows + 1)  # 额外加一行放原始图像
    grid_w = w * n_cols

    # 创建网格
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # 放置原始帧
    grid[:h, :w] = frame.copy()
    cv2.putText(grid, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 放置各跟踪器结果
    for i, tracker_name in enumerate(trackers):
        row = (i // n_cols) + 1  # 从第二行开始
        col = i % n_cols

        tracker_frame = draw_tracking_results(
            frame.copy(),
            results_by_tracker[tracker_name],
            colors.get(tracker_name, (0, 255, 0))
        )

        # 添加跟踪器名称
        cv2.putText(tracker_frame, tracker_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        y_start = row * h
        x_start = col * w
        grid[y_start:y_start + h, x_start:x_start + w] = tracker_frame

    return grid


def plot_metrics_comparison(metrics_by_tracker, output_file=None):
    """绘制不同跟踪器的指标比较图"""
    metrics = ['mota', 'idf1', 'precision', 'recall', 'num_switches']

    # 提取指标
    data = {tracker: {} for tracker in metrics_by_tracker.keys()}
    for tracker, metrics_df in metrics_by_tracker.items():
        for metric in metrics:
            if metric in metrics_df.columns:
                data[tracker][metric] = metrics_df.iloc[0][metric]

    # 转换为DataFrame
    df = pd.DataFrame(data).T

    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if i < len(axes) and metric in df.columns:
            ax = axes[i]
            df[metric].plot(kind='bar', ax=ax)
            ax.set_title(f'{metric.upper()}')
            ax.set_ylim([0, df[metric].max() * 1.2])

            # 添加具体数值
            for j, v in enumerate(df[metric]):
                ax.text(j, v + 0.01, f"{v:.4f}", ha='center')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)

    plt.close()


def generate_trajectory_visualization(results_by_tracker, output_file=None):
    """生成轨迹可视化图"""
    plt.figure(figsize=(12, 10))

    # 为每个跟踪器使用不同的颜色
    colors = ['blue', 'green', 'red', 'orange', 'purple']

    for i, (tracker_name, results) in enumerate(results_by_tracker.items()):
        # 创建轨迹字典
        trajectories = {}

        for frame_id, boxes in results:
            for box in boxes:
                track_id = int(box[4])
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2

                if track_id not in trajectories:
                    trajectories[track_id] = []

                trajectories[track_id].append((center_x, center_y))

        # 绘制每个轨迹
        for track_id, points in trajectories.items():
            if len(points) > 5:  # 只绘制足够长的轨迹
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                plt.plot(x, y, '-', color=colors[i % len(colors)], alpha=0.7, linewidth=1)

                # 标记起点和终点
                plt.scatter(x[0], y[0], color=colors[i % len(colors)], marker='o', s=30, alpha=0.7)
                plt.scatter(x[-1], y[-1], color=colors[i % len(colors)], marker='x', s=30, alpha=0.7)

    plt.legend(results_by_tracker.keys())
    plt.title('Tracking Trajectories Comparison')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True, alpha=0.3)

    if output_file:
        plt.savefig(output_file)

    plt.close()


def analyze_occlusion_recovery(results_by_tracker, gt_data):
    """分析遮挡恢复性能"""
    # 定义遮挡恢复指标
    recovery_results = {}

    for tracker_name, results in results_by_tracker.items():
        # 构建帧ID到跟踪结果的映射
        results_by_frame = {}
        for frame_id, boxes in results:
            results_by_frame[frame_id] = boxes

        # 检测遮挡事件
        occlusion_events = []
        tracked_objects = {}  # track_id -> 最后可见帧

        # 找出真值中的遮挡事件
        for frame_id in sorted(gt_data.keys()):
            gt_ids = set([box[0] for box in gt_data[frame_id]])

            # 更新当前帧可见的对象
            for track_id in gt_ids:
                tracked_objects[track_id] = frame_id

            # 检查前一帧存在但当前帧不存在的对象
            if frame_id > 1:
                for track_id, last_frame in list(tracked_objects.items()):
                    if track_id not in gt_ids and last_frame == frame_id - 1:
                        # 这可能是遮挡开始
                        start_frame = frame_id

                        # 向前查找对象何时重新出现
                        for future_frame in range(frame_id + 1, max(gt_data.keys()) + 1):
                            if future_frame in gt_data:
                                future_ids = set([box[0] for box in gt_data[future_frame]])
                                if track_id in future_ids:
                                    # 对象重新出现
                                    end_frame = future_frame
                                    occlusion_events.append({
                                        'track_id': track_id,
                                        'start_frame': start_frame,
                                        'end_frame': end_frame,
                                        'duration': end_frame - start_frame
                                    })
                                    break

        # 分析跟踪器在每个遮挡事件中的表现
        successful_recoveries = 0
        for event in occlusion_events:
            track_id = event['track_id']
            start_frame = event['start_frame']
            end_frame = event['end_frame']

            # 检查跟踪器在遮挡前是否正确跟踪了对象
            if start_frame - 1 in results_by_frame:
                pre_occlusion_ids = set([int(box[4]) for box in results_by_frame[start_frame - 1]])
                track_found_before = False
                tracked_id = None

                # 找到遮挡前跟踪器分配的ID
                for box in results_by_frame[start_frame - 1]:
                    # 使用IoU匹配
                    for gt_box in gt_data[start_frame - 1]:
                        if gt_box[0] == track_id:
                            iou = calculate_iou(box[:4], gt_box[1:5])
                            if iou > 0.5:  # 假设IoU > 0.5表示匹配
                                track_found_before = True
                                tracked_id = int(box[4])
                                break

                # 检查跟踪器在遮挡后是否仍然使用相同的ID
                if track_found_before and tracked_id is not None and end_frame in results_by_frame:
                    # 检查遮挡后跟踪器是否使用相同ID跟踪了正确的对象
                    for box in results_by_frame[end_frame]:
                        current_tracked_id = int(box[4])
                        if current_tracked_id == tracked_id:
                            # 找到具有相同ID的跟踪结果，检查是否匹配真值
                            for gt_box in gt_data[end_frame]:
                                if gt_box[0] == track_id:
                                    iou = calculate_iou(box[:4], gt_box[1:5])
                                    if iou > 0.5:  # 假设IoU > 0.5表示匹配
                                        successful_recoveries += 1
                                        break

        # 计算恢复率
        if len(occlusion_events) > 0:
            recovery_rate = successful_recoveries / len(occlusion_events)
        else:
            recovery_rate = 0.0

        recovery_results[tracker_name] = {
            'occlusion_events': len(occlusion_events),
            'successful_recoveries': successful_recoveries,
            'recovery_rate': recovery_rate
        }

    return recovery_results


def calculate_iou(boxA, boxB):
    """计算两个边界框的IoU"""
    # 确定交集矩形
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算交集面积
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算两个边界框的面积
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 计算IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def analyze_speed_adaptation(results_by_tracker, gt_data):
    """分析速度适应性能"""
    from scipy.signal import find_peaks

    speed_adaptation_results = {}

    for tracker_name, results in results_by_tracker.items():
        # 构建帧ID到跟踪结果的映射
        results_by_frame = {}
        for frame_id, boxes in results:
            results_by_frame[frame_id] = {}
            for box in boxes:
                results_by_frame[frame_id][int(box[4])] = box[:4]

        # 分析每个跟踪ID的速度变化
        track_speeds = {}
        for frame_id in sorted(gt_data.keys()):
            if frame_id - 1 in gt_data and frame_id in gt_data:
                # 计算真值中的速度
                for current_box in gt_data[frame_id]:
                    track_id = current_box[0]

                    # 查找前一帧中的相同ID
                    for prev_box in gt_data[frame_id - 1]:
                        if prev_box[0] == track_id:
                            # 计算中心点位移
                            prev_center = [(prev_box[1] + prev_box[3]) / 2, (prev_box[2] + prev_box[4]) / 2]
                            curr_center = [(current_box[1] + current_box[3]) / 2, (current_box[2] + current_box[4]) / 2]

                            # 计算速度（像素/帧）
                            speed = np.sqrt((curr_center[0] - prev_center[0]) ** 2 +
                                            (curr_center[1] - prev_center[1]) ** 2)

                            if track_id not in track_speeds:
                                track_speeds[track_id] = []

                            track_speeds[track_id].append((frame_id, speed))
                            break

        # 检测速度急剧变化的点
        speed_events = []
        for track_id, speeds in track_speeds.items():
            if len(speeds) > 10:  # 至少10帧的跟踪历史
                frames, speed_values = zip(*speeds)

                # 使用科学方法检测速度峰值
                peaks, _ = find_peaks(speed_values, height=np.mean(speed_values) * 1.5, distance=5)

                for peak_idx in peaks:
                    speed_events.append({
                        'track_id': track_id,
                        'frame_id': frames[peak_idx],
                        'speed': speed_values[peak_idx],
                        'type': 'acceleration'
                    })

                # 检测速度谷值（减速）
                troughs, _ = find_peaks([-v for v in speed_values], height=np.mean([-v for v in speed_values]) * 1.5,
                                        distance=5)

                for trough_idx in troughs:
                    speed_events.append({
                        'track_id': track_id,
                        'frame_id': frames[trough_idx],
                        'speed': speed_values[trough_idx],
                        'type': 'deceleration'
                    })

        # 分析跟踪器在速度变化时的表现
        successful_adaptations = 0
        for event in speed_events:
            track_id = event['track_id']
            frame_id = event['frame_id']
            event_type = event['type']

            # 查看事件发生前3帧和后3帧
            pre_frames = range(max(1, frame_id - 3), frame_id)
            post_frames = range(frame_id, min(frame_id + 4, max(gt_data.keys()) + 1))

            # 检查跟踪器在速度变化前的跟踪状态
            tracked_id = None
            for pre_frame in pre_frames:
                if pre_frame in results_by_frame and pre_frame in gt_data:
                    # 找到跟踪器为该目标分配的ID
                    for gt_box in gt_data[pre_frame]:
                        if gt_box[0] == track_id:
                            for result_id, result_box in results_by_frame[pre_frame].items():
                                iou = calculate_iou(result_box, gt_box[1:5])
                                if iou > 0.5:
                                    tracked_id = result_id
                                    break
                            if tracked_id is not None:
                                break

                if tracked_id is not None:
                    break

            # 如果找到了跟踪ID，检查速度变化后是否仍然保持正确跟踪
            if tracked_id is not None:
                maintained_tracking = True
                for post_frame in post_frames:
                    if post_frame in results_by_frame and post_frame in gt_data:
                        # 检查该ID是否仍在跟踪正确的目标
                        if tracked_id in results_by_frame[post_frame]:
                            correct_match = False
                            for gt_box in gt_data[post_frame]:
                                if gt_box[0] == track_id:
                                    iou = calculate_iou(results_by_frame[post_frame][tracked_id], gt_box[1:5])
                                    if iou > 0.5:
                                        correct_match = True
                                        break

                            if not correct_match:
                                maintained_tracking = False
                                break
                        else:
                            maintained_tracking = False
                            break

                if maintained_tracking:
                    successful_adaptations += 1

        # 计算速度适应率
        if len(speed_events) > 0:
            adaptation_rate = successful_adaptations / len(speed_events)
        else:
            adaptation_rate = 0.0

        speed_adaptation_results[tracker_name] = {
            'speed_events': len(speed_events),
            'successful_adaptations': successful_adaptations,
            'adaptation_rate': adaptation_rate
        }

    return speed_adaptation_results