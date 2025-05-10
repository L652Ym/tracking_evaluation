import os
import cv2
import numpy as np
import torch
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from datetime import datetime

from src.config import *
from src.utils import *
from src.trackers import ImprovedStrongSORT, OriginalStrongSORT, DeepSORT, OCSORT


def load_trackers():
    """加载所有跟踪器"""
    trackers = {
        'improved_strongsort': ImprovedStrongSORT(
            model_weights=REID_WEIGHTS,
            device=DEVICE,
            fp16=False,
            **TRACKER_PARAMS
        ),
        'strongsort': OriginalStrongSORT(
            model_weights=REID_WEIGHTS,
            device=DEVICE,
            fp16=False,
            **TRACKER_PARAMS
        ),
        'deepsort': DeepSORT(
            model_weights=REID_WEIGHTS,
            device=DEVICE,
            **TRACKER_PARAMS
        ),
        'ocsort': OCSORT(
            det_thresh=0.3,
            max_age=TRACKER_PARAMS['max_age'],
            min_hits=TRACKER_PARAMS['n_init'],
            iou_threshold=0.3,
            delta_t=3,
            asso_func="iou",
            inertia=0.2,
            use_byte=False
        )
    }

    return trackers


def evaluate_sequence(sequence_path, trackers):
    """评估单个序列的多个跟踪器"""
    # 加载序列数据
    seq_info = load_sportsmot_sequence(sequence_path)
    sequence_name = seq_info['sequence_name']
    image_files = seq_info['image_filenames']
    detections = seq_info['detections']
    gt_data = seq_info['gt_data']

    print(f"Evaluating sequence: {sequence_name}")

    # 存储结果
    results_by_tracker = {}

    # 遍历每一帧
    for frame_idx, img_file in enumerate(tqdm(image_files, desc=f"Processing {sequence_name}")):
        # 读取图像
        frame = cv2.imread(img_file)
        if frame is None:
            print(f"WARNING: Could not read image {img_file}")
            continue

        frame_id = frame_idx + 1  # SportsMOT索引从1开始

        # 获取检测结果
        frame_dets = convert_detections_to_tracker_format(detections, frame_id)
        if len(frame_dets) == 0:
            # 如果没有检测结果，通知所有跟踪器
            for tracker_name, tracker in trackers.items():
                if tracker_name not in results_by_tracker:
                    results_by_tracker[tracker_name] = []
            continue

        # 记录每个跟踪器的结果
        frame_results = {}

        # 运行每个跟踪器
        for tracker_name, tracker in trackers.items():
            if tracker_name not in results_by_tracker:
                results_by_tracker[tracker_name] = []

            # 更新跟踪器
            start_time = time.time()
            outputs = tracker.update(frame_dets, frame)
            elapsed = time.time() - start_time

            # 存储结果
            if len(outputs) > 0:
                frame_results[tracker_name] = outputs
                results_by_tracker[tracker_name].append((frame_id, outputs))
            else:
                frame_results[tracker_name] = np.array([])

        # 创建可视化 (每隔10帧)
        if frame_idx % 10 == 0:
            vis_dir = os.path.join(VISUALIZATION_DIR, sequence_name)
            os.makedirs(vis_dir, exist_ok=True)

            vis_img = create_comparison_visualization(
                frame,
                frame_results,
                VISUALIZATION_COLORS
            )

            vis_path = os.path.join(vis_dir, f"{sequence_name}_frame_{frame_id:06d}.jpg")
            cv2.imwrite(vis_path, vis_img)

    # 为每个跟踪器计算MOT指标
    metrics_by_tracker = {}
    for tracker_name, results in results_by_tracker.items():
        summary = evaluate_mot_metrics(results, gt_data)
        metrics_by_tracker[tracker_name] = summary

    # 计算专门的遮挡恢复指标
    occlusion_results = analyze_occlusion_recovery(results_by_tracker, gt_data)

    # 计算速度适应性指标
    speed_adaptation_results = analyze_speed_adaptation(results_by_tracker, gt_data)

    # 生成轨迹可视化
    trajectory_vis_path = os.path.join(VISUALIZATION_DIR, sequence_name, f"{sequence_name}_trajectories.png")
    generate_trajectory_visualization(results_by_tracker, trajectory_vis_path)

    # 绘制指标比较图
    metrics_vis_path = os.path.join(VISUALIZATION_DIR, sequence_name, f"{sequence_name}_metrics.png")
    plot_metrics_comparison(metrics_by_tracker, metrics_vis_path)

    return {
        'sequence_name': sequence_name,
        'metrics': metrics_by_tracker,
        'occlusion_recovery': occlusion_results,
        'speed_adaptation': speed_adaptation_results
    }


def run_evaluation():
    """运行所有序列评估"""
    print("Loading trackers...")
    trackers = load_trackers()

    # 找到所有选定的序列
    sequences_to_evaluate = []
    for seq_name in SELECTED_SEQUENCES:
        # 优先查找训练集
        train_path = os.path.join(SPORTSMOT_TRAIN_DIR, seq_name)
        val_path = os.path.join(SPORTSMOT_VAL_DIR, seq_name)
        test_path = os.path.join(SPORTSMOT_TEST_DIR, seq_name)

        if os.path.exists(train_path):
            sequences_to_evaluate.append(train_path)
        elif os.path.exists(val_path):
            sequences_to_evaluate.append(val_path)
        elif os.path.exists(test_path):
            sequences_to_evaluate.append(test_path)
        else:
            print(f"WARNING: Sequence {seq_name} not found in any split")

    if not sequences_to_evaluate:
        print("ERROR: No valid sequences found to evaluate!")
        return

    print(f"Found {len(sequences_to_evaluate)} sequences to evaluate")

    # 评估每个序列
    all_results = []
    for sequence_path in sequences_to_evaluate:
        results = evaluate_sequence(sequence_path, trackers)
        all_results.append(results)

    # 合并所有序列的结果
    merged_metrics = {tracker_name: [] for tracker_name in trackers.keys()}
    merged_occlusion = {tracker_name: {'occlusion_events': 0, 'successful_recoveries': 0}
                        for tracker_name in trackers.keys()}
    merged_speed = {tracker_name: {'speed_events': 0, 'successful_adaptations': 0}
                    for tracker_name in trackers.keys()}

    for result in all_results:
        seq_name = result['sequence_name']

        # 合并MOT指标
        for tracker_name, metrics in result['metrics'].items():
            merged_metrics[tracker_name].append(metrics)

        # 合并遮挡恢复指标
        for tracker_name, occlusion_data in result['occlusion_recovery'].items():
            merged_occlusion[tracker_name]['occlusion_events'] += occlusion_data['occlusion_events']
            merged_occlusion[tracker_name]['successful_recoveries'] += occlusion_data['successful_recoveries']

        # 合并速度适应性指标
        for tracker_name, speed_data in result['speed_adaptation'].items():
            merged_speed[tracker_name]['speed_events'] += speed_data['speed_events']
            merged_speed[tracker_name]['successful_adaptations'] += speed_data['successful_adaptations']

    # 计算平均MOT指标
    average_metrics = {}
    for tracker_name, metrics_list in merged_metrics.items():
        # 将指标列表转换为DataFrame
        metrics_df = pd.concat(metrics_list)
        # 计算平均值
        average_metrics[tracker_name] = metrics_df.mean()

    # 计算总体遮挡恢复率
    overall_occlusion_recovery = {}
    for tracker_name, data in merged_occlusion.items():
        if data['occlusion_events'] > 0:
            recovery_rate = data['successful_recoveries'] / data['occlusion_events']
        else:
            recovery_rate = 0

        overall_occlusion_recovery[tracker_name] = {
            'total_events': data['occlusion_events'],
            'successful_recoveries': data['successful_recoveries'],
            'recovery_rate': recovery_rate
        }

    # 计算总体速度适应率
    overall_speed_adaptation = {}
    for tracker_name, data in merged_speed.items():
        if data['speed_events'] > 0:
            adaptation_rate = data['successful_adaptations'] / data['speed_events']
        else:
            adaptation_rate = 0

        overall_speed_adaptation[tracker_name] = {
            'total_events': data['speed_events'],
            'successful_adaptations': data['successful_adaptations'],
            'adaptation_rate': adaptation_rate
        }

    # 生成最终结果表
    final_results = {
        'average_metrics': {tracker: metrics.to_dict() for tracker, metrics in average_metrics.items()},
        'occlusion_recovery': overall_occlusion_recovery,
        'speed_adaptation': overall_speed_adaptation,
        'evaluated_sequences': [os.path.basename(s) for s in sequences_to_evaluate],
        'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 保存结果
    results_file = os.path.join(RESULTS_DIR, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"Evaluation complete! Results saved to {results_file}")

    # 创建最终可视化
    create_final_visualizations(final_results)

    return final_results


def create_final_visualizations(results):
    """创建最终结果的可视化"""
    # 创建主要指标的比较图
    plt.figure(figsize=(15, 10))

    # 提取主要指标
    metrics = ['mota', 'idf1', 'mostly_tracked', 'num_switches']
    metrics_data = {metric: [] for metric in metrics}
    trackers = list(results['average_metrics'].keys())

    for metric in metrics:
        for tracker in trackers:
            if metric in results['average_metrics'][tracker]:
                metrics_data[metric].append(results['average_metrics'][tracker][metric])
            else:
                metrics_data[metric].append(0)

    # 创建子图
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        bars = plt.bar(trackers, metrics_data[metric])

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f"{height:.4f}", ha='center', va='bottom')

        plt.title(f"{metric.upper()}")
        plt.ylim(0, max(metrics_data[metric]) * 1.2)
        plt.xticks(rotation=15)
        plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, "overall_mot_metrics.png"))

    # 创建专门指标的比较图
    plt.figure(figsize=(15, 6))

    # 遮挡恢复率
    plt.subplot(1, 2, 1)
    recovery_rates = [results['occlusion_recovery'][t]['recovery_rate'] for t in trackers]
    bars = plt.bar(trackers, recovery_rates)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f"{height:.4f}", ha='center', va='bottom')

    plt.title("Occlusion Recovery Rate")
    plt.ylim(0, max(recovery_rates) * 1.2 if recovery_rates else 1.0)
    plt.xticks(rotation=15)
    plt.grid(axis='y', alpha=0.3)

    # 速度适应率
    plt.subplot(1, 2, 2)
    adaptation_rates = [results['speed_adaptation'][t]['adaptation_rate'] for t in trackers]
    bars = plt.bar(trackers, adaptation_rates)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f"{height:.4f}", ha='center', va='bottom')

    plt.title("Speed Adaptation Rate")
    plt.ylim(0, max(adaptation_rates) * 1.2 if adaptation_rates else 1.0)
    plt.xticks(rotation=15)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, "specialized_metrics.png"))

    # 生成雷达图比较所有指标
    plt.figure(figsize=(10, 10))

    # 选择要在雷达图中显示的指标
    radar_metrics = ['mota', 'idf1', 'mostly_tracked', 'precision', 'recall',
                     'occlusion_recovery_rate', 'speed_adaptation_rate']

    # 准备数据
    radar_data = {tracker: [] for tracker in trackers}

    for tracker in trackers:
        # 标准MOT指标
        for metric in radar_metrics[:5]:
            if metric in results['average_metrics'][tracker]:
                # 归一化到0-1范围
                value = max(0, min(1, results['average_metrics'][tracker][metric]))
                radar_data[tracker].append(value)
            else:
                radar_data[tracker].append(0)

        # 专门指标
        radar_data[tracker].append(results['occlusion_recovery'][tracker]['recovery_rate'])
        radar_data[tracker].append(results['speed_adaptation'][tracker]['adaptation_rate'])

    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax = plt.subplot(111, polar=True)

    for tracker, data in radar_data.items():
        data += data[:1]  # 闭合数据
        ax.plot(angles, data, linewidth=2, label=tracker)
        ax.fill(angles, data, alpha=0.1)

    # 设置雷达图属性
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics)
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(VISUALIZATION_DIR, "radar_comparison.png"))

    print(f"Final visualizations saved to {VISUALIZATION_DIR}")