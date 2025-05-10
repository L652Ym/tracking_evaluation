import os
from pathlib import Path

# 基本路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VISUALIZATION_DIR = os.path.join(BASE_DIR, "visualizations")
WEIGHTS_DIR = os.path.join(BASE_DIR, "models", "weights")

# SportsMOT数据集路径
SPORTSMOT_DIR = os.path.join(DATA_DIR, "SportsMOT")
SPORTSMOT_TRAIN_DIR = os.path.join(SPORTSMOT_DIR, "train")
SPORTSMOT_VAL_DIR = os.path.join(SPORTSMOT_DIR, "val")
SPORTSMOT_TEST_DIR = os.path.join(SPORTSMOT_DIR, "test")

# 确保所有目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# 跟踪器配置
REID_WEIGHTS = os.path.join(WEIGHTS_DIR, "osnet_x0_25_msmt17.pt")

# 根据是否有GPU设置设备
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 通用跟踪器参数
TRACKER_PARAMS = {
    'max_age': 70,
    'n_init': 3,
    'max_dist': 0.2,
    'max_iou_distance': 0.7,
    'nn_budget': 100
}

# 可视化配置
VISUALIZATION_COLORS = {
    'improved_strongsort': (0, 255, 0),   # 绿色
    'strongsort': (0, 0, 255),            # 蓝色
    'deepsort': (255, 0, 0),              # 红色
    'ocsort': (255, 165, 0)               # 橙色
}

# 测试配置
SELECTED_SEQUENCES = [
    # 选择包含快速运动和高遮挡的序列
    'basketball_1',
    'basketball_2',
    'football_1',
    'football_2',
    'running_100m_1',
    'volleyball_1',
    # 添加更多序列...
]

# 检测器配置 - 我们使用相同的检测结果确保公平比较
USE_PROVIDED_DETECTIONS = True  # 使用SportsMOT提供的检测结果