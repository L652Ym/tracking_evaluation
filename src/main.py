import argparse
import os
import sys
from pathlib import Path

from src.config import SPORTSMOT_DIR, SELECTED_SEQUENCES
from src.evaluation import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate multiple trackers on SportsMOT dataset')

    parser.add_argument('--data-dir', type=str, default=SPORTSMOT_DIR,
                        help='Directory containing the SportsMOT dataset')

    parser.add_argument('--sequences', type=str, nargs='+', default=SELECTED_SEQUENCES,
                        help='Specific sequences to evaluate (defaults to those in config.py)')

    parser.add_argument('--visualize-only', action='store_true',
                        help='Only generate visualizations from existing results')

    parser.add_argument('--results-file', type=str, default=None,
                        help='Path to existing results file for visualization')

    return parser.parse_args()


def main():
    args = parse_args()

    # 检查数据集路径
    if not os.path.exists(args.data_dir):
        print(f"ERROR: SportsMOT dataset not found at {args.data_dir}")
        print("Please download the dataset or update the path in config.py")
        sys.exit(1)

    # 更新配置中的序列列表
    if args.sequences and args.sequences != SELECTED_SEQUENCES:
        from src.config import SELECTED_SEQUENCES as config_seqs
        print(f"Overriding configured sequences ({config_seqs}) with: {args.sequences}")
        # 动态更新配置
        import src.config
        src.config.SELECTED_SEQUENCES = args.sequences

    if args.visualize_only and args.results_file:
        # 仅从已有结果生成可视化
        if not os.path.exists(args.results_file):
            print(f"ERROR: Results file not found at {args.results_file}")
            sys.exit(1)

        import json
        from src.evaluation import create_final_visualizations

        print(f"Loading existing results from {args.results_file}")
        with open(args.results_file, 'r') as f:
            results = json.load(f)

        print("Generating visualizations...")
        create_final_visualizations(results)
        print("Done!")
    else:
        # 运行完整评估
        print("Starting tracker evaluation...")
        run_evaluation()
        print("Evaluation complete!")


if __name__ == "__main__":
    main()