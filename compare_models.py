"""
模型对比训练脚本
用于系统性地比较不同模型配置的性能
"""

import os
from ultralytics import YOLO

# 配置
DATA_YAML = "datasets/vehicle_night_rainy_foggy/data.yaml"
DEVICE = "0"  # GPU设备
PROJECT = "runs/compare"

# 实验配置
experiments = [
    {
        "name": "baseline_yolov13",
        "model": "ultralytics/cfg/models/v13/yolov13.yaml",
        "epochs": 400,
        "batch": 32,
        "lr0": 0.01,
        "description": "原生YOLOv13基线模型"
    },
    {
        "name": "lite_hyperace_p2",
        "model": "ultralytics/cfg/models/v13/yolov13_hyperace_p2_lite.yaml",
        "epochs": 400,
        "batch": 32,
        "lr0": 0.01,
        "description": "轻量级版本: HyperACE + P2"
    },
    {
        "name": "hyperace_msda",
        "model": "ultralytics/cfg/models/v13/yolov13_hyperace_msda.yaml",
        "epochs": 400,
        "batch": 32,
        "lr0": 0.01,
        "description": "中等版本: HyperACE + MSDA"
    },
    {
        "name": "full_model_extended",
        "model": "ultralytics/cfg/models/v13/yolov13_ia_hyperace_msda_p2.yaml",
        "epochs": 600,
        "batch": 16,
        "lr0": 0.005,
        "description": "完整改进模型 (更多epochs和调整的超参数)"
    },
]

def train_model(config):
    """训练单个模型"""
    print(f"\n{'='*80}")
    print(f"开始训练: {config['name']}")
    print(f"描述: {config['description']}")
    print(f"模型: {config['model']}")
    print(f"Epochs: {config['epochs']}, Batch: {config['batch']}, LR: {config['lr0']}")
    print(f"{'='*80}\n")
    
    # 加载模型
    model = YOLO(config['model'])
    
    # 训练
    results = model.train(
        data=DATA_YAML,
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=640,
        device=DEVICE,
        project=PROJECT,
        name=config['name'],
        lr0=config['lr0'],
        lrf=0.01,
        warmup_epochs=5,
        patience=100,
        save=True,
        save_period=50,
        plots=True,
        verbose=True,
    )
    
    # 验证
    print(f"\n验证模型: {config['name']}")
    metrics = model.val()
    
    return {
        "name": config['name'],
        "description": config['description'],
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }

def main():
    """主函数"""
    print("="*80)
    print("YOLOv13 模型对比实验")
    print("="*80)
    
    results_summary = []
    
    # 训练所有模型
    for config in experiments:
        try:
            result = train_model(config)
            results_summary.append(result)
        except Exception as e:
            print(f"\n错误: 训练 {config['name']} 时出错")
            print(f"错误信息: {str(e)}")
            continue
    
    # 打印结果摘要
    print("\n" + "="*80)
    print("实验结果摘要")
    print("="*80)
    print(f"{'模型名称':<30} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'Precision':<12} {'Recall':<12}")
    print("-"*80)
    
    for result in results_summary:
        print(f"{result['name']:<30} "
              f"{result['mAP50']:<12.4f} "
              f"{result['mAP50-95']:<15.4f} "
              f"{result['precision']:<12.4f} "
              f"{result['recall']:<12.4f}")
    
    print("="*80)
    
    # 保存结果到文件
    with open(f"{PROJECT}/comparison_results.txt", "w", encoding="utf-8") as f:
        f.write("YOLOv13 模型对比实验结果\n")
        f.write("="*80 + "\n\n")
        
        for result in results_summary:
            f.write(f"模型: {result['name']}\n")
            f.write(f"描述: {result['description']}\n")
            f.write(f"mAP@0.5: {result['mAP50']:.4f}\n")
            f.write(f"mAP@0.5:0.95: {result['mAP50-95']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall: {result['recall']:.4f}\n")
            f.write("-"*80 + "\n\n")
    
    print(f"\n结果已保存到: {PROJECT}/comparison_results.txt")

if __name__ == "__main__":
    main()
