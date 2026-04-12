"""
模型诊断脚本
分析模型复杂度、参数量、计算量等指标
"""

from ultralytics import YOLO
import torch

def analyze_model(model_path, model_name):
    """分析单个模型"""
    print(f"\n{'='*80}")
    print(f"分析模型: {model_name}")
    print(f"路径: {model_path}")
    print(f"{'='*80}\n")
    
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 获取模型信息
        model_info = model.info(verbose=True)
        
        # 打印详细信息
        print(f"\n模型统计:")
        print(f"  总参数量: {model_info[0]:,}")
        print(f"  总GFLOPs: {model_info[1]:.2f}")
        print(f"  层数: {model_info[2]}")
        
        # 测试推理速度
        print(f"\n测试推理速度 (640x640)...")
        dummy_input = torch.randn(1, 3, 640, 640)
        
        if torch.cuda.is_available():
            model.model.cuda()
            dummy_input = dummy_input.cuda()
            
            # 预热
            for _ in range(10):
                _ = model.model(dummy_input)
            
            # 测速
            import time
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                _ = model.model(dummy_input)
            torch.cuda.synchronize()
            end = time.time()
            
            avg_time = (end - start) / 100
            fps = 1 / avg_time
            print(f"  平均推理时间: {avg_time*1000:.2f} ms")
            print(f"  FPS: {fps:.2f}")
        else:
            print("  GPU不可用，跳过速度测试")
        
        return {
            "name": model_name,
            "params": model_info[0],
            "gflops": model_info[1],
            "layers": model_info[2],
        }
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return None

def main():
    """主函数"""
    print("="*80)
    print("YOLOv13 模型复杂度对比分析")
    print("="*80)
    
    models = [
        ("ultralytics/cfg/models/v13/yolov13.yaml", "原生YOLOv13"),
        ("ultralytics/cfg/models/v13/yolov13_hyperace_p2_lite.yaml", "轻量级 (HyperACE+P2)"),
        ("ultralytics/cfg/models/v13/yolov13_hyperace_msda.yaml", "中等 (HyperACE+MSDA)"),
        ("ultralytics/cfg/models/v13/yolov13_ia_hyperace_msda_p2.yaml", "完整改进版"),
    ]
    
    results = []
    for model_path, model_name in models:
        result = analyze_model(model_path, model_name)
        if result:
            results.append(result)
    
    # 打印对比表格
    print("\n" + "="*80)
    print("模型复杂度对比")
    print("="*80)
    print(f"{'模型名称':<30} {'参数量':<15} {'GFLOPs':<12} {'层数':<10}")
    print("-"*80)
    
    baseline_params = results[0]['params'] if results else 1
    
    for result in results:
        params_ratio = result['params'] / baseline_params
        print(f"{result['name']:<30} "
              f"{result['params']:>12,} ({params_ratio:.2f}x)  "
              f"{result['gflops']:>8.2f}  "
              f"{result['layers']:>8}")
    
    print("="*80)
    
    # 分析建议
    print("\n分析建议:")
    if len(results) >= 2:
        params_increase = (results[-1]['params'] - results[0]['params']) / results[0]['params'] * 100
        gflops_increase = (results[-1]['gflops'] - results[0]['gflops']) / results[0]['gflops'] * 100
        
        print(f"1. 完整改进模型相比基线:")
        print(f"   - 参数量增加: {params_increase:.1f}%")
        print(f"   - 计算量增加: {gflops_increase:.1f}%")
        
        if params_increase > 50:
            print(f"\n2. 警告: 参数量增加超过50%，建议:")
            print(f"   - 增加训练epochs到至少 {int(400 * (1 + params_increase/200))}")
            print(f"   - 降低学习率到 0.005 或更低")
            print(f"   - 使用更大的warmup_epochs (5-10)")
        
        if gflops_increase > 30:
            print(f"\n3. 计算量显著增加，建议:")
            print(f"   - 减小batch size以避免OOM")
            print(f"   - 考虑使用梯度累积")
            print(f"   - 使用混合精度训练 (amp=True)")

if __name__ == "__main__":
    main()
