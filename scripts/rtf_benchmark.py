#!/usr/bin/env python3
"""
MERaLiON-2-10B-ASR RTF性能测试脚本
测试转录的实时性能指标
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from malaysian_asr.core.benchmark import RTFBenchmark


def main():
    """主函数"""
    print("=" * 70)
    print("⚡ MERaLiON-2-10B-ASR RTF性能测试")
    print("=" * 70)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        audio_files = [f for f in os.listdir('.') 
                      if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        if not audio_files:
            print("用法: python rtf_benchmark.py <音频文件>")
            return
        
        audio_file = audio_files[0]
        print(f"🔍 使用发现的音频文件: {audio_file}")
    else:
        audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return
    
    # 初始化测试器
    benchmark = RTFBenchmark()
    
    # 加载模型
    if not benchmark.load_model():
        return
    
    # 运行性能测试
    num_runs = 3 if len(sys.argv) < 3 else int(sys.argv[2])
    result = benchmark.benchmark_transcription(audio_file, num_runs)
    
    if result:
        # 保存测试报告
        report_file = f"{audio_file.rsplit('.', 1)[0]}_rtf_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"MERaLiON-2-10B-ASR RTF性能报告\\n")
            f.write(f"{'='*50}\\n")
            f.write(f"音频文件: {audio_file}\\n")
            f.write(f"音频时长: {result['audio_duration']:.2f}秒\\n")
            f.write(f"预处理时间: {result['preprocessing_time']:.3f}秒\\n")
            f.write(f"平均推理时间: {result['avg_inference_time']:.3f}秒\\n")
            f.write(f"总处理时间: {result['total_processing_time']:.3f}秒\\n")
            f.write(f"纯推理RTF: {result['rtf_inference']:.3f}\\n")
            f.write(f"总处理RTF: {result['rtf_total']:.3f}\\n")
            f.write(f"转录结果: {result['transcription']}\\n")
        
        print(f"\\n💾 性能报告已保存: {report_file}")


if __name__ == "__main__":
    main()