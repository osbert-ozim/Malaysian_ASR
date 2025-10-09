#!/usr/bin/env python3
"""
ASR准确率计算工具
以test_sample_transcribe_result.txt为标准答案，计算转录准确率
使用Word Error Rate (WER) 和字符准确率等指标
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from malaysian_asr.utils.accuracy import evaluate_accuracy


def main():
    # 安装jieba如果还没有
    try:
        import jieba
    except ImportError:
        print("正在安装jieba分词库...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "jieba"])
        import jieba
    
    reference_file = "/opt/data/Malaysian_ASR/test_sample_transcribe_result.txt"
    hypothesis_file = "/opt/data/Malaysian_ASR/vocals_only_mono_transcripts_formatted.txt"
    
    result = evaluate_accuracy(reference_file, hypothesis_file)
    
    if result:
        print(f"\n" + "=" * 60)
        print("🎯 最终评估结果:")
        print(f"   词级准确率: {result['overall_word_accuracy']:.3f} ({result['overall_word_accuracy']*100:.1f}%)")
        print(f"   字符准确率: {result['overall_char_accuracy']:.3f} ({result['overall_char_accuracy']*100:.1f}%)")
        print(f"   Word Error Rate: {result['wer']:.3f}")
        print(f"   评估文件数: {result['total_files']}")


if __name__ == "__main__":
    main()