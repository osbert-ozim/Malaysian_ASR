#!/usr/bin/env python3
"""
MERaLiON-2-10B-ASR 批量目录转录脚本
对指定目录下的所有音频文件进行转录
输出格式: 音频文件名|转录结果
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from malaysian_asr.core.batch_transcriber import BatchTranscriber


def main():
    """主函数"""
    print("=" * 70)
    print("📁 MERaLiON 批量目录转录器")
    print("=" * 70)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法:")
        print("  python batch_transcribe.py <音频目录> [输出文件.txt]")
        print("\\n示例:")
        print("  python batch_transcribe.py /opt/data/vocals_only_mono")
        print("  python batch_transcribe.py /opt/data/vocals_only_mono results.txt")
        return
    
    directory_path = sys.argv[1]
    
    # 输出文件名
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        # 默认输出文件名
        dir_name = Path(directory_path).name
        output_file = f"{dir_name}_transcripts.txt"
    
    print(f"📁 音频目录: {directory_path}")
    print(f"📝 输出文件: {output_file}")
    
    # 初始化转录器
    transcriber = BatchTranscriber()
    
    # 加载模型
    if not transcriber.load_model():
        return
    
    # 开始批量转录
    transcriber.batch_transcribe_directory(directory_path, output_file)


if __name__ == "__main__":
    main()