#!/usr/bin/env python3
"""
MERaLiON-2-10B-ASR 专用转录脚本
只实现语音转录功能，简化版本
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from malaysian_asr.core.transcriber import MERaLiONTranscriber


def main():
    """主函数"""
    print("=" * 60)
    print("🎙️  MERaLiON 语音转录器")
    print("=" * 60)
    
    # 初始化转录器
    transcriber = MERaLiONTranscriber()
    
    # 加载模型
    if not transcriber.load_model():
        print("无法加载模型，请检查:")
        print("1. transformers版本是否为4.50.1")
        print("2. 网络连接是否正常")
        print("3. 磁盘空间是否充足")
        return
    
    # 处理命令行参数
    if len(sys.argv) < 2:
        # 自动查找音频文件
        audio_files = [f for f in os.listdir('.') 
                      if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        if not audio_files:
            print("\\n用法:")
            print("  python transcribe.py <音频文件1> [音频文件2] [...]")
            print("\\n支持格式: .wav, .mp3, .flac, .m4a")
            return
        
        print(f"🔍 发现 {len(audio_files)} 个音频文件")
        
        # 询问是否处理所有文件
        if len(audio_files) > 1:
            response = input(f"是否转录所有 {len(audio_files)} 个文件? (y/n): ")
            if response.lower() != 'y':
                audio_files = audio_files[:1]  # 只处理第一个
        
    else:
        audio_files = sys.argv[1:]
    
    # 开始转录
    print(f"\\n📝 开始转录 {len(audio_files)} 个文件...")
    
    if len(audio_files) == 1:
        # 单文件转录
        result = transcriber.transcribe(audio_files[0])
        if result:
            print(f"\\n🎉 转录完成!")
            print(f"📁 文件: {audio_files[0]}")
            print(f"📄 结果: {result}")
    else:
        # 批量转录
        results = transcriber.transcribe_batch(audio_files)
        
        # 显示汇总
        print(f"\\n📊 转录汇总:")
        successful = sum(1 for r in results.values() if r is not None)
        print(f"✅ 成功: {successful}/{len(audio_files)}")
        
        for file, result in results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {file}")


if __name__ == "__main__":
    main()