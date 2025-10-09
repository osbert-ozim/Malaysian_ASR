#!/usr/bin/env python3
"""
MERaLiON Malaysian ASR 模型下载和推理脚本
直接下载模型到当前目录，避免磁盘空间问题
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from malaysian_asr.utils.model_downloader import download_model, setup_directories, test_audio_inference, create_simple_test


def main():
    """主函数"""
    print("MERaLiON Malaysian ASR 模型下载器")
    print("=" * 50)
    
    # 设置目录
    setup_directories()
    
    # 检查磁盘空间
    import shutil
    cache_dir = "/opt/data/Malaysian_ASR/model_cache"
    total, used, free = shutil.disk_usage(cache_dir)
    print(f"可用磁盘空间: {free // (1024**3)} GB")
    
    if free < 30 * 1024**3:  # 少于30GB
        print("⚠️  警告: 可用磁盘空间不足30GB，模型下载可能失败")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            return
    
    # 下载模型
    result = download_model()
    
    if result[0] is not None:
        print("\n✅ 下载完成！")
        print(f"模型缓存位置: {cache_dir}")
        
        # 检查是否有音频文件进行测试
        audio_files = [f for f in os.listdir('.') if f.endswith(('.wav', '.mp3', '.flac'))]
        if audio_files:
            print(f"发现音频文件: {audio_files}")
            test_file = audio_files[0]
            test_audio_inference(result[0], test_file)
        else:
            print("没有找到音频文件进行测试")
            test_file = create_simple_test()
            if test_file:
                test_audio_inference(result[0], test_file)
    else:
        print("❌ 下载失败")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确保有足够的磁盘空间")
        print("3. 尝试使用VPN或镜像源")


if __name__ == "__main__":
    main()