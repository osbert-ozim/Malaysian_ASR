#!/usr/bin/env python3
"""
MERaLiON Malaysian ASR 模型下载和推理脚本
直接下载模型到当前目录，避免磁盘空间问题
"""

import os
import sys
import torch
import warnings
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import pipeline
import torchaudio

warnings.filterwarnings("ignore")

# 设置缓存目录到当前目录
CACHE_DIR = "/opt/data/Malaysian_ASR/model_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

def setup_directories():
    """创建必要的目录"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"缓存目录设置为: {CACHE_DIR}")

def download_model():
    """下载MERaLiON ASR模型"""
    model_name = "MERaLiON/MERaLiON-2-10B-ASR"
    
    print("=" * 60)
    print("开始下载 MERaLiON ASR 模型")
    print(f"模型: {model_name}")
    print(f"下载位置: {CACHE_DIR}")
    print("=" * 60)
    
    try:
        # 检查GPU是否可用
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 先尝试下载配置文件
        print("正在下载模型配置...")
        config = AutoConfig.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        print("配置文件下载完成")
        
        # 下载处理器/tokenizer
        print("正在下载处理器...")
        try:
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            print("处理器下载完成")
        except Exception as e:
            print(f"处理器下载失败: {e}")
            processor = None
        
        # 尝试直接使用transformers pipeline
        print("正在创建ASR pipeline...")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            device=0 if device == "cuda" else -1
        )
        
        print("✅ 模型下载和加载成功！")
        return pipe, processor
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n尝试替代方案...")
        return download_alternative()

def download_alternative():
    """替代下载方案 - 手动下载组件"""
    model_name = "MERaLiON/MERaLiON-2-10B-ASR"
    
    try:
        print("尝试手动下载模型组件...")
        
        # 只下载必要的文件，不立即加载
        from huggingface_hub import snapshot_download
        
        print("正在下载模型文件...")
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=CACHE_DIR,
            local_files_only=False,
            ignore_patterns=["*.bin"]  # 跳过较大的文件，优先下载safetensors
        )
        
        print(f"✅ 模型文件已下载到: {model_path}")
        print("注意: 由于模型较大，可能需要特殊的加载方式")
        
        return model_path, None
        
    except Exception as e:
        print(f"❌ 替代方案也失败了: {e}")
        return None, None

def test_audio_inference(pipe, audio_file):
    """测试音频推理"""
    if pipe is None:
        print("模型未加载，无法进行推理")
        return None
        
    try:
        print(f"正在识别音频: {audio_file}")
        result = pipe(audio_file)
        print(f"识别结果: {result['text']}")
        return result
    except Exception as e:
        print(f"推理失败: {e}")
        return None

def create_simple_test():
    """创建简单的音频测试文件"""
    print("创建测试音频文件...")
    try:
        import numpy as np
        import soundfile as sf
        
        # 创建一个简单的测试音频（正弦波）
        sample_rate = 16000
        duration = 2  # 2秒
        frequency = 440  # A音符
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        test_file = "test_audio.wav"
        sf.write(test_file, audio, sample_rate)
        print(f"创建测试音频文件: {test_file}")
        return test_file
        
    except ImportError:
        print("需要安装 soundfile: pip install soundfile")
        return None
    except Exception as e:
        print(f"创建测试音频失败: {e}")
        return None

def main():
    """主函数"""
    print("MERaLiON Malaysian ASR 模型下载器")
    print("=" * 50)
    
    # 设置目录
    setup_directories()
    
    # 检查磁盘空间
    import shutil
    total, used, free = shutil.disk_usage(CACHE_DIR)
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
        print(f"模型缓存位置: {CACHE_DIR}")
        
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
