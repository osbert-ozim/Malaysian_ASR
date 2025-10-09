"""
Model downloader utilities for MERaLiON-2-10B-ASR.
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

# Default cache directory
DEFAULT_CACHE_DIR = "/opt/data/Malaysian_ASR/model_cache"


def setup_directories(cache_dir: str = None):
    """Create necessary directories."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    print(f"缓存目录设置为: {cache_dir}")


def download_model(cache_dir: str = None):
    """Download MERaLiON ASR model."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    model_name = "MERaLiON/MERaLiON-2-10B-ASR"
    
    print("=" * 60)
    print("开始下载 MERaLiON ASR 模型")
    print(f"模型: {model_name}")
    print(f"下载位置: {cache_dir}")
    print("=" * 60)
    
    # Set environment variables
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    
    try:
        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # Try to download config file first
        print("正在下载模型配置...")
        config = AutoConfig.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        print("配置文件下载完成")
        
        # Download processor/tokenizer
        print("正在下载处理器...")
        try:
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            print("处理器下载完成")
        except Exception as e:
            print(f"处理器下载失败: {e}")
            processor = None
        
        # Try to use transformers pipeline directly
        print("正在创建ASR pipeline...")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            device=0 if device == "cuda" else -1
        )
        
        print("✅ 模型下载和加载成功！")
        return pipe, processor
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n尝试替代方案...")
        return download_alternative(cache_dir)


def download_alternative(cache_dir: str = None):
    """Alternative download method - manual component download."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    model_name = "MERaLiON/MERaLiON-2-10B-ASR"
    
    try:
        print("尝试手动下载模型组件...")
        
        # Only download necessary files, don't load immediately
        from huggingface_hub import snapshot_download
        
        print("正在下载模型文件...")
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=False,
            ignore_patterns=["*.bin"]  # Skip larger files, prioritize safetensors
        )
        
        print(f"✅ 模型文件已下载到: {model_path}")
        print("注意: 由于模型较大，可能需要特殊的加载方式")
        
        return model_path, None
        
    except Exception as e:
        print(f"❌ 替代方案也失败了: {e}")
        return None, None


def test_audio_inference(pipe, audio_file: str):
    """Test audio inference."""
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
    """Create simple audio test file."""
    print("创建测试音频文件...")
    try:
        import numpy as np
        import soundfile as sf
        
        # Create a simple test audio (sine wave)
        sample_rate = 16000
        duration = 2  # 2 seconds
        frequency = 440  # A note
        
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
