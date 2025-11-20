#!/usr/bin/env python3
"""
MERaLiON-2-10B-ASR 正确版本推理脚本
使用transformers==4.50.1，遵循官方文档要求
"""

import os
import sys
import torch
import librosa
import warnings
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

warnings.filterwarnings("ignore")

# 设置缓存目录
CACHE_DIR = "/opt/data/Malaysian_ASR/model_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

def load_model_with_correct_version():
    """使用正确版本的transformers加载模型"""
    repo_id = "MERaLiON/MERaLiON-2-10B-ASR"
    
    print("🚀 使用transformers 4.50.1加载MERaLiON-2-10B-ASR...")
    print(f"缓存目录: {CACHE_DIR}")
    
    try:
        # 检查设备
        if torch.cuda.is_available():
            print("✅ 检测到GPU，使用CUDA加速")
            device = "cuda"
            torch_dtype = torch.bfloat16
        else:
            print("💻 使用CPU模式")
            device = "cpu"
            torch_dtype = torch.float32
        
        # 加载处理器
        print("📥 正在加载处理器...")
        processor = AutoProcessor.from_pretrained(
            repo_id, 
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        print("   ✅ 处理器加载成功")
        
        # 加载模型
        print("📥 正在加载模型...")
        if device == "cuda":
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                repo_id,
                use_safetensors=True,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto",
                cache_dir=CACHE_DIR
            )
        else:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                repo_id,
                use_safetensors=True,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                cache_dir=CACHE_DIR
            )
        
        print("   ✅ 模型加载成功")
        return model, processor, device
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None, None

def transcribe_with_official_format(model, processor, device, audio_path, task="transcribe"):
    """使用官方格式进行推理"""
    try:
        print(f"\\n🎵 正在处理音频文件: {audio_path}")
        
        # 使用librosa加载音频，16kHz采样率
        audio_array, sample_rate = librosa.load(audio_path, sr=16000)
        duration = len(audio_array) / 16000
        print(f"   📊 音频时长: {duration:.2f}秒")
        
        # 官方推荐的prompt模板
        prompt_template = "Instruction: {query} \\nFollow the text instruction based on the following audio: <SpeechHere>"
        
        # 根据任务类型设置查询
        if task == "transcribe":
            query = "Please transcribe this speech."
        elif task == "translate_chinese":
            query = "Can you please translate this speech into written Chinese?"
        elif task == "translate_english":
            query = "Can you please translate this speech into English?"
        else:
            query = task  # 自定义查询
        
        print(f"   🎯 任务: {query}")
        
        # 创建对话格式（官方要求）
        conversation = [
            [{"role": "user", "content": prompt_template.format(query=query)}]
        ]
        
        # 应用聊天模板
        print("   📝 应用聊天模板...")
        chat_prompt = processor.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 处理输入
        print("   🔄 预处理输入...")
        inputs = processor(text=chat_prompt, audios=[audio_array])
        
        # 移动到设备
        if device == "cuda":
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = inputs[key].to(device)
                    if value.dtype == torch.float32:
                        inputs[key] = inputs[key].to(torch.bfloat16)
        
        # 生成推理结果
        print("   🧠 正在进行推理...")
        print("   ⏳ 请耐心等待（大模型推理需要时间）...")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # 解码结果
        generated_ids = outputs[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        result = response[0] if response else "无法识别"
        print(f"\\n🎉 推理完成!")
        print(f"📄 结果: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("=" * 70)
    print("🔥 MERaLiON-2-10B-ASR 官方推理器")
    print("📚 基于官方文档: https://huggingface.co/MERaLiON/MERaLiON-2-10B-ASR")
    print("🔧 使用transformers 4.50.1")
    print("=" * 70)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        # 查找音频文件
        audio_files = [f for f in os.listdir('.') 
                      if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        if not audio_files:
            print("\\n使用方法:")
            print("  python correct_inference.py <音频文件> [任务类型]")
            print("\\n任务类型:")
            print("  transcribe (默认) - 语音转录")  
            print("  translate_chinese - 翻译成中文")
            print("  translate_english - 翻译成英文")
            print("\\n支持格式: .wav, .mp3, .flac, .m4a")
            return
        
        audio_file = audio_files[0]
        task = "transcribe"
        print(f"🔍 自动使用发现的音频文件: {audio_file}")
    else:
        audio_file = sys.argv[1]
        task = sys.argv[2] if len(sys.argv) > 2 else "transcribe"
    
    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return
    
    # 加载模型
    model, processor, device = load_model_with_correct_version()
    
    if model is None:
        print("\\n❌ 无法加载模型。请检查:")
        print("  1. 网络连接")
        print("  2. transformers版本 (需要4.50.1)")
        print("  3. 模型文件完整性")
        return
    
    # 进行推理
    result = transcribe_with_official_format(model, processor, device, audio_file, task)
    
    if result:
        print(f"\\n✅ 处理成功完成!")
        print(f"   📁 输入文件: {audio_file}")
        print(f"   🎯 任务类型: {task}")
        print(f"   📝 识别结果: {result}")
        
        # 保存结果到文件
        output_file = f"{audio_file.rsplit('.', 1)[0]}_{task}_result.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"音频文件: {audio_file}\\n")
            f.write(f"任务类型: {task}\\n")
            f.write(f"结果: {result}\\n")
        print(f"   💾 结果已保存: {output_file}")
    else:
        print("\\n❌ 推理失败")

if __name__ == "__main__":
    main()
