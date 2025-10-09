#!/usr/bin/env python3
"""
MERaLiON-2-10B-ASR 专用转录脚本
只实现语音转录功能，简化版本
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

class MERaLiONTranscriber:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.repo_id = "MERaLiON/MERaLiON-2-10B-ASR"
    
    def load_model(self):
        """加载转录模型"""
        print("🚀 正在加载MERaLiON转录模型...")
        
        try:
            # 检测设备
            if torch.cuda.is_available():
                print("✅ 使用GPU加速")
                self.device = "cuda"
                torch_dtype = torch.bfloat16
            else:
                print("💻 使用CPU模式")
                self.device = "cpu"
                torch_dtype = torch.float32
            
            # 加载处理器
            print("📥 加载处理器...")
            self.processor = AutoProcessor.from_pretrained(
                self.repo_id, 
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            
            # 加载模型
            print("📥 加载模型...")
            if self.device == "cuda":
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.repo_id,
                    use_safetensors=True,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    cache_dir=CACHE_DIR
                )
            else:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.repo_id,
                    use_safetensors=True,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    cache_dir=CACHE_DIR
                )
            
            print("✅ 模型加载完成！")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def transcribe(self, audio_path):
        """转录音频文件"""
        try:
            print(f"🎵 正在转录: {audio_path}")
            
            # 加载音频 (16kHz)
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            duration = len(audio_array) / 16000
            print(f"   ⏱️  时长: {duration:.2f}秒")
            
            # 转录专用提示词
            prompt_template = "Instruction: Please transcribe this speech. \\nFollow the text instruction based on the following audio: <SpeechHere>"
            
            # 创建对话格式
            conversation = [
                [{"role": "user", "content": prompt_template}]
            ]
            
            # 处理输入
            chat_prompt = self.processor.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(text=chat_prompt, audios=[audio_array])
            
            # 移动到设备
            if self.device == "cuda":
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                        if value.dtype == torch.float32:
                            inputs[key] = inputs[key].to(torch.bfloat16)
            
            # 生成转录结果
            print("   🧠 正在转录...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # 解码结果
            generated_ids = outputs[:, inputs['input_ids'].size(1):]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            result = response[0] if response else "转录失败"
            print(f"✅ 转录完成: {result}")
            
            return result
            
        except Exception as e:
            print(f"❌ 转录失败: {e}")
            return None
    
    def transcribe_batch(self, audio_files):
        """批量转录多个音频文件"""
        results = {}
        
        for i, audio_file in enumerate(audio_files):
            print(f"\\n=== 处理第 {i+1}/{len(audio_files)} 个文件 ===")
            
            if not os.path.exists(audio_file):
                print(f"❌ 文件不存在: {audio_file}")
                results[audio_file] = None
                continue
            
            result = self.transcribe(audio_file)
            results[audio_file] = result
            
            # 保存结果
            if result:
                output_file = f"{audio_file.rsplit('.', 1)[0]}_transcript.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"音频文件: {audio_file}\\n")
                    f.write(f"转录结果: {result}\\n")
                print(f"💾 结果已保存: {output_file}")
        
        return results

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
