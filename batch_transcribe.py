#!/usr/bin/env python3
"""
MERaLiON-2-10B-ASR 批量目录转录脚本
对指定目录下的所有音频文件进行转录
输出格式: 音频文件名|转录结果
"""

import os
import sys
import torch
import librosa
import warnings
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pathlib import Path
import time

warnings.filterwarnings("ignore")

# 设置缓存目录
CACHE_DIR = "/opt/data/Malaysian_ASR/model_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

class BatchTranscriber:
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
    
    def transcribe_single(self, audio_path):
        """转录单个音频文件"""
        try:
            # 加载音频 (16kHz)
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            duration = len(audio_array) / 16000
            
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
            
            result = response[0] if response else ""
            
            # 清理结果（去掉多余的空白和换行）
            result = result.strip().replace('\\n', ' ').replace('\\r', ' ')
            while '  ' in result:  # 去掉多余空格
                result = result.replace('  ', ' ')
            
            return result, duration
            
        except Exception as e:
            print(f"❌ 转录失败 {audio_path}: {e}")
            return None, 0
    
    def batch_transcribe_directory(self, directory_path, output_file):
        """批量转录目录下的音频文件"""
        
        # 支持的音频格式
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma']
        
        # 查找所有音频文件
        audio_files = []
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"❌ 目录不存在: {directory_path}")
            return
        
        for ext in audio_extensions:
            audio_files.extend(directory.glob(f"*{ext}"))
            audio_files.extend(directory.glob(f"*{ext.upper()}"))
        
        if not audio_files:
            print(f"❌ 在目录 {directory_path} 中未找到音频文件")
            print(f"支持的格式: {', '.join(audio_extensions)}")
            return
        
        # 按文件名排序
        audio_files = sorted(audio_files)
        
        print(f"📁 找到 {len(audio_files)} 个音频文件")
        print(f"📝 结果将保存到: {output_file}")
        
        # 开始批量转录
        results = []
        total_duration = 0
        successful_count = 0
        failed_files = []
        
        start_time = time.time()
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\\n🎵 [{i:3d}/{len(audio_files)}] 正在处理: {audio_file.name}")
            
            # 转录
            transcribe_start = time.time()
            result, duration = self.transcribe_single(str(audio_file))
            transcribe_time = time.time() - transcribe_start
            
            if result is not None:
                # 成功转录
                successful_count += 1
                total_duration += duration
                
                # 只保留文件名（不含路径）
                filename = audio_file.name
                
                # 格式: 音频文件名|转录结果
                line = f"{filename}|{result}"
                results.append(line)
                
                # 显示结果
                print(f"   ✅ ({duration:.1f}s, {transcribe_time:.2f}s) {result[:100]}{'...' if len(result) > 100 else ''}")
                
                # 实时保存（防止程序中断丢失结果）
                with open(output_file, 'w', encoding='utf-8') as f:
                    for line in results:
                        f.write(line + '\\n')
            else:
                # 转录失败
                failed_files.append(audio_file.name)
                print(f"   ❌ 转录失败")
        
        # 最终统计
        total_time = time.time() - start_time
        
        print(f"\\n{'='*60}")
        print(f"📊 批量转录完成统计:")
        print(f"   📁 处理目录: {directory_path}")
        print(f"   📄 输出文件: {output_file}")
        print(f"   🎵 总文件数: {len(audio_files)}")
        print(f"   ✅ 成功转录: {successful_count}")
        print(f"   ❌ 失败数量: {len(failed_files)}")
        print(f"   ⏱️  总处理时间: {total_time:.1f}秒")
        print(f"   🎶 音频总时长: {total_duration:.1f}秒")
        
        if total_duration > 0:
            rtf = total_time / total_duration
            print(f"   📈 整体RTF: {rtf:.3f}")
            if rtf < 1.0:
                print(f"   🚀 平均速度: 比实时快 {1/rtf:.1f}x")
            else:
                print(f"   ⏳ 平均速度: 比实时慢 {rtf:.1f}x")
        
        # 显示失败的文件
        if failed_files:
            print(f"\\n❌ 转录失败的文件:")
            for filename in failed_files:
                print(f"   - {filename}")
        
        print(f"\\n💾 所有结果已保存到: {output_file}")
        print(f"格式: 音频文件名|转录结果")

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
