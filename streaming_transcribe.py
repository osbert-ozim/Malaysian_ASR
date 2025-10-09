#!/usr/bin/env python3
"""
MERaLiON-2-10B-ASR 流式推理脚本
实现实时流式语音转录
"""

import os
import sys
import time
import torch
import librosa
import numpy as np
import warnings
from collections import deque
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import threading
import queue

warnings.filterwarnings("ignore")

# 设置缓存目录
CACHE_DIR = "/opt/data/Malaysian_ASR/model_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

class StreamingTranscriber:
    def __init__(self, chunk_duration=3.0, overlap_duration=0.5):
        """
        流式转录器
        
        Args:
            chunk_duration: 每个音频块的时长（秒）
            overlap_duration: 重叠时长（秒），用于保持上下文
        """
        self.model = None
        self.processor = None
        self.device = None
        self.repo_id = "MERaLiON/MERaLiON-2-10B-ASR"
        
        # 流式参数
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = 16000
        self.chunk_samples = int(chunk_duration * self.sample_rate)
        self.overlap_samples = int(overlap_duration * self.sample_rate)
        
        # 音频缓冲区
        self.audio_buffer = deque()
        self.results_queue = queue.Queue()
        
        print(f"🎵 流式参数:")
        print(f"   📊 音频块时长: {chunk_duration}秒")
        print(f"   🔄 重叠时长: {overlap_duration}秒")
        print(f"   📈 每块样本数: {self.chunk_samples}")
    
    def load_model(self):
        """加载模型"""
        print("🚀 正在加载流式转录模型...")
        
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
    
    def transcribe_chunk(self, audio_chunk):
        """转录单个音频块"""
        try:
            # 准备输入
            prompt_template = "Instruction: Please transcribe this speech. \\nFollow the text instruction based on the following audio: <SpeechHere>"
            conversation = [
                [{"role": "user", "content": prompt_template}]
            ]
            
            chat_prompt = self.processor.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(text=chat_prompt, audios=[audio_chunk])
            
            # 移动到设备
            if self.device == "cuda":
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                        if value.dtype == torch.float32:
                            inputs[key] = inputs[key].to(torch.bfloat16)
            
            # 推理
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=128,  # 减少tokens以提高速度
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # 解码
            generated_ids = outputs[:, inputs['input_ids'].size(1):]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            return response[0] if response else ""
            
        except Exception as e:
            print(f"❌ 块转录失败: {e}")
            return ""
    
    def stream_from_file(self, audio_file):
        """从音频文件进行流式转录"""
        print(f"\\n🎵 开始流式转录: {audio_file}")
        
        # 加载完整音频
        audio_data, sr = librosa.load(audio_file, sr=self.sample_rate)
        total_duration = len(audio_data) / self.sample_rate
        print(f"   📊 音频总时长: {total_duration:.2f}秒")
        
        # 计算处理块数
        step_samples = self.chunk_samples - self.overlap_samples
        num_chunks = (len(audio_data) - self.overlap_samples) // step_samples + 1
        print(f"   📦 将处理 {num_chunks} 个音频块")
        
        results = []
        processing_times = []
        
        print(f"\\n🎙️  开始实时转录:")
        print(f"{'='*60}")
        
        for i in range(num_chunks):
            # 计算当前块的开始和结束位置
            start_idx = i * step_samples
            end_idx = min(start_idx + self.chunk_samples, len(audio_data))
            
            # 提取音频块
            audio_chunk = audio_data[start_idx:end_idx]
            
            # 计算时间戳
            start_time = start_idx / self.sample_rate
            end_time = end_idx / self.sample_rate
            chunk_duration = (end_idx - start_idx) / self.sample_rate
            
            print(f"🔄 块 {i+1}/{num_chunks} [{start_time:.1f}s-{end_time:.1f}s] ({chunk_duration:.1f}s)")
            
            # 转录计时
            transcribe_start = time.time()
            result = self.transcribe_chunk(audio_chunk)
            transcribe_time = time.time() - transcribe_start
            processing_times.append(transcribe_time)
            
            # 计算RTF
            chunk_rtf = transcribe_time / chunk_duration if chunk_duration > 0 else 0
            
            # 显示结果
            if result.strip():
                print(f"   ✅ ({transcribe_time:.2f}s, RTF:{chunk_rtf:.3f}) {result}")
                results.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': chunk_duration,
                    'processing_time': transcribe_time,
                    'rtf': chunk_rtf,
                    'text': result
                })
            else:
                print(f"   ⚪ ({transcribe_time:.2f}s, RTF:{chunk_rtf:.3f}) [静音或无识别]")
            
            # 模拟实时处理延迟
            # time.sleep(0.1)  # 可以注释掉以获得最快处理速度
        
        # 统计信息
        print(f"\\n{'='*60}")
        print(f"📊 流式转录完成统计:")
        
        total_processing_time = sum(processing_times)
        avg_processing_time = total_processing_time / len(processing_times)
        avg_rtf = total_processing_time / total_duration
        
        print(f"   🎵 音频总时长: {total_duration:.2f}秒")
        print(f"   ⏱️  总处理时间: {total_processing_time:.2f}秒")
        print(f"   📈 平均每块处理时间: {avg_processing_time:.3f}秒")
        print(f"   🎯 整体RTF: {avg_rtf:.3f}")
        
        if avg_rtf < 1.0:
            print(f"   🚀 性能: 比实时快 {1/avg_rtf:.1f}x")
        else:
            print(f"   ⚠️  性能: 比实时慢 {avg_rtf:.1f}x")
        
        return results
    
    def simulate_realtime_stream(self, audio_file, realtime_factor=1.0):
        """模拟实时流式处理"""
        print(f"\\n🔴 模拟实时流式转录 (速度: {realtime_factor}x)")
        
        # 加载音频
        audio_data, sr = librosa.load(audio_file, sr=self.sample_rate)
        total_duration = len(audio_data) / self.sample_rate
        
        print(f"   📊 音频时长: {total_duration:.2f}秒")
        print(f"   🎮 播放速度: {realtime_factor}x实时")
        
        # 模拟实时播放和转录
        step_samples = self.chunk_samples - self.overlap_samples
        chunk_play_time = step_samples / self.sample_rate / realtime_factor
        
        print(f"\\n🎙️  开始模拟实时转录:")
        print(f"{'='*60}")
        
        start_total = time.time()
        
        for i in range(0, len(audio_data) - self.overlap_samples, step_samples):
            chunk_start_time = time.time()
            
            # 提取音频块
            end_idx = min(i + self.chunk_samples, len(audio_data))
            audio_chunk = audio_data[i:end_idx]
            
            # 音频时间戳
            audio_start = i / self.sample_rate
            audio_end = end_idx / self.sample_rate
            
            # 转录
            result = self.transcribe_chunk(audio_chunk)
            processing_time = time.time() - chunk_start_time
            
            # 显示结果
            elapsed = time.time() - start_total
            print(f"[{elapsed:06.2f}s] {audio_start:05.1f}s-{audio_end:05.1f}s: {result}")
            
            # 等待到下一个块的时间（模拟实时）
            time_to_wait = chunk_play_time - processing_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)
        
        print(f"\\n✅ 模拟实时转录完成")

def main():
    """主函数"""
    print("=" * 70)
    print("🌊 MERaLiON-2-10B-ASR 流式转录器")
    print("=" * 70)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        audio_files = [f for f in os.listdir('.') 
                      if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        if not audio_files:
            print("用法:")
            print("  python streaming_transcribe.py <音频文件> [模式]")
            print("\\n模式:")
            print("  batch  - 批量流式转录 (默认)")
            print("  realtime - 模拟实时流式转录")
            return
        
        audio_file = audio_files[0]
        print(f"🔍 使用发现的音频文件: {audio_file}")
    else:
        audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return
    
    mode = sys.argv[2] if len(sys.argv) > 2 else "batch"
    
    # 初始化流式转录器
    transcriber = StreamingTranscriber(
        chunk_duration=3.0,    # 3秒音频块
        overlap_duration=0.5   # 0.5秒重叠
    )
    
    # 加载模型
    if not transcriber.load_model():
        return
    
    # 执行流式转录
    if mode == "realtime":
        transcriber.simulate_realtime_stream(audio_file, realtime_factor=1.0)
    else:
        results = transcriber.stream_from_file(audio_file)
        
        # 保存结果
        if results:
            output_file = f"{audio_file.rsplit('.', 1)[0]}_streaming_transcript.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"流式转录结果: {audio_file}\\n")
                f.write(f"{'='*50}\\n")
                for r in results:
                    f.write(f"[{r['start_time']:.1f}s-{r['end_time']:.1f}s] {r['text']}\\n")
            
            print(f"\\n💾 流式转录结果已保存: {output_file}")

if __name__ == "__main__":
    main()
