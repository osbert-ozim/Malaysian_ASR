#!/usr/bin/env python3
"""
MERaLiON-2-10B-ASR RTF性能测试脚本
测试转录的实时性能指标
"""

import os
import sys
import time
import torch
import librosa
import warnings
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

warnings.filterwarnings("ignore")

# 设置缓存目录
CACHE_DIR = "/opt/data/Malaysian_ASR/model_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

class RTFBenchmark:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.repo_id = "MERaLiON/MERaLiON-2-10B-ASR"
        self.model_loaded = False
    
    def load_model(self):
        """加载模型并计时"""
        if self.model_loaded:
            return True
            
        print("🚀 正在加载MERaLiON模型...")
        load_start = time.time()
        
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
            processor_start = time.time()
            self.processor = AutoProcessor.from_pretrained(
                self.repo_id, 
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            processor_time = time.time() - processor_start
            
            # 加载模型
            model_start = time.time()
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
            model_time = time.time() - model_start
            
            total_load_time = time.time() - load_start
            
            print(f"⏱️  加载时间统计:")
            print(f"   处理器加载: {processor_time:.2f}秒")
            print(f"   模型加载: {model_time:.2f}秒")
            print(f"   总加载时间: {total_load_time:.2f}秒")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def benchmark_transcription(self, audio_path, num_runs=3):
        """测试转录性能并计算RTF"""
        try:
            print(f"\\n🎵 性能测试: {audio_path}")
            
            # 加载音频
            audio_load_start = time.time()
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            audio_load_time = time.time() - audio_load_start
            
            audio_duration = len(audio_array) / 16000
            print(f"   📊 音频时长: {audio_duration:.2f}秒")
            print(f"   📥 音频加载时间: {audio_load_time:.3f}秒")
            
            # 准备输入（只做一次）
            prompt_template = "Instruction: Please transcribe this speech. \\nFollow the text instruction based on the following audio: <SpeechHere>"
            conversation = [
                [{"role": "user", "content": prompt_template}]
            ]
            
            preprocessing_start = time.time()
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
            
            preprocessing_time = time.time() - preprocessing_start
            print(f"   🔄 预处理时间: {preprocessing_time:.3f}秒")
            
            # 多次运行测试
            inference_times = []
            results = []
            
            print(f"\\n🧪 开始 {num_runs} 次推理测试...")
            
            for i in range(num_runs):
                print(f"   第 {i+1}/{num_runs} 次推理...")
                
                # 推理计时
                inference_start = time.time()
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # 解码结果（不计入推理时间）
                generated_ids = outputs[:, inputs['input_ids'].size(1):]
                response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                result = response[0] if response else "转录失败"
                results.append(result)
                
                print(f"      ⏱️  推理时间: {inference_time:.3f}秒")
            
            # 计算统计数据
            avg_inference_time = sum(inference_times) / len(inference_times)
            min_inference_time = min(inference_times)
            max_inference_time = max(inference_times)
            
            # 计算RTF
            avg_rtf = avg_inference_time / audio_duration
            min_rtf = min_inference_time / audio_duration
            max_rtf = max_inference_time / audio_duration
            
            # 总处理时间（包括预处理）
            total_processing_time = preprocessing_time + avg_inference_time
            total_rtf = total_processing_time / audio_duration
            
            # 显示结果
            print(f"\\n📈 性能统计报告:")
            print(f"   🎵 音频时长: {audio_duration:.2f}秒")
            print(f"   🔄 预处理时间: {preprocessing_time:.3f}秒")
            print(f"   🧠 平均推理时间: {avg_inference_time:.3f}秒 (范围: {min_inference_time:.3f}~{max_inference_time:.3f}秒)")
            print(f"   ⏱️  总处理时间: {total_processing_time:.3f}秒")
            print(f"")
            print(f"📊 RTF指标:")
            print(f"   🎯 纯推理RTF: {avg_rtf:.3f} (范围: {min_rtf:.3f}~{max_rtf:.3f})")
            print(f"   🎯 总处理RTF: {total_rtf:.3f}")
            print(f"")
            
            # RTF解读
            if total_rtf < 1.0:
                speed_desc = f"比实时快 {1/total_rtf:.1f}x"
                performance = "🚀 优秀"
            elif total_rtf < 2.0:
                speed_desc = f"比实时慢 {total_rtf:.1f}x"
                performance = "✅ 良好"
            else:
                speed_desc = f"比实时慢 {total_rtf:.1f}x"
                performance = "⚠️  较慢"
            
            print(f"🎉 性能评价: {performance}")
            print(f"   📈 处理速度: {speed_desc}")
            print(f"   💡 解读: RTF={total_rtf:.3f} 意味着处理{audio_duration:.1f}秒音频需要{total_processing_time:.1f}秒")
            
            print(f"\\n📝 转录结果:")
            print(f"   {results[0]}")
            
            return {
                'audio_duration': audio_duration,
                'preprocessing_time': preprocessing_time,
                'avg_inference_time': avg_inference_time,
                'total_processing_time': total_processing_time,
                'rtf_inference': avg_rtf,
                'rtf_total': total_rtf,
                'transcription': results[0]
            }
            
        except Exception as e:
            print(f"❌ 性能测试失败: {e}")
            return None

def main():
    """主函数"""
    print("=" * 70)
    print("⚡ MERaLiON-2-10B-ASR RTF性能测试")
    print("=" * 70)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        audio_files = [f for f in os.listdir('.') 
                      if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        if not audio_files:
            print("用法: python rtf_benchmark.py <音频文件>")
            return
        
        audio_file = audio_files[0]
        print(f"🔍 使用发现的音频文件: {audio_file}")
    else:
        audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return
    
    # 初始化测试器
    benchmark = RTFBenchmark()
    
    # 加载模型
    if not benchmark.load_model():
        return
    
    # 运行性能测试
    num_runs = 3 if len(sys.argv) < 3 else int(sys.argv[2])
    result = benchmark.benchmark_transcription(audio_file, num_runs)
    
    if result:
        # 保存测试报告
        report_file = f"{audio_file.rsplit('.', 1)[0]}_rtf_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"MERaLiON-2-10B-ASR RTF性能报告\\n")
            f.write(f"{'='*50}\\n")
            f.write(f"音频文件: {audio_file}\\n")
            f.write(f"音频时长: {result['audio_duration']:.2f}秒\\n")
            f.write(f"预处理时间: {result['preprocessing_time']:.3f}秒\\n")
            f.write(f"平均推理时间: {result['avg_inference_time']:.3f}秒\\n")
            f.write(f"总处理时间: {result['total_processing_time']:.3f}秒\\n")
            f.write(f"纯推理RTF: {result['rtf_inference']:.3f}\\n")
            f.write(f"总处理RTF: {result['rtf_total']:.3f}\\n")
            f.write(f"转录结果: {result['transcription']}\\n")
        
        print(f"\\n💾 性能报告已保存: {report_file}")

if __name__ == "__main__":
    main()
