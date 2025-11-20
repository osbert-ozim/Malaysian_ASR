"""
RTF (Real-Time Factor) benchmark module for performance testing.
"""

import os
import time
import torch
import librosa
import warnings
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

warnings.filterwarnings("ignore")

# Default cache directory - use local directory instead of /opt/data
DEFAULT_CACHE_DIR = "./model_cache"


class RTFBenchmark:
    """RTF performance testing for MERaLiON-2-10B-ASR model."""
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the benchmark.
        
        Args:
            cache_dir: Directory to cache the model files
        """
        self.model = None
        self.processor = None
        self.device = None
        self.repo_id = "MERaLiON/MERaLiON-2-10B-ASR"
        self.model_loaded = False
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        
        # Set environment variables
        os.environ["HF_HOME"] = self.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
    
    def load_model(self):
        """Load the model and time the process."""
        if self.model_loaded:
            return True
            
        print("🚀 正在加载MERaLiON模型...")
        load_start = time.time()
        
        try:
            # Detect device
            if torch.cuda.is_available():
                print("✅ 使用GPU加速")
                self.device = "cuda"
                torch_dtype = torch.bfloat16
            else:
                print("💻 使用CPU模式")
                self.device = "cpu"
                torch_dtype = torch.float32
            
            # Load processor
            processor_start = time.time()
            self.processor = AutoProcessor.from_pretrained(
                self.repo_id, 
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            processor_time = time.time() - processor_start
            
            # Load model using official approach (CPU vs GPU)
            model_start = time.time()
            if self.device == "cuda":
                # GPU version from official code
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.repo_id,
                    use_safetensors=True,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    cache_dir=self.cache_dir
                ).to(self.device)
            else:
                # CPU version from official code
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.repo_id,
                    use_safetensors=True,
                    trust_remote_code=True,
                    cache_dir=self.cache_dir
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
    
    def benchmark_transcription(self, audio_path: str, num_runs: int = 3) -> dict:
        """
        Test transcription performance and calculate RTF.
        
        Args:
            audio_path: Path to the audio file
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            print(f"\\n🎵 性能测试: {audio_path}")
            
            # Load audio
            audio_load_start = time.time()
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            audio_load_time = time.time() - audio_load_start
            
            audio_duration = len(audio_array) / 16000
            print(f"   📊 音频时长: {audio_duration:.2f}秒")
            print(f"   📥 音频加载时间: {audio_load_time:.3f}秒")
            
            # Prepare input (only once)
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
            
            # Move to device (official GPU approach)
            if self.device == "cuda":
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                        if value.dtype == torch.float32:
                            inputs[key] = inputs[key].to(torch.bfloat16)
            
            preprocessing_time = time.time() - preprocessing_start
            print(f"   🔄 预处理时间: {preprocessing_time:.3f}秒")
            
            # Multiple runs for testing
            inference_times = []
            results = []
            
            print(f"\\n🧪 开始 {num_runs} 次推理测试...")
            
            for i in range(num_runs):
                print(f"   第 {i+1}/{num_runs} 次推理...")
                
                # Inference timing
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
                
                # Decode result (not counted in inference time)
                generated_ids = outputs[:, inputs['input_ids'].size(1):]
                response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                result = response[0] if response else "转录失败"
                results.append(result)
                
                print(f"      ⏱️  推理时间: {inference_time:.3f}秒")
            
            # Calculate statistics
            avg_inference_time = sum(inference_times) / len(inference_times)
            min_inference_time = min(inference_times)
            max_inference_time = max(inference_times)
            
            # Calculate RTF
            avg_rtf = avg_inference_time / audio_duration
            min_rtf = min_inference_time / audio_duration
            max_rtf = max_inference_time / audio_duration
            
            # Total processing time (including preprocessing)
            total_processing_time = preprocessing_time + avg_inference_time
            total_rtf = total_processing_time / audio_duration
            
            # Display results
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
            
            # RTF interpretation
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
