"""
Core transcriber module for MERaLiON-2-10B-ASR model.
"""

import os
import torch
import librosa
import warnings
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

warnings.filterwarnings("ignore")

# Default cache directory - use local directory instead of /opt/data
DEFAULT_CACHE_DIR = "./model_cache"


class MERaLiONTranscriber:
    """MERaLiON-2-10B-ASR transcriber for single file transcription."""
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the transcriber.
        
        Args:
            cache_dir: Directory to cache the model files
        """
        self.model = None
        self.processor = None
        self.device = None
        self.repo_id = "MERaLiON/MERaLiON-2-10B-ASR"
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        
        # Set environment variables
        os.environ["HF_HOME"] = self.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
    
    def load_model(self):
        """Load the transcription model."""
        print("🚀 正在加载MERaLiON转录模型...")
        
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
            print("📥 加载处理器...")
            self.processor = AutoProcessor.from_pretrained(
                self.repo_id, 
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            
            # Load model using official approach (CPU vs GPU)
            print("📥 加载模型...")
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
            
            print("✅ 模型加载完成！")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            print(f"🎵 正在转录: {audio_path}")
            
            # Load audio (16kHz)
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            duration = len(audio_array) / 16000
            print(f"   ⏱️  时长: {duration:.2f}秒")
            
            # Transcription prompt
            prompt_template = "Instruction: Please transcribe this speech. \\nFollow the text instruction based on the following audio: <SpeechHere>"
            
            # Create conversation format
            conversation = [
                [{"role": "user", "content": prompt_template}]
            ]
            
            # Process input
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
            
            # Generate transcription
            print("   🧠 正在转录...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode result
            generated_ids = outputs[:, inputs['input_ids'].size(1):]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            result = response[0] if response else "转录失败"
            print(f"✅ 转录完成: {result}")
            
            return result
            
        except Exception as e:
            print(f"❌ 转录失败: {e}")
            return None
    
    def transcribe_batch(self, audio_files: list) -> dict:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            Dictionary mapping file paths to transcription results
        """
        results = {}
        
        for i, audio_file in enumerate(audio_files):
            print(f"\\n=== 处理第 {i+1}/{len(audio_files)} 个文件 ===")
            
            if not os.path.exists(audio_file):
                print(f"❌ 文件不存在: {audio_file}")
                results[audio_file] = None
                continue
            
            result = self.transcribe(audio_file)
            results[audio_file] = result
            
            # Save result
            if result:
                output_file = f"{audio_file.rsplit('.', 1)[0]}_transcript.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"音频文件: {audio_file}\\n")
                    f.write(f"转录结果: {result}\\n")
                print(f"💾 结果已保存: {output_file}")
        
        return results
