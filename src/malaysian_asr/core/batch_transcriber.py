"""
Batch transcriber module for processing multiple audio files.
"""

import os
import time
from pathlib import Path
from .transcriber import MERaLiONTranscriber


class BatchTranscriber(MERaLiONTranscriber):
    """Batch transcriber for processing directories of audio files."""
    
    def transcribe_single(self, audio_path: str) -> tuple:
        """
        Transcribe a single audio file (internal method).
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (transcription_result, duration) or (None, 0) if failed
        """
        try:
            # Load audio (16kHz)
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            duration = len(audio_array) / 16000
            
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
            
            # Move to device
            if self.device == "cuda":
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                        if value.dtype == torch.float32:
                            inputs[key] = inputs[key].to(torch.bfloat16)
            
            # Generate transcription
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
            
            result = response[0] if response else ""
            
            # Clean result (remove extra whitespace and newlines)
            result = result.strip().replace('\\n', ' ').replace('\\r', ' ')
            while '  ' in result:  # Remove extra spaces
                result = result.replace('  ', ' ')
            
            return result, duration
            
        except Exception as e:
            print(f"❌ 转录失败 {audio_path}: {e}")
            return None, 0
    
    def batch_transcribe_directory(self, directory_path: str, output_file: str):
        """
        Batch transcribe all audio files in a directory.
        
        Args:
            directory_path: Path to directory containing audio files
            output_file: Path to output file for results
        """
        # Supported audio formats
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma']
        
        # Find all audio files
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
        
        # Sort by filename
        audio_files = sorted(audio_files)
        
        print(f"📁 找到 {len(audio_files)} 个音频文件")
        print(f"📝 结果将保存到: {output_file}")
        
        # Start batch transcription
        results = []
        total_duration = 0
        successful_count = 0
        failed_files = []
        
        start_time = time.time()
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\\n🎵 [{i:3d}/{len(audio_files)}] 正在处理: {audio_file.name}")
            
            # Transcribe
            transcribe_start = time.time()
            result, duration = self.transcribe_single(str(audio_file))
            transcribe_time = time.time() - transcribe_start
            
            if result is not None:
                # Successful transcription
                successful_count += 1
                total_duration += duration
                
                # Keep only filename (without path)
                filename = audio_file.name
                
                # Format: audio_filename|transcription_result
                line = f"{filename}|{result}"
                results.append(line)
                
                # Show result
                print(f"   ✅ ({duration:.1f}s, {transcribe_time:.2f}s) {result[:100]}{'...' if len(result) > 100 else ''}")
                
                # Real-time save (prevent data loss if program is interrupted)
                with open(output_file, 'w', encoding='utf-8') as f:
                    for line in results:
                        f.write(line + '\\n')
            else:
                # Transcription failed
                failed_files.append(audio_file.name)
                print(f"   ❌ 转录失败")
        
        # Final statistics
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
        
        # Show failed files
        if failed_files:
            print(f"\\n❌ 转录失败的文件:")
            for filename in failed_files:
                print(f"   - {filename}")
        
        print(f"\\n💾 所有结果已保存到: {output_file}")
        print(f"格式: 音频文件名|转录结果")
