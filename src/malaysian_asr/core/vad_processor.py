#!/usr/bin/env python3
"""
VAD-based audio processing script for Malaysian ASR API.
Processes long audio files by detecting voice activity, splitting by channels,
and processing chunks through the API.
"""

import os
import sys
import time
import requests
import librosa
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import json
from datetime import datetime, timedelta

# Add VAD dependencies
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("⚠️  webrtcvad not available. Install with: pip install webrtcvad")

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available. Install with: pip install scipy")


class VADProcessor:
    """Voice Activity Detection processor for audio segmentation."""
    
    def __init__(self, sample_rate: int = 16000, frame_duration: int = 30, 
                 min_segment_duration: float = 0.5, max_gap_duration: float = 0.3):
        """
        Initialize VAD processor.
        
        Args:
            sample_rate: Audio sample rate
            frame_duration: Frame duration in ms (10, 20, or 30)
            min_segment_duration: Minimum segment duration in seconds
            max_gap_duration: Maximum gap to merge segments in seconds
        """
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self.min_segment_duration = min_segment_duration
        self.max_gap_duration = max_gap_duration
        
        if VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(1)  # Less aggressive (was 2)
        else:
            self.vad = None
            print("⚠️  Using simple energy-based VAD (webrtcvad not available)")
    
    def detect_voice_activity(self, audio: np.ndarray, channel: int = None) -> List[Tuple[float, float]]:
        """
        Detect voice activity segments in audio.
        
        Args:
            audio: Audio array (mono or stereo)
            channel: Channel index for stereo audio (0 or 1), None for mono
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        if len(audio.shape) > 1 and channel is not None:
            # Extract specific channel
            audio = audio[channel]
        elif len(audio.shape) > 1:
            # Use first channel for mono processing
            audio = audio[0]
        
        if self.vad is not None:
            segments = self._webrtc_vad(audio)
        else:
            segments = self._energy_vad(audio)
        
        # Filter and merge segments
        segments = self._filter_and_merge_segments(segments)
        return segments
    
    def _webrtc_vad(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Use WebRTC VAD for voice activity detection."""
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        segments = []
        is_speech = False
        speech_start = 0
        
        for i in range(0, len(audio_int16) - self.frame_size, self.frame_size):
            frame = audio_int16[i:i + self.frame_size]
            
            try:
                is_speech_frame = self.vad.is_speech(frame.tobytes(), self.sample_rate)
            except:
                # Fallback to energy-based detection
                is_speech_frame = self._is_speech_energy(frame)
            
            current_time = i / self.sample_rate
            
            if is_speech_frame and not is_speech:
                # Speech starts
                speech_start = current_time
                is_speech = True
            elif not is_speech_frame and is_speech:
                # Speech ends
                segments.append((speech_start, current_time))
                is_speech = False
        
        # Handle case where speech continues to end
        if is_speech:
            segments.append((speech_start, len(audio_int16) / self.sample_rate))
        
        return segments
    
    def _filter_and_merge_segments(self, segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Filter out short segments and merge nearby ones."""
        if not segments:
            return segments
        
        # Sort segments by start time
        segments = sorted(segments, key=lambda x: x[0])
        
        filtered_segments = []
        current_start, current_end = segments[0]
        
        for start, end in segments[1:]:
            # Check if segments should be merged (gap is small)
            if start - current_end <= self.max_gap_duration:
                # Merge segments
                current_end = end
            else:
                # Check if current segment is long enough
                if current_end - current_start >= self.min_segment_duration:
                    filtered_segments.append((current_start, current_end))
                
                # Start new segment
                current_start, current_end = start, end
        
        # Add the last segment if it's long enough
        if current_end - current_start >= self.min_segment_duration:
            filtered_segments.append((current_start, current_end))
        
        return filtered_segments
    
    def _energy_vad(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Simple energy-based voice activity detection."""
        # Calculate frame energy
        frame_size = int(self.sample_rate * 0.025)  # 25ms frames
        hop_size = int(self.sample_rate * 0.010)    # 10ms hop
        
        energy = []
        times = []
        
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            frame_energy = np.sum(frame ** 2)
            energy.append(frame_energy)
            times.append(i / self.sample_rate)
        
        energy = np.array(energy)
        times = np.array(times)
        
        # Adaptive threshold
        energy_threshold = np.mean(energy) + 2 * np.std(energy)
        
        # Find speech segments
        speech_mask = energy > energy_threshold
        
        # Smooth the mask (remove short gaps)
        min_speech_duration = 0.1  # 100ms
        min_gap_duration = 0.05    # 50ms
        
        smoothed_mask = self._smooth_speech_mask(speech_mask, times, min_speech_duration, min_gap_duration)
        
        # Convert mask to segments
        segments = []
        in_speech = False
        speech_start = 0
        
        for i, is_speech in enumerate(smoothed_mask):
            if is_speech and not in_speech:
                speech_start = times[i]
                in_speech = True
            elif not is_speech and in_speech:
                segments.append((speech_start, times[i]))
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech:
            segments.append((speech_start, times[-1]))
        
        return segments
    
    def _is_speech_energy(self, frame: np.ndarray) -> bool:
        """Simple energy-based speech detection for a frame."""
        energy = np.sum(frame ** 2)
        return energy > np.mean(frame) + 2 * np.std(frame)
    
    def _smooth_speech_mask(self, mask: np.ndarray, times: np.ndarray, 
                           min_speech_duration: float, min_gap_duration: float) -> np.ndarray:
        """Smooth speech mask by removing short gaps and segments."""
        smoothed = mask.copy()
        
        # Remove short speech segments
        i = 0
        while i < len(smoothed):
            if smoothed[i]:
                # Find end of current speech segment
                j = i
                while j < len(smoothed) and smoothed[j]:
                    j += 1
                
                # Check if segment is too short
                if j < len(smoothed) and times[j] - times[i] < min_speech_duration:
                    smoothed[i:j] = False
                
                i = j
            else:
                i += 1
        
        # Fill short gaps
        i = 0
        while i < len(smoothed):
            if not smoothed[i]:
                # Find end of current gap
                j = i
                while j < len(smoothed) and not smoothed[j]:
                    j += 1
                
                # Check if gap is too short
                if j < len(smoothed) and times[j] - times[i] < min_gap_duration:
                    smoothed[i:j] = True
                
                i = j
            else:
                i += 1
        
        return smoothed


class AudioProcessor:
    """Main audio processor for VAD-based transcription."""
    
    def __init__(self, api_url: str = "http://192.168.1.41:54321"):
        """
        Initialize audio processor.
        
        Args:
            api_url: URL of the Malaysian ASR API server
        """
        self.api_url = api_url
        # Use less sensitive VAD with longer minimum segments
        self.vad_processor = VADProcessor(
            min_segment_duration=1.0,  # Minimum 1 second segments
            max_gap_duration=0.5       # Merge gaps up to 0.5 seconds
        )
        self.temp_dir = Path("./temp_chunks")
        self.temp_dir.mkdir(exist_ok=True)
    
    def process_audio_file(self, audio_path: str, output_path: str = None) -> str:
        """
        Process a long audio file using VAD and API.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to output transcript file
            
        Returns:
            Path to output transcript file
        """
        print(f"🎵 Processing audio file: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=False)
        duration = len(audio[0]) / sr if len(audio.shape) > 1 else len(audio) / sr
        
        print(f"   📊 Duration: {duration:.2f} seconds")
        print(f"   🎧 Channels: {audio.shape[0] if len(audio.shape) > 1 else 1}")
        
        # Detect voice activity for each channel
        all_segments = []
        
        if len(audio.shape) > 1:
            # Stereo audio - process each channel
            for channel in range(audio.shape[0]):
                print(f"   🔍 Detecting voice activity in channel {channel + 1}...")
                segments = self.vad_processor.detect_voice_activity(audio, channel)
                
                for start_time, end_time in segments:
                    all_segments.append({
                        'channel': channel,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time
                    })
                
                print(f"      Found {len(segments)} speech segments")
        else:
            # Mono audio
            print(f"   🔍 Detecting voice activity...")
            segments = self.vad_processor.detect_voice_activity(audio)
            
            for start_time, end_time in segments:
                all_segments.append({
                    'channel': 0,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })
            
            print(f"      Found {len(segments)} speech segments")
        
        # Sort segments by start time
        all_segments.sort(key=lambda x: x['start_time'])
        
        print(f"   📝 Total segments to process: {len(all_segments)}")
        
        # Process segments through API
        results = []
        for i, segment in enumerate(all_segments):
            print(f"   🔄 Processing segment {i+1}/{len(all_segments)} "
                  f"(Channel {segment['channel']+1}, "
                  f"{segment['start_time']:.1f}s-{segment['end_time']:.1f}s)")
            
            # Extract audio segment
            start_sample = int(segment['start_time'] * sr)
            end_sample = int(segment['end_time'] * sr)
            
            if len(audio.shape) > 1:
                segment_audio = audio[segment['channel'], start_sample:end_sample]
            else:
                segment_audio = audio[start_sample:end_sample]
            
            # Save temporary file
            temp_file = self.temp_dir / f"chunk_{i:04d}_ch{segment['channel']}.wav"
            import soundfile as sf
            sf.write(str(temp_file), segment_audio, sr)
            
            # Process through API
            try:
                result = self._process_chunk_api(str(temp_file))
                if result:
                    results.append({
                        'channel': segment['channel'],
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'text': result.strip()
                    })
                    print(f"      ✅ Transcribed: {result[:50]}...")
                else:
                    print(f"      ❌ Transcription failed")
            except Exception as e:
                print(f"      ❌ Error: {e}")
            
            # Clean up temp file
            temp_file.unlink()
        
        # Format and save results
        if output_path is None:
            output_path = Path(audio_path).with_suffix('.txt')
        
        self._save_results(results, str(output_path))
        
        print(f"✅ Processing complete! Results saved to: {output_path}")
        return str(output_path)
    
    def _process_chunk_api(self, audio_file: str) -> str:
        """Process a single audio chunk through the API."""
        try:
            with open(audio_file, 'rb') as f:
                files = {'file': (Path(audio_file).name, f, 'audio/wav')}
                response = requests.post(f"{self.api_url}/upload", files=files, timeout=60)
            
            if response.status_code != 200:
                print(f"      ❌ Upload failed: {response.status_code}")
                return None
            
            task_data = response.json()
            task_id = task_data['task_id']
            
            # Wait for completion
            max_wait = 120  # 2 minutes max
            wait_time = 0
            
            while wait_time < max_wait:
                time.sleep(2)
                wait_time += 2
                
                progress_response = requests.get(f"{self.api_url}/progress")
                if progress_response.status_code == 200:
                    progress = progress_response.json()
                    
                    if progress['current_task'] is None and progress['queue_length'] == 0:
                        # Task completed, get result
                        result_response = requests.get(f"{self.api_url}/tasks/{task_id}/result")
                        if result_response.status_code == 200:
                            result_data = result_response.json()
                            return result_data['result']
                        else:
                            print(f"      ❌ Failed to get result: {result_response.status_code}")
                            return None
                
            print(f"      ⏰ Timeout waiting for completion")
            return None
            
        except Exception as e:
            print(f"      ❌ API error: {e}")
            return None
    
    def _save_results(self, results: List[Dict], output_path: str):
        """Save formatted results to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Transcription Results\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                start_time = self._format_time(result['start_time'])
                speaker = f"Speaker {result['channel'] + 1}"
                
                f.write(f"[{start_time}] {speaker}: {result['text']}\n\n")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="VAD-based audio processing for Malaysian ASR")
    parser.add_argument("audio_file", help="Path to input audio file")
    parser.add_argument("-o", "--output", help="Output transcript file path")
    parser.add_argument("-u", "--url", default="http://192.168.1.41:54321", 
                       help="API server URL")
    
    args = parser.parse_args()
    
    if not Path(args.audio_file).exists():
        print(f"❌ Audio file not found: {args.audio_file}")
        return 1
    
    # Check API availability
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ API server not available at {args.url}")
            return 1
    except Exception as e:
        print(f"❌ Cannot connect to API server: {e}")
        return 1
    
    # Process audio
    processor = AudioProcessor(args.url)
    try:
        output_path = processor.process_audio_file(args.audio_file, args.output)
        print(f"\n🎉 Success! Transcript saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
