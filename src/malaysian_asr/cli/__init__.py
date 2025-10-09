"""
Command-line interface for Malaysian ASR package.
"""

import os
import sys
from pathlib import Path
from ..core.transcriber import MERaLiONTranscriber
from ..core.batch_transcriber import BatchTranscriber
from ..core.benchmark import RTFBenchmark
from ..utils.accuracy import evaluate_accuracy
from ..utils.model_downloader import download_model, setup_directories
from ..utils.check_dependencies import main as check_dependencies_main


def transcribe():
    """CLI entry point for single file transcription."""
    print("=" * 60)
    print("🎙️  MERaLiON 语音转录器")
    print("=" * 60)
    
    # Initialize transcriber
    transcriber = MERaLiONTranscriber()
    
    # Load model
    if not transcriber.load_model():
        print("无法加载模型，请检查:")
        print("1. transformers版本是否为4.50.1")
        print("2. 网络连接是否正常")
        print("3. 磁盘空间是否充足")
        return
    
    # Process command line arguments
    if len(sys.argv) < 2:
        # Auto-find audio files
        audio_files = [f for f in os.listdir('.') 
                      if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        if not audio_files:
            print("\\n用法:")
            print("  malaysian-asr-transcribe <音频文件1> [音频文件2] [...]")
            print("\\n支持格式: .wav, .mp3, .flac, .m4a")
            return
        
        print(f"🔍 发现 {len(audio_files)} 个音频文件")
        
        # Ask if process all files
        if len(audio_files) > 1:
            response = input(f"是否转录所有 {len(audio_files)} 个文件? (y/n): ")
            if response.lower() != 'y':
                audio_files = audio_files[:1]  # Only process first
        
    else:
        audio_files = sys.argv[1:]
    
    # Start transcription
    print(f"\\n📝 开始转录 {len(audio_files)} 个文件...")
    
    if len(audio_files) == 1:
        # Single file transcription
        result = transcriber.transcribe(audio_files[0])
        if result:
            print(f"\\n🎉 转录完成!")
            print(f"📁 文件: {audio_files[0]}")
            print(f"📄 结果: {result}")
    else:
        # Batch transcription
        results = transcriber.transcribe_batch(audio_files)
        
        # Show summary
        print(f"\\n📊 转录汇总:")
        successful = sum(1 for r in results.values() if r is not None)
        print(f"✅ 成功: {successful}/{len(audio_files)}")
        
        for file, result in results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {file}")


def batch_transcribe():
    """CLI entry point for batch directory transcription."""
    print("=" * 70)
    print("📁 MERaLiON 批量目录转录器")
    print("=" * 70)
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("用法:")
        print("  malaysian-asr-batch <音频目录> [输出文件.txt]")
        print("\\n示例:")
        print("  malaysian-asr-batch /opt/data/vocals_only_mono")
        print("  malaysian-asr-batch /opt/data/vocals_only_mono results.txt")
        return
    
    directory_path = sys.argv[1]
    
    # Output filename
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        # Default output filename
        dir_name = Path(directory_path).name
        output_file = f"{dir_name}_transcripts.txt"
    
    print(f"📁 音频目录: {directory_path}")
    print(f"📝 输出文件: {output_file}")
    
    # Initialize transcriber
    transcriber = BatchTranscriber()
    
    # Load model
    if not transcriber.load_model():
        return
    
    # Start batch transcription
    transcriber.batch_transcribe_directory(directory_path, output_file)


def benchmark():
    """CLI entry point for RTF benchmark."""
    print("=" * 70)
    print("⚡ MERaLiON-2-10B-ASR RTF性能测试")
    print("=" * 70)
    
    # Check command line arguments
    if len(sys.argv) < 2:
        audio_files = [f for f in os.listdir('.') 
                      if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        if not audio_files:
            print("用法: malaysian-asr-benchmark <音频文件>")
            return
        
        audio_file = audio_files[0]
        print(f"🔍 使用发现的音频文件: {audio_file}")
    else:
        audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return
    
    # Initialize benchmark
    benchmark = RTFBenchmark()
    
    # Load model
    if not benchmark.load_model():
        return
    
    # Run performance test
    num_runs = 3 if len(sys.argv) < 3 else int(sys.argv[2])
    result = benchmark.benchmark_transcription(audio_file, num_runs)
    
    if result:
        # Save test report
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


def download_model_cli():
    """CLI entry point for model download."""
    print("MERaLiON Malaysian ASR 模型下载器")
    print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    # Check disk space
    import shutil
    cache_dir = "./model_cache"
    total, used, free = shutil.disk_usage(cache_dir)
    print(f"可用磁盘空间: {free // (1024**3)} GB")
    
    if free < 30 * 1024**3:  # Less than 30GB
        print("⚠️  警告: 可用磁盘空间不足30GB，模型下载可能失败")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Download model
    result = download_model()
    
    if result[0] is not None or result[1] is not None:
        print("\n✅ 下载完成！")
        print(f"模型缓存位置: {cache_dir}")
        
        if result[0] is not None:
            # Model was loaded successfully
            print("✅ 模型和处理器加载成功！")
            
            # Check if result[0] is a tuple (model, processor) or a pipeline
            if isinstance(result[0], tuple):
                model, processor = result[0]
                print("📦 使用直接加载的模型和处理器")
                print("✅ 模型已准备就绪，可以进行转录")
            else:
                # It's a pipeline
                print("✅ Pipeline创建成功，可以进行推理测试")
                
                # Check if there are audio files for testing
                audio_files = [f for f in os.listdir('.') if f.endswith(('.wav', '.mp3', '.flac'))]
                if audio_files:
                    print(f"发现音频文件: {audio_files}")
                    test_file = audio_files[0]
                    from .utils.model_downloader import test_audio_inference
                    test_audio_inference(result[0], test_file)
                else:
                    print("没有找到音频文件进行测试")
                    from .utils.model_downloader import create_simple_test
                    test_file = create_simple_test()
                    if test_file:
                        test_audio_inference(result[0], test_file)
        else:
            # Only model files downloaded, model loading failed
            print("⚠️  模型文件已下载，但模型加载失败")
            print("这可能是由于transformers版本兼容性问题")
            print("模型文件位置: ./model_cache")
            print("您可以尝试使用以下命令进行转录:")
            print("  make transcribe AUDIO_FILE=your_audio.wav")
            print("  make batch DIRECTORY=/path/to/audio/directory")
    else:
        print("❌ 下载失败")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确保有足够的磁盘空间")
        print("3. 尝试使用VPN或镜像源")


def check_dependencies():
    """CLI entry point for dependency checking."""
    check_dependencies_main()


def calculate_accuracy_cli():
    """CLI entry point for accuracy calculation."""
    # Install jieba if not already installed
    try:
        import jieba
    except ImportError:
        print("正在安装jieba分词库...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "jieba"])
        import jieba
    
    reference_file = "./src/malaysian_asr/data/test_sample_transcribe_result.txt"
    hypothesis_file = "./src/malaysian_asr/data/vocals_only_mono_transcripts_formatted.txt"
    
    result = evaluate_accuracy(reference_file, hypothesis_file)
    
    if result:
        print(f"\n" + "=" * 60)
        print("🎯 最终评估结果:")
        print(f"   词级准确率: {result['overall_word_accuracy']:.3f} ({result['overall_word_accuracy']*100:.1f}%)")
        print(f"   字符准确率: {result['overall_char_accuracy']:.3f} ({result['overall_char_accuracy']*100:.1f}%)")
        print(f"   Word Error Rate: {result['wer']:.3f}")
        print(f"   评估文件数: {result['total_files']}")
