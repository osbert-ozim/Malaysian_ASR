# MERaLiON-2-10B-ASR 完整使用指南

## 📋 目录概览

这是MERaLiON-2-10B-ASR马来西亚语音识别模型的完整工具包，包含模型下载、转录、评估和分析等功能。

---

## 🚀 快速开始

### 最简单的使用方法

1. **单文件转录**:
```bash
python transcribe.py your_audio.wav
```

2. **批量目录转录**:
```bash
python batch_transcribe.py /path/to/audio/directory
```

---

## 📁 文件说明

### 🎙️ 核心转录工具

| 文件名 | 用途 | 使用方法 |
|--------|------|----------|
| **transcribe.py** | 🌟**主要转录工具** - 单文件或批量转录 | `python transcribe.py audio.wav` |
| **batch_transcribe.py** | 📁**批量目录转录** - 处理整个目录的音频文件 | `python batch_transcribe.py /path/to/directory` |
| **correct_inference.py** | 🔧**标准推理脚本** - 支持转录和翻译 | `python correct_inference.py audio.wav [task]` |
| **streaming_transcribe.py** | 🌊**流式转录** - 实时流式处理 | `python streaming_transcribe.py audio.wav [mode]` |

### ⚙️ 模型管理工具

| 文件名 | 用途 | 使用方法 |
|--------|------|----------|
| **download_model.py** | 📥**模型下载器** - 首次下载和安装模型 | `python download_model.py` |
| **model_cache/** | 💾**模型存储目录** - 包含已下载的模型文件 | 自动管理，无需手动操作 |

### 📊 评估和分析工具

| 文件名 | 用途 | 使用方法 |
|--------|------|----------|
| **calculate_accuracy.py** | 📈**准确率计算** - 评估转录准确性 | `python calculate_accuracy.py` |
| **rtf_benchmark.py** | ⚡**性能测试** - 测试实时因子RTF | `python rtf_benchmark.py audio.wav` |
| **compare_transcripts.py** | 🔍**结果对比** - 比较不同转录结果 | `python compare_transcripts.py` |
| **compare_transcripts_fixed.py** | 🔍**修正版对比工具** - 处理文件名差异 | `python compare_transcripts_fixed.py` |

### 🛠️ 辅助工具

| 文件名 | 用途 | 使用方法 |
|--------|------|----------|
| **format_transcripts.py** | 📝**格式化工具** - 转换转录结果格式 | `python format_transcripts.py` |

### 📚 文档和报告

| 文件名 | 用途 | 内容 |
|--------|------|------|
| **README.md** | 📖**主要说明文档** | 本文件 |
| **COMPLETE_GUIDE.md** | 📘**完整指南** | 详细的安装和使用指南 |
| **TRANSCRIBE_GUIDE.md** | 🎯**转录专用指南** | 专门的转录功能说明 |
| **ASR_Accuracy_Report.md** | 📊**准确率评估报告** | 详细的性能分析报告 |
| **ASR_Quick_Report.md** | ⚡**快速报告** | 核心评估结果摘要 |

### 📄 示例和测试文件

| 文件名 | 用途 | 说明 |
|--------|------|------|
| **test_sample.wav** | 🎵**测试音频** | 用于测试的音频样本 |
| **test_sample_*.txt** | 📝**测试结果** | 各种转录和翻译的测试输出 |
| **vocals_only_mono_transcripts*.txt** | 📋**批量转录结果** | 目录转录的结果文件 |

---

## 🎯 ASR转录使用指南

### 1. 环境准备

#### 安装依赖
```bash
# 安装正确版本的transformers (关键!)
pip install transformers==4.50.1

# 安装其他依赖
pip install torch torchaudio librosa jieba
```

#### 首次下载模型
```bash
python download_model.py
```

### 2. 转录方法

#### 🌟 方法1: 简单转录（推荐）
```bash
# 转录单个文件
python transcribe.py your_audio.wav

# 自动发现并转录当前目录的音频文件
python transcribe.py
```

#### 📁 方法2: 批量目录转录
```bash
# 转录整个目录，输出格式: 文件名|转录结果
python batch_transcribe.py /path/to/audio/directory

# 指定输出文件
python batch_transcribe.py /path/to/audio/directory output.txt
```

#### 🔧 方法3: 标准推理（支持翻译）
```bash
# 转录
python correct_inference.py audio.wav transcribe

# 翻译成中文
python correct_inference.py audio.wav translate_chinese

# 翻译成英文
python correct_inference.py audio.wav translate_english
```

#### 🌊 方法4: 流式转录
```bash
# 批量流式转录（最快）
python streaming_transcribe.py audio.wav batch

# 模拟实时流式转录
python streaming_transcribe.py audio.wav realtime
```

### 3. 性能测试

#### ⚡ RTF性能测试
```bash
python rtf_benchmark.py audio.wav
```

#### 📈 准确率评估
```bash
# 需要标准答案文件
python calculate_accuracy.py
```

---

## 🎵 支持的音频格式

- ✅ **WAV** (.wav) - 推荐格式
- ✅ **MP3** (.mp3)
- ✅ **FLAC** (.flac)
- ✅ **M4A** (.m4a)
- ✅ **OGG** (.ogg)

### 音频要求
- **采样率**: 16000 Hz（自动转换）
- **声道**: 单声道（自动转换）
- **推荐时长**: 30秒以内获得最佳效果
- **最大时长**: 300秒 (5分钟)

---

## 🚀 模型性能

### 性能指标
- **RTF**: 0.288（比实时快3.5倍）
- **准确率**: 86.6%（词级准确率）
- **字符准确率**: 91.7%
- **支持GPU加速**: 是

### 支持语言
- 🇸🇬 新加坡英语（含Singlish）
- 🇨🇳 华语（中文）
- 🇲🇾 马来语
- 🇮🇳 泰米尔语
- 🇮🇩 印尼语
- 🇹🇭 泰语
- 🇻🇳 越南语

---

## 📊 使用示例

### 基本转录示例
```bash
# 最简单的使用
python transcribe.py audio.wav

# 批量处理目录
python batch_transcribe.py /opt/data/vocals_only_mono

# 性能测试
python rtf_benchmark.py audio.wav
```

### 输出示例
```
🎵 正在转录: audio.wav
   ⏱️  时长: 10.60秒
   🧠 正在转录...
✅ 转录完成: 这是一段测试音频的转录结果

🎉 转录完成!
💾 结果已保存: audio_transcript.txt
```

---

## ⚙️ 系统要求

### 最低要求
- **Python**: 3.7+
- **内存**: 16GB RAM
- **存储**: 30GB可用空间
- **transformers**: 4.50.1（必须）

### 推荐配置
- **Python**: 3.9+
- **内存**: 32GB RAM
- **GPU**: NVIDIA GPU with CUDA
- **存储**: 50GB+ SSD

---

## 🔧 故障排除

### 常见问题

#### Q: 模型加载失败
```bash
# 检查transformers版本
pip show transformers
# 应该是4.50.1

# 重新安装正确版本
pip install transformers==4.50.1
```

#### Q: 磁盘空间不足
```bash
# 清理缓存
rm -rf ~/.cache/pip/*
rm -rf ~/.cache/huggingface/*
```

#### Q: GPU不可用
```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 安装CUDA版PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 性能优化
- ✅ 使用GPU加速（推荐）
- ✅ 确保音频质量良好
- ✅ 音频长度控制在30秒以内
- ✅ 使用SSD存储

---

## 📝 输出格式

### 转录结果格式
```
# 单文件转录
audio_transcript.txt:
音频文件: audio.wav
转录结果: [转录内容]

# 批量转录
directory_transcripts.txt:
file1.wav|转录结果1
file2.wav|转录结果2
...
```

---

## 🎯 最佳实践

1. **首次使用**: 运行 `python download_model.py` 下载模型
2. **日常转录**: 使用 `python transcribe.py your_audio.wav`
3. **批量处理**: 使用 `python batch_transcribe.py /path/to/directory`
4. **性能测试**: 使用 `python rtf_benchmark.py` 评估性能
5. **准确率评估**: 准备标准答案后使用 `python calculate_accuracy.py`

---

## 📞 技术支持

- 📖 [官方文档](https://huggingface.co/MERaLiON/MERaLiON-2-10B-ASR)
- 🔧 [vLLM插件要求](https://huggingface.co/MERaLiON/MERaLiON-2-10B/blob/main/vllm_plugin_meralion2/readme.md)
- 📋 查看完整指南: `COMPLETE_GUIDE.md`

---

## 📜 许可证

MERaLiON Public License - 请查看模型官方许可证条款

---

*最后更新: 2025年10月9日*