# MERaLiON-2-10B-ASR 完整使用指南

## 🎯 成功解决方案总结

### ✅ 问题解决
通过以下步骤成功解决了MERaLiON-2-10B-ASR模型的下载和推理问题：

1. **磁盘空间清理** - 释放了8GB+空间
2. **版本兼容性修复** - 安装了正确的transformers==4.50.1
3. **官方推理格式** - 使用了正确的prompt模板和处理流程

### 📚 参考文档
- [MERaLiON-2-10B-ASR主页](https://huggingface.co/MERaLiON/MERaLiON-2-10B-ASR)
- [vLLM插件要求](https://huggingface.co/MERaLiON/MERaLiON-2-10B/blob/main/vllm_plugin_meralion2/readme.md)

## 🚀 模型信息

### 基本信息
- **模型名称**: MERaLiON-2-10B-ASR
- **参数规模**: 100亿参数
- **开发方**: I2R, A*STAR, Singapore
- **许可证**: MERaLiON Public License

### 性能特点
- **比Whisper-large-v3提升5-30%**
- **支持新加坡4种官方语言**: 英语、华语、马来语、泰米尔语
- **支持东南亚语言**: 印尼语、泰语、越南语
- **处理代码转换和方言**

## 💻 技术要求

### 关键版本要求
```bash
transformers==4.50.1  # 必须是这个版本！
torch>=1.9.0
librosa
```

### 硬件要求
- **GPU**: 推荐使用CUDA（显著加速）
- **内存**: 至少16GB RAM
- **存储**: 约25GB用于模型文件

## 📥 安装和下载

### 1. 环境准备
```bash
# 安装正确版本的依赖
pip install transformers==4.50.1
pip install torch torchaudio librosa

# 设置缓存目录
export HF_HOME="/opt/data/Malaysian_ASR/model_cache"
export TRANSFORMERS_CACHE="/opt/data/Malaysian_ASR/model_cache"
```

### 2. 模型下载
运行我们的下载脚本：
```bash
python download_model.py
```

模型会自动下载到指定的缓存目录。

## 🎙️ 推理使用

### 基本转录
```bash
python correct_inference.py your_audio.wav
```

### 翻译功能
```bash
# 翻译成中文
python correct_inference.py your_audio.wav translate_chinese

# 翻译成英文
python correct_inference.py your_audio.wav translate_english
```

### 支持的音频格式
- .wav (推荐)
- .mp3
- .flac
- .m4a

### 音频要求
- **采样率**: 16000 Hz
- **声道**: 单声道（自动转换）
- **最佳长度**: 30秒以内（ASR任务）
- **最大长度**: 300秒（5分钟）

## 📊 测试结果

### 转录示例
**输入音频**: 10.6秒英语对话
**输出**: 
> <Speaker1>: Yes, exactly correct. Okay ya, but just before that no problem. Can I check Supria? Actually, does your child have any tuition for math at the moment?

### 翻译示例
**英译中结果**:
> 是的，没错。好的，但是在那之前，没问题。我可以问一下苏普里亚吗？你的孩子现在有数学补习班吗？

## 📁 文件结构

```
/opt/data/Malaysian_ASR/
├── download.py                     # 原始pipeline脚本
├── download_model.py              # 完整下载脚本
├── correct_inference.py           # 正确版本推理脚本 ⭐
├── official_inference.py          # 官方格式脚本
├── cpu_inference.py              # CPU版本脚本
├── model_cache/                   # 模型缓存目录
│   └── models--MERaLiON--MERaLiON-2-10B-ASR/
├── test_sample.wav               # 测试音频
├── INSTALLATION_GUIDE.md         # 安装指南
└── README.md                     # 使用说明
```

## 🔧 故障排除

### 常见问题

1. **版本不兼容错误**
   - 确保使用 `transformers==4.50.1`
   - 重新安装：`pip install transformers==4.50.1`

2. **磁盘空间不足**
   - 清理缓存：`rm -rf ~/.cache/pip/*`
   - 清理日志：`sudo journalctl --vacuum-size=50M`

3. **GPU内存不足**
   - 使用CPU模式
   - 减少batch_size
   - 使用较小的max_new_tokens

4. **网络下载问题**
   - 使用稳定网络
   - 考虑使用代理
   - 模型支持断点续传

## 🎯 最佳实践

1. **推理优化**
   - 使用GPU加速
   - 预处理音频到16kHz
   - 保持音频长度在30秒以内

2. **提示词优化**
   - 使用官方推荐的prompt模板
   - 明确指定任务类型
   - 保持模板格式不变

3. **资源管理**
   - 定期清理缓存
   - 监控磁盘空间
   - 合理设置批处理大小

## 🏆 成功标志

当看到以下输出时，说明一切正常：
```
✅ 模型加载成功
🎉 推理完成!
📄 结果: [识别内容]
💾 结果已保存: [文件名]
```

## 📞 技术支持

如遇问题，请检查：
1. transformers版本是否为4.50.1
2. 磁盘空间是否充足
3. 音频文件格式是否支持
4. 网络连接是否稳定

---

*最后更新: 2025年10月9日*
*版本: v1.0 - 稳定版本*
