#!/usr/bin/env python3
"""
ASR准确率计算工具
以test_sample_transcribe_result.txt为标准答案，计算转录准确率
使用Word Error Rate (WER) 和字符准确率等指标
"""

import re
import sys
import jieba
from difflib import SequenceMatcher

def clean_text(text):
    """清理文本：移除标点符号和括号内容"""
    # 移除括号及其内容，包括中英文括号
    text = re.sub(r'\([^)]*\)', '', text)  # 英文括号
    text = re.sub(r'（[^）]*）', '', text)  # 中文括号
    text = re.sub(r'\[[^\]]*\]', '', text)  # 方括号
    text = re.sub(r'<[^>]*>', '', text)    # 尖括号 (如<Speaker1>)
    
    # 移除常见标点符号
    punctuation = ',.!?;:"\'""''，。！？；：、…—·'
    for p in punctuation:
        text = text.replace(p, '')
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_filename(filename):
    """标准化文件名"""
    filename = filename.replace('_vocals.wav', '.wav')
    return filename

def parse_file(file_path):
    """解析文件"""
    data = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            # 按｜分割（支持中英文竖线）
            separator = '｜' if '｜' in line else '|'
            if separator in line:
                parts = line.split(separator, 1)
                if len(parts) == 2:
                    filename = parts[0].strip()
                    content = parts[1].strip()
                    
                    # 标准化文件名
                    normalized_filename = normalize_filename(filename)
                    
                    # 清理内容
                    cleaned_content = clean_text(content)
                    data[normalized_filename] = cleaned_content
    
    except Exception as e:
        print(f"❌ 读取文件失败 {file_path}: {e}")
        return {}
    
    return data

def segment_chinese_text(text):
    """分词处理中文文本"""
    # 使用jieba进行中文分词
    words = list(jieba.cut(text))
    # 过滤空白词
    words = [w.strip() for w in words if w.strip()]
    return words

def calculate_wer(reference_words, hypothesis_words):
    """计算Word Error Rate (WER)"""
    # 使用动态规划计算编辑距离
    r_len = len(reference_words)
    h_len = len(hypothesis_words)
    
    # 创建距离矩阵
    d = [[0 for _ in range(h_len + 1)] for _ in range(r_len + 1)]
    
    # 初始化第一行和第一列
    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j
    
    # 填充距离矩阵
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if reference_words[i-1] == hypothesis_words[j-1]:
                d[i][j] = d[i-1][j-1]  # 匹配，不需要操作
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,      # 删除
                    d[i][j-1] + 1,      # 插入
                    d[i-1][j-1] + 1     # 替换
                )
    
    # 编辑距离 = 错误数
    edit_distance = d[r_len][h_len]
    
    # WER = 错误数 / 参考词数
    if r_len == 0:
        return 1.0 if h_len > 0 else 0.0
    
    wer = edit_distance / r_len
    return wer

def calculate_character_accuracy(ref_text, hyp_text):
    """计算字符级准确率"""
    # 将文本转为字符列表
    ref_chars = list(ref_text.replace(' ', ''))
    hyp_chars = list(hyp_text.replace(' ', ''))
    
    # 使用SequenceMatcher计算相似度
    matcher = SequenceMatcher(None, ref_chars, hyp_chars)
    similarity = matcher.ratio()
    
    return similarity

def evaluate_accuracy(reference_file, hypothesis_file):
    """评估准确率"""
    
    print("🎯 ASR转录准确率评估工具")
    print("=" * 60)
    print(f"标准答案文件: {reference_file}")
    print(f"待评估文件: {hypothesis_file}")
    print("=" * 60)
    
    # 解析文件
    print("📖 正在解析文件...")
    ref_data = parse_file(reference_file)
    hyp_data = parse_file(hypothesis_file)
    
    print(f"✅ 标准答案: {len(ref_data)} 条记录")
    print(f"✅ 待评估: {len(hyp_data)} 条记录")
    
    # 找到共同文件
    common_files = set(ref_data.keys()) & set(hyp_data.keys())
    print(f"📊 共同文件: {len(common_files)} 个")
    
    if not common_files:
        print("❌ 没有找到共同的文件进行比较")
        return
    
    # 计算各种准确率指标
    total_wer = 0
    total_char_acc = 0
    total_ref_words = 0
    total_hyp_words = 0
    total_ref_chars = 0
    
    results = []
    
    print(f"\n🔍 开始评估...")
    
    for filename in sorted(common_files):
        ref_text = ref_data[filename]
        hyp_text = hyp_data[filename]
        
        # 分词
        ref_words = segment_chinese_text(ref_text)
        hyp_words = segment_chinese_text(hyp_text)
        
        # 计算WER
        wer = calculate_wer(ref_words, hyp_words)
        
        # 计算字符准确率
        char_acc = calculate_character_accuracy(ref_text, hyp_text)
        
        # 统计
        total_wer += wer * len(ref_words)
        total_char_acc += char_acc
        total_ref_words += len(ref_words)
        total_hyp_words += len(hyp_words)
        total_ref_chars += len(ref_text.replace(' ', ''))
        
        results.append({
            'filename': filename,
            'ref_text': ref_text,
            'hyp_text': hyp_text,
            'ref_words': len(ref_words),
            'hyp_words': len(hyp_words),
            'wer': wer,
            'word_accuracy': 1 - wer,
            'char_accuracy': char_acc
        })
    
    # 计算总体指标
    overall_wer = total_wer / total_ref_words if total_ref_words > 0 else 0
    overall_word_acc = 1 - overall_wer
    overall_char_acc = total_char_acc / len(common_files)
    
    print(f"\n📊 准确率评估结果:")
    print("-" * 60)
    print(f"📈 总体词级准确率: {overall_word_acc:.3f} ({overall_word_acc*100:.1f}%)")
    print(f"📈 总体字符准确率: {overall_char_acc:.3f} ({overall_char_acc*100:.1f}%)")
    print(f"📉 Word Error Rate (WER): {overall_wer:.3f} ({overall_wer*100:.1f}%)")
    print(f"📊 总参考词数: {total_ref_words}")
    print(f"📊 总输出词数: {total_hyp_words}")
    
    # 显示最好和最差的例子
    results.sort(key=lambda x: x['word_accuracy'], reverse=True)
    
    print(f"\n🏆 词级准确率最高的5个文件:")
    for i, result in enumerate(results[:5], 1):
        print(f"  {i}. {result['filename']}: {result['word_accuracy']:.3f} ({result['word_accuracy']*100:.1f}%)")
    
    print(f"\n📉 词级准确率最低的5个文件:")
    for i, result in enumerate(results[-5:], 1):
        acc = result['word_accuracy']
        print(f"  {i}. {result['filename']}: {acc:.3f} ({acc*100:.1f}%)")
        if acc < 0.8:  # 显示准确率低于80%的详细信息
            print(f"     参考: {result['ref_text'][:60]}{'...' if len(result['ref_text']) > 60 else ''}")
            print(f"     识别: {result['hyp_text'][:60]}{'...' if len(result['hyp_text']) > 60 else ''}")
    
    # 准确率分布统计
    high_acc = sum(1 for r in results if r['word_accuracy'] >= 0.9)
    medium_acc = sum(1 for r in results if 0.7 <= r['word_accuracy'] < 0.9)
    low_acc = sum(1 for r in results if r['word_accuracy'] < 0.7)
    
    print(f"\n📊 准确率分布:")
    print(f"   高准确率 (≥90%): {high_acc} 个 ({high_acc/len(results)*100:.1f}%)")
    print(f"   中等准确率 (70-90%): {medium_acc} 个 ({medium_acc/len(results)*100:.1f}%)")
    print(f"   低准确率 (<70%): {low_acc} 个 ({low_acc/len(results)*100:.1f}%)")
    
    return {
        'overall_word_accuracy': overall_word_acc,
        'overall_char_accuracy': overall_char_acc,
        'wer': overall_wer,
        'total_files': len(common_files),
        'high_accuracy_files': high_acc,
        'medium_accuracy_files': medium_acc,
        'low_accuracy_files': low_acc
    }

def main():
    # 安装jieba如果还没有
    try:
        import jieba
    except ImportError:
        print("正在安装jieba分词库...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "jieba"])
        import jieba
    
    reference_file = "/opt/data/Malaysian_ASR/test_sample_transcribe_result.txt"
    hypothesis_file = "/opt/data/Malaysian_ASR/vocals_only_mono_transcripts_formatted.txt"
    
    result = evaluate_accuracy(reference_file, hypothesis_file)
    
    if result:
        print(f"\n" + "=" * 60)
        print("🎯 最终评估结果:")
        print(f"   词级准确率: {result['overall_word_accuracy']:.3f} ({result['overall_word_accuracy']*100:.1f}%)")
        print(f"   字符准确率: {result['overall_char_accuracy']:.3f} ({result['overall_char_accuracy']*100:.1f}%)")
        print(f"   Word Error Rate: {result['wer']:.3f}")
        print(f"   评估文件数: {result['total_files']}")

if __name__ == "__main__":
    main()
