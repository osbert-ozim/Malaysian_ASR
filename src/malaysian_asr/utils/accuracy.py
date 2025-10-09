"""
Accuracy calculation utilities for ASR evaluation.
"""

import re
import sys
import jieba
from difflib import SequenceMatcher


def clean_text(text: str) -> str:
    """Clean text: remove punctuation and bracket content."""
    # Remove brackets and their content, including Chinese and English brackets
    text = re.sub(r'\([^)]*\)', '', text)  # English brackets
    text = re.sub(r'（[^）]*）', '', text)  # Chinese brackets
    text = re.sub(r'\[[^\]]*\]', '', text)  # Square brackets
    text = re.sub(r'<[^>]*>', '', text)    # Angle brackets (like <Speaker1>)
    
    # Remove common punctuation
    punctuation = ',.!?;:"\'""''，。！？；：、…—·'
    for p in punctuation:
        text = text.replace(p, '')
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_filename(filename: str) -> str:
    """Normalize filename."""
    filename = filename.replace('_vocals.wav', '.wav')
    return filename


def parse_file(file_path: str) -> dict:
    """Parse file."""
    data = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            # Split by | (support Chinese and English vertical bars)
            separator = '｜' if '｜' in line else '|'
            if separator in line:
                parts = line.split(separator, 1)
                if len(parts) == 2:
                    filename = parts[0].strip()
                    content = parts[1].strip()
                    
                    # Normalize filename
                    normalized_filename = normalize_filename(filename)
                    
                    # Clean content
                    cleaned_content = clean_text(content)
                    data[normalized_filename] = cleaned_content
    
    except Exception as e:
        print(f"❌ 读取文件失败 {file_path}: {e}")
        return {}
    
    return data


def segment_chinese_text(text: str) -> list:
    """Segment Chinese text."""
    # Use jieba for Chinese word segmentation
    words = list(jieba.cut(text))
    # Filter empty words
    words = [w.strip() for w in words if w.strip()]
    return words


def calculate_wer(reference_words: list, hypothesis_words: list) -> float:
    """Calculate Word Error Rate (WER)."""
    # Use dynamic programming to calculate edit distance
    r_len = len(reference_words)
    h_len = len(hypothesis_words)
    
    # Create distance matrix
    d = [[0 for _ in range(h_len + 1)] for _ in range(r_len + 1)]
    
    # Initialize first row and first column
    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j
    
    # Fill distance matrix
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if reference_words[i-1] == hypothesis_words[j-1]:
                d[i][j] = d[i-1][j-1]  # Match, no operation needed
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,      # Delete
                    d[i][j-1] + 1,      # Insert
                    d[i-1][j-1] + 1     # Replace
                )
    
    # Edit distance = number of errors
    edit_distance = d[r_len][h_len]
    
    # WER = number of errors / number of reference words
    if r_len == 0:
        return 1.0 if h_len > 0 else 0.0
    
    wer = edit_distance / r_len
    return wer


def calculate_character_accuracy(ref_text: str, hyp_text: str) -> float:
    """Calculate character-level accuracy."""
    # Convert text to character lists
    ref_chars = list(ref_text.replace(' ', ''))
    hyp_chars = list(hyp_text.replace(' ', ''))
    
    # Use SequenceMatcher to calculate similarity
    matcher = SequenceMatcher(None, ref_chars, hyp_chars)
    similarity = matcher.ratio()
    
    return similarity


def evaluate_accuracy(reference_file: str, hypothesis_file: str) -> dict:
    """Evaluate accuracy."""
    
    print("🎯 ASR转录准确率评估工具")
    print("=" * 60)
    print(f"标准答案文件: {reference_file}")
    print(f"待评估文件: {hypothesis_file}")
    print("=" * 60)
    
    # Parse files
    print("📖 正在解析文件...")
    ref_data = parse_file(reference_file)
    hyp_data = parse_file(hypothesis_file)
    
    print(f"✅ 标准答案: {len(ref_data)} 条记录")
    print(f"✅ 待评估: {len(hyp_data)} 条记录")
    
    # Find common files
    common_files = set(ref_data.keys()) & set(hyp_data.keys())
    print(f"📊 共同文件: {len(common_files)} 个")
    
    if not common_files:
        print("❌ 没有找到共同的文件进行比较")
        return {}
    
    # Calculate various accuracy metrics
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
        
        # Word segmentation
        ref_words = segment_chinese_text(ref_text)
        hyp_words = segment_chinese_text(hyp_text)
        
        # Calculate WER
        wer = calculate_wer(ref_words, hyp_words)
        
        # Calculate character accuracy
        char_acc = calculate_character_accuracy(ref_text, hyp_text)
        
        # Statistics
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
    
    # Calculate overall metrics
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
    
    # Show best and worst examples
    results.sort(key=lambda x: x['word_accuracy'], reverse=True)
    
    print(f"\n🏆 词级准确率最高的5个文件:")
    for i, result in enumerate(results[:5], 1):
        print(f"  {i}. {result['filename']}: {result['word_accuracy']:.3f} ({result['word_accuracy']*100:.1f}%)")
    
    print(f"\n📉 词级准确率最低的5个文件:")
    for i, result in enumerate(results[-5:], 1):
        acc = result['word_accuracy']
        print(f"  {i}. {result['filename']}: {acc:.3f} ({acc*100:.1f}%)")
        if acc < 0.8:  # Show detailed info for accuracy below 80%
            print(f"     参考: {result['ref_text'][:60]}{'...' if len(result['ref_text']) > 60 else ''}")
            print(f"     识别: {result['hyp_text'][:60]}{'...' if len(result['hyp_text']) > 60 else ''}")
    
    # Accuracy distribution statistics
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
