#!/usr/bin/env python3
"""
修正版文件比较脚本 - 处理文件名差异
"""

import re
import sys
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
    
    # 移除多余空格并转为小写
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    return text

def normalize_filename(filename):
    """标准化文件名，移除_vocals后缀以便匹配"""
    # 移除_vocals后缀
    filename = filename.replace('_vocals.wav', '.wav')
    return filename

def parse_file(file_path):
    """解析文件，返回标准化文件名和清理后的文本对"""
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
                parts = line.split(separator, 1)  # 只分割第一个分隔符
                if len(parts) == 2:
                    filename = parts[0].strip()
                    content = parts[1].strip()
                    
                    # 标准化文件名
                    normalized_filename = normalize_filename(filename)
                    
                    # 清理内容
                    cleaned_content = clean_text(content)
                    data[normalized_filename] = {
                        'original_filename': filename,
                        'original_content': content,
                        'cleaned_content': cleaned_content,
                        'line_num': line_num
                    }
                else:
                    print(f"⚠️  警告: 第{line_num}行格式不正确: {line[:50]}...")
    
    except Exception as e:
        print(f"❌ 读取文件失败 {file_path}: {e}")
        return {}
    
    return data

def calculate_similarity(text1, text2):
    """计算两个文本的相似度"""
    return SequenceMatcher(None, text1, text2).ratio()

def main():
    file1_path = "/opt/data/Malaysian_ASR/test_sample_transcribe_result.txt"
    file2_path = "/opt/data/Malaysian_ASR/vocals_only_mono_transcripts_formatted.txt"
    
    print("📊 转录文件内容比较工具（修正版）")
    print("=" * 60)
    print(f"文件1: {file1_path}")
    print(f"文件2: {file2_path}")
    print(f"比较规则: 忽略标点符号和括号内容，标准化文件名")
    print("=" * 60)
    
    # 解析两个文件
    print("📖 正在解析文件...")
    data1 = parse_file(file1_path)
    data2 = parse_file(file2_path)
    
    print(f"✅ 文件1解析完成: {len(data1)} 条记录")
    print(f"✅ 文件2解析完成: {len(data2)} 条记录")
    
    # 获取所有文件名
    all_filenames = set(data1.keys()) | set(data2.keys())
    only_in_file1 = set(data1.keys()) - set(data2.keys())
    only_in_file2 = set(data2.keys()) - set(data1.keys())
    common_files = set(data1.keys()) & set(data2.keys())
    
    print(f"\n📈 统计信息:")
    print(f"   总文件数: {len(all_filenames)}")
    print(f"   共同文件: {len(common_files)}")
    print(f"   仅在文件1: {len(only_in_file1)}")
    print(f"   仅在文件2: {len(only_in_file2)}")
    
    # 显示仅在某个文件中的条目
    if only_in_file1:
        print(f"\n📋 仅在文件1中的条目:")
        for filename in sorted(list(only_in_file1)[:5]):  # 最多显示5个
            print(f"   - {filename}")
        if len(only_in_file1) > 5:
            print(f"   ... 还有 {len(only_in_file1) - 5} 个")
    
    if only_in_file2:
        print(f"\n📋 仅在文件2中的条目:")
        for filename in sorted(list(only_in_file2)[:5]):  # 最多显示5个
            print(f"   - {filename}")
        if len(only_in_file2) > 5:
            print(f"   ... 还有 {len(only_in_file2) - 5} 个")
    
    # 比较共同文件的内容
    if common_files:
        print(f"\n🔍 内容差异分析:")
        print("-" * 60)
        
        identical_count = 0
        different_count = 0
        differences = []
        
        for filename in sorted(common_files):
            text1 = data1[filename]['cleaned_content']
            text2 = data2[filename]['cleaned_content']
            
            similarity = calculate_similarity(text1, text2)
            
            if text1 == text2:
                identical_count += 1
            else:
                different_count += 1
                differences.append({
                    'filename': filename,
                    'similarity': similarity,
                    'text1': text1,
                    'text2': text2,
                    'original1': data1[filename]['original_content'],
                    'original2': data2[filename]['original_content']
                })
        
        print(f"✅ 完全相同: {identical_count} 个文件")
        print(f"❗ 有差异: {different_count} 个文件")
        
        if differences:
            print(f"\n📋 差异详情 (显示前10个):")
            
            # 按相似度排序，最不相似的在前
            differences.sort(key=lambda x: x['similarity'])
            
            for i, diff in enumerate(differences[:10], 1):
                print(f"\n{i}. 文件: {diff['filename']}")
                print(f"   相似度: {diff['similarity']:.3f}")
                print(f"   文件1 (清理后): {diff['text1'][:80]}{'...' if len(diff['text1']) > 80 else ''}")
                print(f"   文件2 (清理后): {diff['text2'][:80]}{'...' if len(diff['text2']) > 80 else ''}")
                
                # 显示原始内容的差异（仅对相似度很低的）
                if diff['similarity'] < 0.7:
                    print(f"   ---")
                    print(f"   文件1 (原始): {diff['original1'][:80]}{'...' if len(diff['original1']) > 80 else ''}")
                    print(f"   文件2 (原始): {diff['original2'][:80]}{'...' if len(diff['original2']) > 80 else ''}")
            
            if len(differences) > 10:
                print(f"\n   ... 还有 {len(differences) - 10} 个差异文件")
        
        # 统计相似度分布
        if differences:
            print(f"\n📊 相似度分布:")
            high_similarity = sum(1 for d in differences if d['similarity'] >= 0.9)
            medium_similarity = sum(1 for d in differences if 0.7 <= d['similarity'] < 0.9)
            low_similarity = sum(1 for d in differences if d['similarity'] < 0.7)
            
            print(f"   高相似度 (≥90%): {high_similarity} 个")
            print(f"   中等相似度 (70-90%): {medium_similarity} 个")
            print(f"   低相似度 (<70%): {low_similarity} 个")
            
            # 计算总体匹配率
            if common_files:
                match_rate = identical_count / len(common_files) * 100
                avg_similarity = sum(d['similarity'] for d in differences) / len(differences) if differences else 1.0
                print(f"\n📈 总体匹配率: {match_rate:.1f}% (完全相同)")
                print(f"📈 平均相似度: {avg_similarity:.3f}")
    
    print(f"\n" + "=" * 60)
    print("📋 比较结果汇总:")
    print(f"   总文件数: {len(all_filenames)}")
    print(f"   共同文件: {len(common_files)}")
    if common_files:
        print(f"   完全相同: {identical_count}")
        print(f"   有差异: {different_count}")
    print(f"   仅在文件1: {len(only_in_file1)}")
    print(f"   仅在文件2: {len(only_in_file2)}")

if __name__ == "__main__":
    main()
