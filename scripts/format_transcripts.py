#!/usr/bin/env python3
"""
格式化转录结果文件，将其转换为每行一个条目的格式
"""

import sys

def format_transcript_file(input_file, output_file):
    """格式化转录文件为每行一个条目"""
    
    try:
        # 读取原文件
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 处理转义的换行符
        content = content.replace('\\n', '\n')
        
        # 按换行符分割
        lines = content.strip().split('\n')
        
        # 过滤空行
        lines = [line.strip() for line in lines if line.strip()]
        
        print(f"📄 原文件: {input_file}")
        print(f"📝 输出文件: {output_file}")
        print(f"📊 总条目数: {len(lines)}")
        
        # 写入格式化后的文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, line in enumerate(lines, 1):
                f.write(line + '\n')
                
                # 显示前几行预览
                if i <= 5:
                    # 限制显示长度
                    preview = line[:100] + '...' if len(line) > 100 else line
                    print(f"  {i:2d}. {preview}")
        
        print(f"✅ 格式化完成！共处理 {len(lines)} 条记录")
        print(f"💾 结果已保存到: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 格式化失败: {e}")
        return False

def main():
    input_file = "/opt/data/Malaysian_ASR/vocals_only_mono_transcripts.txt"
    output_file = "/opt/data/Malaysian_ASR/vocals_only_mono_transcripts_formatted.txt"
    
    print("📝 转录文件格式化工具")
    print("=" * 50)
    
    # 执行格式化
    success = format_transcript_file(input_file, output_file)
    
    if success:
        print(f"\n🎉 格式化成功完成！")
        print(f"原文件: {input_file}")
        print(f"新文件: {output_file}")
        print(f"\n格式: 每行一个转录结果")
        print(f"示例格式: 音频文件名|转录内容")
    else:
        print("❌ 格式化失败")

if __name__ == "__main__":
    main()
