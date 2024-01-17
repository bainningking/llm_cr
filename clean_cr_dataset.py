import pandas as pd
from pathlib import Path
# 指定JSONL文件路径
jsonl_file_path = r"D:\常用文件夹\文档\WXWork\1688850221731084\Cache\File\2024-01\Comment_Generation\msg-valid.jsonl"

# 指定每个块的大小（行数）
chunk_size = 1000

# 逐块读取JSONL文件
chunks = pd.read_json(Path(jsonl_file_path), lines=True, chunksize=chunk_size)

# 循环处理每个块
for chunk_number, chunk_df in enumerate(chunks, start=1):
    # 在这里进行块级别的处理，chunk_df是一个DataFrame
    print(f"Processing Chunk {chunk_number}")
    print(chunk_df.head()) 
    # 清洗

    # 将DataFrame写回到新的JSONL文件
    chunk_df.to_json('cleaned_data.jsonl', orient='records', lines=True, mode='a')
