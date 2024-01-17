# llm_cr

## 项目目录说明
-- db   翻译库libretranslate运行时，对每个翻译任务产生的临时存储（包括会话）
-- translate_bleu_score.py    对不同翻译方式进行bleu评估
-- clean_cr_dataset.py      清洗CodeReview数据集
-- result_df.csv            BLEU评估结果


## 使用说明
1. libre翻译启动
```python
虚拟环境：py310
windows ffmpeg安装：winget install "FFmpeg (Essentials Build)"
库安装：pip install libretranslate
词典下载：命令行 libretranslate --load-only zh,en
web使用：命令行 libretranslate
```
2. BLEU评估数据集来源：https://www.heywhale.com/mw/dataset/60c41b7a19d601001898b34a/content