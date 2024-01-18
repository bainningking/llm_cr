import html
import json
import re
from pathlib import Path

import jieba
import pandas as pd
import requests
from nltk.translate.bleu_score import sentence_bleu
from retrying import retry
from tqdm import tqdm


def remove_chinese_punctuation(input_string):
    # 使用正则表达式匹配中文符号并替换为空格
    pattern = re.compile(
        "[\u3000\u3001-\u3011\u201c\u201d\u2018\u2019\uff01-\uff0f\uff1a-\uff1f\uff3b-\uff40\uff5b-\uff5e]+")
    result_string = re.sub(pattern, " ", input_string)
    return result_string


def split_list(input_list, chunk_size):
    """
    将列表切分为指定大小的子列表
    """
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]

# 对英文使用libretranslate进行翻译
def libre_translate_en(en_txt):
    url = "http://127.0.0.1:5000/translate"

    # 请求体
    payload = {
        "q": en_txt,
        "source": "en",
        "target": "zh",
        "format": "text",
        "api_key": ""
    }

    # 请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 发送POST请求
    response = requests.post(url, json=payload, headers=headers)

    # 返回译文结果
    return response.json()['translatedText']


# 对英文使用llama2进行翻译
@retry(wait_fixed=3000, stop_max_attempt_number=3)
def llama2_translate_en(en_txt):
    url = "https://www.llama2.ai/api"

    sys_prompt = "You're a professional translator who translates English into Chinese. Just reply to the translation and do not answer any other redundant statements.All I'm saying is what I need to translate:"
    # 请求体
    payload = {
        "prompt": "<s>[INST] <<SYS>>\n"+sys_prompt+"\n<</SYS>>\n\n\""+en_txt+"\" [/INST]\n",
        "model": "meta/llama-2-70b-chat",
        "systemPrompt": sys_prompt,
        "temperature": 0.75,
        "topP": 0.9,
        "maxTokens": 800,
        "image": None,
        "audio": None
    }

    # 请求头
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # 发送POST请求
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status code: {}".format(response.status_code))

    # 返回译文结果
    return response.text


def has_chinese(sentence):
    # 使用正则表达式匹配中文字符
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(sentence))

# 对英文使用llama2中文--Atom-7B进行翻译
@retry(wait_fixed=3000, stop_max_attempt_number=3)
def atom_translate_en(en_txt):
    url = "https://llama.family/serverApi/llama/chat/ask-question"

    sys_prompt = "You're a professional translator who translates English into Chinese. Just reply to the translation and do not answer any other redundant statements.The curly brackets surround is what I need to translate, please translate the following english sentence to chinese: "
    # 请求体
    payload = {
        "question": sys_prompt+'{'+en_txt.replace('\"', '')+'}',
        "questionId": 0,
        "modelCode": "bdbefed4-03a8-4e32-b650-974246184783",
        "contextQIds": []
    }

    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Cookie": "ajs_anonymous_id=b0992128-efec-41da-8253-9dc16e024ad0; atomecho_session=2a45d807-af0b-4864-b4fb-b70d2e225509",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # 发送POST请求
    response = requests.post(url, json=payload, headers=headers, stream=True)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status code: {}".format(response.status_code))

    # 返回译文结果
    last_response = json.loads(list(filter(None, response.content.decode(
        'utf-8').split('\n\n')))[-1].replace("id:\nevent:message\ndata:", ""))
    if last_response['status'] == 'FINISH':
        trans_res = json.loads(last_response['answer'])['answer_text']
        if has_chinese(trans_res):
            return trans_res
        else:
            print(f'输出的翻译不含有中文：{trans_res}')
    else:
        print(f'翻译接口出问题了，last_response是{last_response}')


# 按行获取文本
def get_txt_origin(file_path: str):
    # 打开文件并按行读取
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = []
        if file_path.suffix.lower() == '.en':
            for line in file:
                # 对每一行英文进行html转义字符处理
                lines.append(html.unescape(line.strip().replace('@-@', '')))

        elif file_path.suffix.lower() == '.zh':
            for line in file:
                lines.append(line.strip())
        else:
            print('请输入正确的文件名')
    return lines

# 评估译文与原文的bleu分数
def eval_bleu_score(zh_origin: list, trans_zh: list):
    bleu_score_list = []

    for ori, trans in zip(zh_origin, trans_zh):
        # 将中文参考答案进行分词
        reference_tokenized = re.split(r'\s+', remove_chinese_punctuation(ori))

        # 将机器翻译结果进行分词
        candidate_tokenized = list(
            filter(lambda x: x.strip() != "", jieba.cut(remove_chinese_punctuation(trans))))

        # 计算1-gram和2-gram的BLEU分数
        bleu_score = sentence_bleu(
            [reference_tokenized], candidate_tokenized, weights=(0.5, 0.5, 0, 0))
        bleu_score_list.append(bleu_score)

    return bleu_score_list


if __name__ == "__main__":
    # 读取英文及其翻译
    en_origin_path = r"D:\常用文件夹\下载\Compressed\中英翻译数据集\test\newstest2017.tc.en"
    zh_origin_path = r"D:\常用文件夹\下载\Compressed\中英翻译数据集\test\newstest2017.tc.zh"
    en_origin = get_txt_origin(Path(en_origin_path))
    zh_origin = get_txt_origin(Path(zh_origin_path))

    # 对英文进行批量翻译
    en_origin = en_origin[0:500]

    zh_trans_libre = []
    zh_trans_llama2 = []
    zh_trans_atom = []

    for sub_en_origin in tqdm(en_origin, desc="翻译进度", unit="条"):
        zh_trans_libre.append(libre_translate_en(sub_en_origin))
        zh_trans_llama2.append(llama2_translate_en(sub_en_origin))
        zh_trans_atom.append(atom_translate_en(sub_en_origin))

    # bleu评估分数
    bleu_score_libre = eval_bleu_score(zh_origin, zh_trans_libre)
    bleu_score_llama2 = eval_bleu_score(zh_origin, zh_trans_llama2)
    bleu_score_atom = eval_bleu_score(zh_origin, zh_trans_atom)

    result_df = pd.DataFrame(columns=['英文原文', '中文原文', 'libre译文', 'llama2译文', 'atom-7B译文', 'bleu_score_libre', 'bleu_score_llama2', 'bleu_score_atom-7B'],
                             data=list(zip(en_origin, zh_origin, zh_trans_libre, zh_trans_llama2, zh_trans_atom, bleu_score_libre, bleu_score_llama2, bleu_score_atom)))
    result_df.to_csv('result_df.csv', index=False)

    # atom_df = pd.DataFrame(columns=['atom-7B译文', 'bleu_score_atom-7B'], data=list(zip(zh_trans_atom, bleu_score_atom)))
    # atom_df.to_csv('atom_df2.csv', index=False)