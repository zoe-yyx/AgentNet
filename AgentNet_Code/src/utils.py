import yaml
import logging
import os
import re
import time
import json
import pickle
# from .setting import HTTP_PROXY, HTTPS_PROXY, OPENAI_API_KEY, DOUBAO_AK, DOUBAO_SK, DOUBAO_ENDPOINT_ID
# from .setting import HTTP_PROXY, HTTPS_PROXY, OPENAI_API_KEY, DOUBAO_AK, DOUBAO_SK, DOUBAO_ENDPOINT_ID
from config.setting import HTTP_PROXY, HTTPS_PROXY, OPENAI_API_KEY, DOUBAO_AK, DOUBAO_SK, DOUBAO_ENDPOINT_ID
from config.setting import BASIC_PROBLEM_COMPLEXITY
from config.setting import task_to_ability_map
import numpy as np
logger = logging.getLogger(__name__)


MAX_RETRIES = 3



def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as w:
        json.dump(data, w, indent=4)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_yaml(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config




def get_doubao_response(system_prompt, query_prompt):
    from volcenginesdkarkruntime import Ark
    client = Ark(ak=DOUBAO_AK, sk=DOUBAO_SK)
    completion = client.chat.completions.create(
        model=DOUBAO_ENDPOINT_ID,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_prompt},
        ],
    )
    generated_text = completion.choices[0].message.content
    return generated_text



def get_gpt_response(system_prompt, query_prompt):
    for attempt in range(MAX_RETRIES):
        try:
            os.environ['HTTP_PROXY'] = HTTP_PROXY
            os.environ['HTTPS_PROXY'] = HTTPS_PROXY
            import openai  
            openai.api_key = OPENAI_API_KEY

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_prompt}
                ],
                temperature=0.0,
                max_tokens = 2048,
                top_p=1.0,
            )
            generated_text = response.choices[0].message['content'].strip()

            break
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES-1:
                raise
            time.sleep(1)  # 重试前等待

    return generated_text


def get_qwen_response(system_prompt, query_prompt, model_name:str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 2048, top_p: float = 1.0):
    for attempt in range(MAX_RETRIES):
        try:
            os.environ['HTTP_PROXY'] = HTTP_PROXY
            os.environ['HTTPS_PROXY'] = HTTPS_PROXY

            from openai import OpenAI
            # 初始化OpenAI客户端
            client = OpenAI(
                # 如果没有配置环境变量，请用百炼API Key替换：api_key="sk-xxx"
                api_key = "sk-af2e49cd852f4b1e9be41859a7680ffb",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
  
            reasoning_content = ""  # 定义完整思考过程
            answer_content = ""     # 定义完整回复
            is_answering = False   # 判断是否结束思考过程并开始回复 # 创建聊天完成请求

            response = client.chat.completions.create(
                model="qwen-turbo", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_prompt}
                ],
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
           
            for chunk in response:
                # 如果chunk.choices为空，则打印usage
                if not chunk.choices:
                    print("\nUsage:")
                    print(chunk.usage)
                else:
                    delta = chunk.choices[0].delta
                    # 打印思考过程
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                        # print(delta.reasoning_content, end='', flush=True)
                        reasoning_content += delta.reasoning_content
                    else:
                        # 开始回复
                        if delta.content != "" and is_answering is False:
                            #print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                            is_answering = True # 打印回复过程
                            #print(delta.content, end='', flush=True)
                        answer_content += delta.content

            generated_text=answer_content.strip()

            break
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES-1:
                raise
            time.sleep(1)

    return generated_text


import re

def parse_decision_text(decision_text):
    # 初始化字典
    output_dict = {}
    current_key = None
    current_value = []

    # 按行分割字符串
    lines = decision_text.strip().split('\n')

    # 遍历每一行
    for line in lines:
        line = line.strip()  # 去除行首行尾空格

        # 如果行包含一个大写字母的单词并且后面跟着冒号
        if re.match(r'^[A-Z_]+:', line):
            # 如果之前有键值对，保存
            if current_key is not None:
                output_dict[current_key] = ' '.join(current_value).strip()

            # 提取键值对：键是冒号前的大写字母单词，值是冒号后面的部分
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # 保存当前的键和值
            output_dict[key] = value
            current_key = None  # 重置当前键
            current_value = []  # 清空当前值
        else:
            # 如果当前行不是键，而是值的一部分，加入到当前值中
            if current_key is not None:
                current_value.append(line.strip())
            else:
                # 如果没有当前键，认为是继续拼接前一个键的值
                if current_value:
                    current_value.append(line.strip())

    # 如果最后还有剩余的键值对，保存
    if current_key is not None:
        output_dict[current_key] = ' '.join(current_value).strip()

    return output_dict




def parse_string_to_dict(text):
    # 使用正则表达式查找全大写字母的单词（可能包含下划线），排除单个字母的单词
    # matches = list(re.finditer(r"\b[A-Z_]{2,}\b", text))
    result = {}
    
    lines = text.strip().split("\n")

    for line in lines:
        match = re.match(r"\b[A-Z_]{2,}\b", line)
        if match:
            key = match.group(0)  # 提取匹配的键
            value = line[match.end():].strip()  # 获取冒号后面的内容
            value = re.sub(r"^[\s:]+|[\s]+$", "", value)
            result[key] = value

    # for i, match in enumerate(matches):
    #     # 获取当前键以及下一个键之间的内容
    #     key = match.group()
    #     start = match.end()
        
    #     # 确定内容的结束位置
    #     if i + 1 < len(matches):
    #         end = matches[i + 1].start()
    #         value = text[start:end].strip()
    #     else:
    #         value = text[start:].strip()
        
        # # 去除值的首尾冒号和空白字符（包括换行符）
        # value = re.sub(r"^[\s:]+|[\s]+$", "", value)
        
        # # 将结果添加到字典
        # result[key] = value

    return result


def extract_content_as_dict(long_string, short_strings):
    """
    将短字符串作为键，长字符串中这些短字符串之间的内容作为值，构造成字典。
    
    Args:
        long_string (str): 输入的长字符串。
        short_strings (list): 包含多个短字符串的列表。
        
    Returns:
        dict: 以短字符串为键的字典，每个键对应长字符串中位于该短字符串之后的内容作为值。
    """
    # 构造正则表达式，将短字符串用 | 连接，匹配所有可能的短字符串
    pattern = "|".join(re.escape(short_string) for short_string in short_strings)
    regex = re.compile(f"({pattern})")

    # 使用正则匹配短字符串，找到所有匹配的起始位置
    matches = list(regex.finditer(long_string))
    result = {}

    for i, match in enumerate(matches):
        key = match.group()  # 当前匹配的短字符串
        start = match.end()  # 当前短字符串结束的位置
        
        # 确定值的结束位置
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(long_string)  # 最后一个短字符串取到末尾
        
        # 提取内容并清理空格和冒号
        value = long_string[start:end].strip().strip(":")
        result[key] = value

    return result






def extract_agent_id(text):
    """从字符串中提取Agent ID"""
    try:
        # 首先尝试提取方括号中的内容
        bracket_match = re.search(r'\[(.*?)\]', text)
        if bracket_match:
            # 如果字符串用方括号包裹，则提取其中内容
            text = bracket_match.group(1).strip()

        # 改进的正则表达式：匹配类似 'Agent 4' 或 '4' 的格式
        agent_match = re.search(r'(Agent\s*(\d+)|(\d+))', text, re.IGNORECASE)
        if agent_match:
            # 如果找到匹配项，返回Agent ID
            agent_id = agent_match.group(2) if agent_match.group(2) else agent_match.group(3)
            return int(agent_id.strip())
        
        # 如果没有匹配到“Agent X”的格式，尝试提取纯数字
        return int(text.strip())

    except (ValueError, TypeError) as e:
        print(f"Could not extract agent ID from: {text}")
        logger.info(f"Could not extract agent ID from: {text}")
        return None






def parse_split_pattern_to_subtask(split_text):
    """解析拆分模式描述"""
    
    if not split_text:
        # 报错
        return None
        
    try:
        
        re_pattern = r'(\b(?:\d+\.\s+|Subtask\s+\d+)\s*[^;]+(?:;|$))'
        tasks = re.findall(re_pattern, split_text)
        tasks = [task.strip() for task in tasks]
        difficulty = [estimate_subtask_difficulty(task) for task in tasks]
        required_capabilities = [analyze_required_capabilities(task) for task in tasks]


        return tasks, difficulty, required_capabilities
    
    except Exception as e:
        print(f"Error when parsing text: {split_text}")
        print(f"Error parsing split pattern: {e}")
        return None



def estimate_subtask_difficulty(description):
        """估计子任务难度"""
        base_difficulty = len(description.split()) / 10
        if any(kw in description.lower() for kw in ['complex', 'difficult', 'advanced']):
            base_difficulty *= 1.5
        return min(5.0, base_difficulty)
    
def analyze_required_capabilities(description):
    """分析所需能力"""
    capabilities = []
    keywords = {
        'math': ['calculate', 'solve', 'equation'],
        'coding': ['function', 'code', 'implement'],
        'language': ['translate', 'write', 'summarize']
    }
    
    description = description.lower()
    for capability, words in keywords.items():
        if any(word in description for word in words):
            capabilities.append(capability)
            
    return capabilities
    



def calculate_task_complexity(task):
    """计算任务复杂度"""
    
    # 基础复杂度
    complexity = BASIC_PROBLEM_COMPLEXITY.get(task.task_type, 1.0)
    
    # 根据提示长度调整
    length_factor = len(task.major_problem + task.context) / 100
    complexity *= (1 + 0.1 * length_factor)
    
    # 考虑任务优先级
    complexity *= (1 + 0.2 * task.priority)
    
    return min(1.0, complexity / 5.0)



def extract_result(text):
    # 使用正则表达式匹配 "RESULT:" 后面的答案
    match = re.search(r'RESULT:\s*(\d+)', text)
    if match:
        return int(match.group(1))  # 返回匹配到的数字
    else:
        return None  # 如果没有找到匹配的结果 



def calculate_cos_similarity_A_and_Batch_B(A, B):
    dot_product = np.dot(A, B.T)
    # 计算模长
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B, axis=1)  # (4,)
    # 计算余弦相似度
    cosine_sim = dot_product / (norm_B * norm_A)
    return cosine_sim


def calculate_cos_similarity_A_and_B(A, B):
    dot_product = np.dot(B, A)
    # 计算模长
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    # 计算余弦相似度
    cosine_sim = dot_product / (norm_B * norm_A)
    return cosine_sim




def get_average_abilities_from_task_type(task_type, abilities_dict):
    ability = 0
    count = 0
    for ability_type in task_to_ability_map[task_type]:
        count += 1
        ability += abilities_dict.get(ability_type, 0)
    if count != 0:
        ability /= count
        return ability
    else:
        return 0

