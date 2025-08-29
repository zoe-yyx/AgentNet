import re

# def extract_all_results(text):
#     # 使用正则表达式匹配所有以 "Result:" 或 "RESULT:" 开头的行，处理换行符
#     result_pattern = re.compile(r"(?:Result:|RESULT:)\s*(.*?)(?=\n|$)", re.DOTALL)
#     results = re.findall(result_pattern, text)
#     return results

# def extract_answer(text):
#     # 提取所有的Result后面的答案
#     results = extract_all_results(text)
    
#     # # 找到最后一个答案并返回（假设最后一个答案是正确的）
#     # if results:
#     #     last_result = results[-1].strip()
#     #     # 判断最后一个结果是否为空
#     #     if last_result:
#     #         return last_result
#     return results


# def extract_answer(text):
#     """
#     Extracts the code block after 'RESULT:' in a given text.

#     Parameters:
#     text (str): The input text containing 'RESULT:' followed by a code block.

#     Returns:
#     str: Extracted code or an empty string if not found.
#     """
#     match = re.search(r"RESULT:\s*```python\n(.*?)```", text, re.DOTALL)
#     return match.group(1) if match else ""


def extract_answer(text):
    """
    Extracts the Python code block from a given text.

    Parameters:
    text (str): The input text containing a Python code block.

    Returns:
    str: Extracted code or an empty string if not found.
    """
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else ""