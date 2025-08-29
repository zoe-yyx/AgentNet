import yaml
import logging
import os
import re
import time
import json
import pickle
from config.setting import DOUBAO_AK, DOUBAO_SK, DOUBAO_ENDPOINT_ID
from config.setting import BASIC_PROBLEM_COMPLEXITY
from config.setting import task_to_ability_map
import numpy as np
logger = logging.getLogger(__name__)
import openai
from openai import OpenAI

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



def get_deepseek_response(system_prompt, query_prompt):

    for attempt in range(MAX_RETRIES):
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL"))
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_prompt},
                ],
                temperature=0.0,
                max_tokens = 2048,
                top_p=1.0,
                stream=False
            )
            generated_text = response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES-1:
                raise
            time.sleep(1)  # Wait before retry

    return generated_text



def get_gpt_response(system_prompt, query_prompt):
    # Legacy OpenAI version
    for attempt in range(MAX_RETRIES):
        try:
            import openai  
            openai.api_key = os.getenv("OPENAI_API_KEY")
    
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
            time.sleep(1)  # Wait before retry

    return generated_text


import re
def get_qwen_response(system_prompt, query_prompt, model_name:str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 2048, top_p: float = 1.0):
    for attempt in range(MAX_RETRIES):
        try:

            from openai import OpenAI
            # Initialize OpenAI client
            client = OpenAI(
                # If environment variables are not configured, replace with Qianwen API Key: api_key="sk-xxx"
                api_key = os.getenv("QWEN_API_KEY"),
                base_url=os.getenv("QWEN_BASE_URL")
            )
  
            reasoning_content = ""  # Define complete reasoning process
            answer_content = ""     # Define complete response
            is_answering = False   # Determine if reasoning is finished and response started

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
                # If chunk.choices is empty, print usage
                if not chunk.choices:
                    print("\nUsage:")
                    print(chunk.usage)
                else:
                    delta = chunk.choices[0].delta
                    # Print reasoning process
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                        # print(delta.reasoning_content, end='', flush=True)
                        reasoning_content += delta.reasoning_content
                    else:
                        # Start response
                        if delta.content != "" and is_answering is False:
                            is_answering = True # Print response process
                            #print(delta.content, end='', flush=True)
                        answer_content += delta.content

            generated_text=answer_content.strip()

            break
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES-1:
                raise
            time.sleep(1)
    matches = re.findall(r'<answer>(.*?)</answer>', generated_text)
    last_answer = matches[-1] if matches else generated_text
    return last_answer


def parse_decision_text(decision_text):
    # Initialize dictionary
    output_dict = {}
    current_key = None
    current_value = []

    # Split string by line
    lines = decision_text.strip().split('\n')

    # Iterate through each line
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace

        # If line contains a word starting with a capital letter followed by a colon
        if re.match(r'^[A-Z_]+:', line):
            # If there are key-value pairs before, save
            if current_key is not None:
                output_dict[current_key] = ' '.join(current_value).strip()

            # Extract key-value pair: key is the word before the colon, value is the part after the colon
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Save current key and value
            output_dict[key] = value
            current_key = None  # Reset current key
            current_value = []  # Clear current value
        else:
            # If current line is not a key, but part of a value, add to current value
            if current_key is not None:
                current_value.append(line.strip())
            else:
                # If no current key, it means continuing to append to the previous key's value
                if current_value:
                    current_value.append(line.strip())

    # If there are remaining key-value pairs at the end, save
    if current_key is not None:
        output_dict[current_key] = ' '.join(current_value).strip()

    return output_dict




def parse_string_to_dict(text):
    # Use regex to find all uppercase words (possibly with underscores), excluding single letter words
    # matches = list(re.finditer(r"\b[A-Z_]{2,}\b", text))
    result = {}
    
    lines = text.strip().split("\n")

    for line in lines:
        match = re.match(r"\b[A-Z_]{2,}\b", line)
        if match:
            key = match.group(0)  # Extract the key
            value = line[match.end():].strip()  # Get the content after the colon
            value = re.sub(r"^[\s:]+|[\s]+$", "", value)
            result[key] = value

    return result


def extract_content_as_dict(long_string, short_strings):
    """
    Constructs a dictionary where short strings are keys and the content between these short strings are values.
    
    Args:
        long_string (str): The input long string.
        short_strings (list): A list containing multiple short strings.
        
    Returns:
        dict: A dictionary where short strings are keys, and the content following each short string is the value.
    """
    # Construct regex, join short strings with | to match all possible short strings
    pattern = "|".join(re.escape(short_string) for short_string in short_strings)
    regex = re.compile(f"({pattern})")

    # Use regex to find all starting positions of short strings
    matches = list(regex.finditer(long_string))
    result = {}

    for i, match in enumerate(matches):
        key = match.group()  # The current short string matched
        start = match.end()  # The end position of the current short string
        
        # Determine the end position of the value
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(long_string)  # The last short string goes to the end
        
        # Extract content and clean up spaces and colons
        value = long_string[start:end].strip().strip(":")
        result[key] = value

    return result






def extract_agent_id(text):
    """Extracts Agent ID from a string"""
    try:
        # First, try to extract content within brackets
        bracket_match = re.search(r'\[(.*?)\]', text)
        if bracket_match:
            # If the string is wrapped in brackets, extract the content
            text = bracket_match.group(1).strip()

        # Improved regex: match formats like 'Agent 4' or '4'
        agent_match = re.search(r'(Agent\s*(\d+)|(\d+))', text, re.IGNORECASE)
        if agent_match:
            # If a match is found, return the Agent ID
            agent_id = agent_match.group(2) if agent_match.group(2) else agent_match.group(3)
            return int(agent_id.strip())
        
        # If no match to "Agent X" format, try to extract pure numbers
        return int(text.strip())

    except (ValueError, TypeError) as e:
        print(f"Could not extract agent ID from: {text}")
        logger.info(f"Could not extract agent ID from: {text}")
        return None






def parse_split_pattern_to_subtask(split_text):
    """Parses the split pattern description"""
    
    if not split_text:
        # Error
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
        """Estimates subtask difficulty"""
        base_difficulty = len(description.split()) / 10
        if any(kw in description.lower() for kw in ['complex', 'difficult', 'advanced']):
            base_difficulty *= 1.5
        return min(5.0, base_difficulty)
    
def analyze_required_capabilities(description):
    """Analyzes required capabilities"""
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
    """Calculates task complexity"""
    
    # Basic complexity
    complexity = BASIC_PROBLEM_COMPLEXITY.get(task.task_type, 1.0)
    
    # Adjust based on prompt length
    length_factor = len(task.major_problem + task.context) / 100
    complexity *= (1 + 0.1 * length_factor)
    
    # Consider task priority
    complexity *= (1 + 0.2 * task.priority)
    
    return min(1.0, complexity / 5.0)



def extract_result(text):
    # Use regex to match "RESULT:" followed by the answer
    match = re.search(r'RESULT:\s*(\d+)', text)
    if match:
        return int(match.group(1))  # Return the matched number
    else:
        return None  # If no match found for the result 



def calculate_cos_similarity_A_and_Batch_B(A, B):
    dot_product = np.dot(A, B.T)
    # Calculate norms
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B, axis=1)  # (4,)
    # Calculate cosine similarity
    cosine_sim = dot_product / (norm_B * norm_A)
    return cosine_sim


def calculate_cos_similarity_A_and_B(A, B):
    dot_product = np.dot(B, A)
    # Calculate norms
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    # Calculate cosine similarity
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

