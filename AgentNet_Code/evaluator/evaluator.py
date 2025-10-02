import re

def match_option(input_answer, model_answer):
    '''eliminate the parentheses and compare the answer'''
    cleaned_input = input_answer.strip().lower().replace('(','').replace(')','')
    cleaned_model = model_answer.strip().lower().replace('(', '').replace(')', '')
    
    print(f"compare {cleaned_input} and {cleaned_model}")
    
    return cleaned_input == cleaned_model


def match_sorted_words(input_answer, model_answer):
    '''eliminate the (comma and space) and compare the answer'''
    cleaned_input = input_answer.strip().lower().replace(',', '').replace(' ', '')
    cleaned_model = model_answer.strip().lower().replace(',', '').replace(' ', '')
    
    print(f"compare {cleaned_input} and {cleaned_model}")
    
    return cleaned_input == cleaned_model


def match_yes_no(input_answer, model_answer):
    '''compare the answer'''
    cleaned_input = input_answer.strip().lower()
    cleaned_model = model_answer.strip().lower()
    
    print(f"compare {cleaned_input} and {cleaned_model}")
    
    return cleaned_input == cleaned_model

def match_dyck_language(input_answer, model_answer):
    '''detect the brackets sequence'''
    cleaned_input = input_answer.strip().lower().replace(' ', '')
    cleaned_model = model_answer.strip().lower().replace(' ', '')   
    
    print(f"compare {cleaned_input} and {cleaned_model}")
    
    return cleaned_input == cleaned_model




def evaluate_answer_old(ground_truth_answer, predicted_answer):
    '''auto determine the type of the answer and compare the answer'''
    ground_truth_answer = ground_truth_answer.strip()
    predicted_answer = predicted_answer.strip()
    
    if ground_truth_answer.lower() in ['yes', 'no','true','false','valid','invalid'] and predicted_answer.lower() in ['yes', 'no','true','false','valid','invalid']:
        print("detect Yes/No type match")
        return match_yes_no(ground_truth_answer, predicted_answer)
    
    # 去掉括号并转为小写后判断是否为单个英文字母
    cleaned_input = ground_truth_answer.strip().lower().replace('(','').replace(')','')
    cleaned_model = predicted_answer.strip().lower().replace('(','').replace(')','')
    
    if len(cleaned_input) == 1 and cleaned_input.isalpha() and len(cleaned_model) == 1 and cleaned_model.isalpha():
        print("detect single letter option type match")   
        return match_option(ground_truth_answer, predicted_answer)
    
    # 判断是否为纯英文字母（去掉逗号和空格后）
    cleaned_input_alpha = ground_truth_answer.replace(',', '').replace(' ', '')
    cleaned_model_alpha = predicted_answer.replace(',', '').replace(' ', '')
    print(cleaned_input_alpha, cleaned_model_alpha)
    if cleaned_input_alpha.isalpha() and cleaned_model_alpha.isalpha():
        print("detect pure letters type match")
        return match_sorted_words(ground_truth_answer, predicted_answer)
    
    # 判断是否为括号序列
    cleaned_input_brackets = ground_truth_answer.replace(' ', '')
    cleaned_model_brackets = predicted_answer.replace(' ', '')
    
    if all(c in '()[]{}' for c in cleaned_input_brackets) and all(c in '()[]{}' for c in cleaned_model_brackets):
        print("detect brackets sequence type match")
        return match_dyck_language(ground_truth_answer, predicted_answer)
    
    # 如果没有匹配到特殊类型，返回False
    print("cannot match the answer type, compare the answer directly")
    return ground_truth_answer == predicted_answer


# New
def contains_lowercase_in_parentheses(s):
    if re.search(r'\([a-z]\)', s):
        return True
    return False

def normalize_option_answer(answer):
    answer = re.sub(r'[()]', '', answer)
    return answer.strip().lower()

def clean_and_extract(predicted_answer):
    predicted_answer = predicted_answer.replace('.', ' ').replace(',', ' ')
    match = re.match(r'\(([a-z])\)', predicted_answer.strip())
    if match:
        return match.group(1)
    return predicted_answer.strip().replace(" ", "").lower()


def evaluate_answer(ground_truth_answer, predicted_answer):
    ground_truth_answer = ground_truth_answer.strip().lower()
    predicted_answer = predicted_answer.strip().lower()

    # Yes/No Binary Type
    if ground_truth_answer in ['yes', 'no', 'valid', 'invalid', 'true', 'false']:
        return ground_truth_answer == predicted_answer
    
    # Option (A)/(B)... Type
    elif contains_lowercase_in_parentheses(ground_truth_answer):
        predicted_answer = clean_and_extract(predicted_answer)
        return normalize_option_answer(ground_truth_answer) == normalize_option_answer(predicted_answer)
    
    return ground_truth_answer == predicted_answer

def test_function():
    return 
