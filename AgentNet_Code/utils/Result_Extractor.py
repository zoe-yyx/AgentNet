import re

def extract_all_results(text):
    # Use regex to match all lines starting with "Result:" or "RESULT:", handle newlines
    result_pattern = re.compile(r"(?:Result:|RESULT:)\s*(.*?)(?=\n|$)", re.DOTALL)
    results = re.findall(result_pattern, text)
    return results

def extract_answer(text):
    # Extract all answers after "Result"
    results = extract_all_results(text)
    
    # Return the last answer (assuming the last one is the correct answer)
    if results:
        last_result = results[-1].strip()
        # Check if the last result is not empty
        if last_result:
            return last_result
    return text
