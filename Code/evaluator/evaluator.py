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



"""
Run solutions from one problem in APPS.
"""
import sys
sys.path.append('/hdd/yxyang/AgentNet-Experiments_APPS/evaluator')  # Adjust the path

import argparse
import json
import numpy as np
import os
import pprint
import multiprocessing
import testing_util as test_util
import subprocess
# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from types import SimpleNamespace
from typing import Dict


EXAMPLE_RESULTS = {"0": [[-2]],"1": [[False,False,False]],"2": [[True,True]],"3": [[False,True,False,True,False,False,False,True,False,True,False,True,True,True,False,True]],"4": [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]}
EXAMPLE_ARGS = SimpleNamespace(debug=True)
TIMEOUT = 10


def check_correctness(prob_path, generation, timeout):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(prob_path, generation, result):
        result.append(test_util.run_test(prob_path=prob_path, test=generation))  # Test instance + generated code
    # print("prob_path",prob_path,"generation",generation)
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(prob_path, generation, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        # Reamark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead 
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]
    return result[0]




def evaluate_answer(test_cases, output_str):
    # output_str is generated code, test_case is test input/output
    res = []

    curr_res = [-2]
    try:
        curr_res = check_correctness(prob_path=test_cases, generation=output_str, timeout=10)  
        fixed = []
        for e in curr_res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        curr_res = fixed
        if not np.all(curr_res):
            print(f"Results were not all True: {curr_res}")
    except Exception as e:
        print(f"test framework exception = {repr(e)}{e}\n")
    finally:
        assert isinstance(curr_res, list)
        res.append(curr_res)
        print(f"results = {res}")

    return res




def test_function():
    return 
