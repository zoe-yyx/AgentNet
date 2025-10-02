import sys
sys.dont_write_bytecode = True
import os

from config.setting import *

import argparse
import logging
import os
import colorlog
import json
from src.experiment import Experiment
from evaluator.datasets.bigbenchhard_dataset import BigBenchHardDataset
from src.utils import read_yaml
from prompt.bigbenchhard_prompt_set import BigBenchHardPromptSet
import os



def setup_logging(log_file_path):
    """Configure logging system"""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'
        },
    ))
    
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel("INFO")
    
    # Add file handler
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)




def get_args():
    parser = argparse.ArgumentParser(description="Run Agent DAG Task Processing System")
    parser.add_argument("--experiment_name", type=str, default="bigbenchhard", help="Name of our experiment")
    parser.add_argument("--global_router_experience", action="store_true")

    args = parser.parse_args()
    return args



def main():
    print("API Key configured via environment variable")
    args = get_args()

    import datetime
    date = datetime.datetime.now().strftime('%m%d-%H') #:%M
    log_file_path = os.path.join("./log", f"{args.experiment_name}_{date}.log" )
    setup_logging(log_file_path)
    logger = logging.getLogger(__name__)
    logger.info("Progress Start!")
    
    # Define paths
    json_file_path_result = f"./save/{args.experiment_name}_{date}/result.json"
    json_file_path_ability = f"./save/{args.experiment_name}_{date}/ability.json"
    json_file_path_edge_weight = f"./save/{args.experiment_name}_{date}/edge_weight.json"
    json_file_path_task_history = f"./save/{args.experiment_name}_{date}/task_history.json"
    json_file_path_experience = f"./save/{args.experiment_name}_{date}/experience.json"

    # Ensure directories for JSON files exist
    os.makedirs(os.path.dirname(json_file_path_result), exist_ok=True)
    os.makedirs(os.path.dirname(json_file_path_ability), exist_ok=True)
    os.makedirs(os.path.dirname(json_file_path_edge_weight), exist_ok=True)
    os.makedirs(os.path.dirname(json_file_path_task_history), exist_ok=True)
    os.makedirs(os.path.dirname(json_file_path_experience), exist_ok=True)

    with open(json_file_path_result, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    with open(json_file_path_ability, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    with open(json_file_path_edge_weight, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    with open(json_file_path_task_history, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    with open(json_file_path_experience, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)

    config_dir_path = "./config/experiment"
    dataset_root_path = "./big_datasets/bigbenchhard"
    
    total_experiment_config = read_yaml(os.path.join(config_dir_path, "bigbenchhard_new_abilities.yaml" ))
    
    experiment_config = total_experiment_config["experiment_config"]
    agent_config = total_experiment_config["agent_config"]
    agent_graph_config = total_experiment_config["agent_graph_config"]
    experiment_config["global_router_experience"] = args.global_router_experience
    agent_graph_config["global_router_experience"] = args.global_router_experience


    logger.info(f'default_agent_config is {total_experiment_config["default_agent_config"]}')
    logger.info(f'global_router_experience is {args.global_router_experience}')

    assert experiment_config["agent_num"] == len(agent_config), "Wrong With the Number of Agents in Initialization"
    dataset = BigBenchHardDataset()
    train_dataset = os.path.join(dataset_root_path, "bigbenchhard_train.jsonl")
    test_dataset = os.path.join(dataset_root_path, "bigbenchhard_test_same.jsonl")
    
    prompt_set = BigBenchHardPromptSet()
    constraints = prompt_set.get_constraint()
    thought_constraints = prompt_set.get_thought_constraint()
    # Initialize Experiment class:
    experiment = Experiment(experiment_config, 
                            agent_config, 
                            agent_graph_config, 
                            train_dataset, 
                            test_dataset, 
                            json_file_path_task_result=json_file_path_result, 
                            json_file_path_agent_info=json_file_path_ability, 
                            json_file_path_edge_weight=json_file_path_edge_weight,  
                            json_file_path_task_history=json_file_path_task_history,
                            json_file_path_experience=json_file_path_experience,
                            constraints = constraints, 
                            thought_constraints = thought_constraints)
    experiment.fit()
    experiment.evaluate()

    return 0


if __name__=="__main__":
    # try:
    main()
    # except Exception as e:
    #     logging.info(str(e))
    pass
