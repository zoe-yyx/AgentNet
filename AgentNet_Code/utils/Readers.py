import pandas as pd 
import json


def get_agent_ability_trend(agent_id,file_path='save_read/save/bigbenchhard_ability.json'):
    """
    get the ability trend of the agent  
    
    parameters:
        agent_id: int, agent id 
        file_path: str, the path of the ability json file
        
    return:
        pandas DataFrame, the ability trend of the agent
    """
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    abilities = []
    for datapoint in data:
        if str(agent_id) in datapoint:
            agent_data = datapoint[str(agent_id)]['abilities']
            abilities.append(agent_data)
    
    df = pd.DataFrame(abilities)
    df.index.name = 'task_chain_id'
    return df


def get_accuracy(file_path='save/bigbenchhard_result.json'):
    """
    calculate the accuracy of the experiment

    parameters:
        file_path: str, the path of the result json file
        
    return:
        float, the accuracy of the experiment
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    success_count = sum(item['success'] for item in data)
    accuracy = success_count / len(data)
    return accuracy


def get_accuracy_trend(file_path='save/bigbenchhard_result.json'):
    """
    get the accuracy trend of the experiment
    
    parameters:
        file_path: str, the path of the result json file
        
    return:
        list, the accuracy trend of the experiment
    """
    accuracies = []
    with open(file_path, 'r') as f:
        data = json.load(f)

    for i in range(len(data)):
        accuracy = sum(item['success'] for item in data[:i+1]) / (i+1)
        accuracies.append(accuracy)
    return accuracies