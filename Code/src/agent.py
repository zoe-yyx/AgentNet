'''
This file contains the Agent class.
The Agent class is our core component, which is an LLM-based Agent. It consists of two main parts: Router and Executor, 
responsible for deciding how to handle tasks and executing tasks respectively.
We don't call other Agents through Agent, but control and call through the Experiment class. 
For specific LLM calls, we implement them in utils.
'''


import random
import time
import numpy as np
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Union
import copy 
import json

logger = logging.getLogger(__name__)


from .pool import RouterExperiencePool, ExecutorExperiencePool
from .pool import RouterExperience, ExecutorExperience
from .utils import get_doubao_response, get_gpt_response, get_qwen_response, get_deepseek_response
from .utils import parse_decision_text, extract_agent_id, parse_split_pattern_to_subtask, parse_string_to_dict, extract_content_as_dict
from .utils import calculate_task_complexity
from .utils import get_average_abilities_from_task_type


from config.setting import ROUTER_SYSTEM_PROMPT, ROUTER_DECIDE_NEXT_AGENT_ID_SYSTEM_PROMPT, EXECUTOR_SYSTEM_PROMPT, THOUGHT_SYSTEM_PROMPT
from config.setting import ROUTER_PROMPT_FORMAT, ROUTER_PROMPT_SPLIT_AND_EXECUTE_FORMAT, ROUTER_PROMPT_DECIDE_NEXT_AGENT_ID_FORMAT
from config.setting import EXECUTOR_PROMPT_FORMAT, THOUGHT_PROMPT_FORMAT, EXECUTOR_THOUGHT_PROMPT_FORMAT, SPLIT_EXECUTOR_PROMPT_FORMAT, SPLIT_THOUGHT_PROMPT_FORMAT
from config.setting import ROUTER_CLASSIFY_SYSTEM_PROMPT, ROUTER_CLASSIFY_PROMPT
from config.setting import BASIC_PROBLEM_COMPLEXITY, task_to_ability_map


llm_model_response_map = {
    "doubao": get_doubao_response,
    "gpt-4o-mini": get_gpt_response,
    "qwen": get_qwen_response,
    "deepseek": get_deepseek_response,
}


class RouterModule:
    def __init__(self, agent_id, memory_limit, llm, global_router_experience_flag, retrieval_num):
        self.agent_id = agent_id
        self.llm = llm
        self.retrieval_num = retrieval_num
        from .global_vars import global_router_experience_pool

        if global_router_experience_flag == False:
            self.router_pool = RouterExperiencePool(capacity=memory_limit)
        else:
            self.router_pool = global_router_experience_pool

        self.get_llm_response = llm_model_response_map[llm]
        self.get_self_agent_info = None


    def set_get_self_agent_info(self, get_self_info):
        self.get_self_agent_info = get_self_info
        
        

    def router_decide_action(self, task, self_info, neighbors_info, task_history, decide_mode="Normal"):
        start_time = time.time()

        experiences = self.router_pool.get_relevant_experiences(task, self.agent_id, self.retrieval_num)
        
        query_prompt = self.router_make_prompt(
            experiences=experiences,
            task=task, 
            decide_mode=decide_mode, 
            self_info=self_info, 
            neighbors_info=neighbors_info
            )
        
        response = self.get_llm_response(
            system_prompt=ROUTER_SYSTEM_PROMPT, 
            query_prompt=query_prompt
            )
        
        response_dict = extract_content_as_dict(long_string=response, short_strings=["DECISION", "REASON", "EXECUTABLE", "DELEGATE", "DESCRIPTION", "NEXT_AGENT_ID"])
        
        decision = response_dict.get('DECISION', '').lower().strip()
        decision = decision.replace("[", "").replace("]", "")
        if decision == "delegate":
            decision = 'forward'
        next_agent_id = None        # Next Agent ID for forwarding
        executable_tasks = None     # Tasks that current Agent can perform after split
        description = None          # Description of task thinking during execution

        if decision in ['forward']:
            
            try:
                next_agent_id = extract_agent_id(str(response_dict.get('NEXT_AGENT_ID', None)))
                
                if next_agent_id == None:
                    next_agent_id = self.find_best_alternative_agent(task.task_type, neighbors_info)
                else:
                    # Check target Agent's capabilities, load, and connectivity
                    if next_agent_id in neighbors_info.keys():
                        next_agent_id_info = neighbors_info[next_agent_id]

                        ability = get_average_abilities_from_task_type(task.task_type, next_agent_id_info['abilities'])

                        if (ability >= 0.5 and
                            next_agent_id_info['current_load'] < 3 and
                            (next_agent_id_info['is_outgoing'] or next_agent_id_info['is_incoming'])):
                            next_agent_id = next_agent_id
                        else:
                            next_agent_id = self.find_best_alternative_agent(task.task_type, neighbors_info)
                    else:
                        next_agent_id = self.find_best_alternative_agent(task.task_type, neighbors_info)

            except Exception as e:
                next_agent_id = self.find_best_alternative_agent(task)

        if decision in ["split"]:
            executable_tasks = response_dict.get('EXECUTABLE', '').lower().strip()

        if decision in ["execute"]:
            description = response_dict.get('DESCRIPTION', '').lower().strip()
            
        execution_time = time.time() - start_time

        router_experience=RouterExperience(
            decision=decision,

            task_id=task.task_id,
            task_type=task.task_type,
            task_major_problem=task.major_problem,
            task_progress_text=task.progress_text,
            task_description=task.description,

            initial_agent_id=0,                
            source_agent_id=self.agent_id,
            target_agent_id=next_agent_id,      

            route_history=task_history,         # Complete history of current task, temporarily just storing it
            execution_time=execution_time,
            success=False,               
            timestamp=time.time()
        )

        # self.router_pool.add_experience(self_info["abilities"],router_experience)

        output_dict = {
            "response": response,
            "response_dict": response_dict,
            "query_prompt":query_prompt,

            "decision": decision,
            "next_agent_id": next_agent_id,
            "executable_tasks": executable_tasks,
            "description": description,
            "execution_time": execution_time,
            "experience": router_experience,
        }

        return output_dict


    def router_decide_next_agent_id(self, task, self_info, neighbors_info, task_history, constraints):
        start_time = time.time()
    
        examples = self.router_pool.get_relevant_experiences(task, self.agent_id, self.retrieval_num)

        query_prompt = self.router_make_prompt_for_next_agent_id(
            experiences=examples,
            task=task, 
            prompt_template=ROUTER_PROMPT_DECIDE_NEXT_AGENT_ID_FORMAT, 
            self_info=self_info, 
            neighbors_info=neighbors_info,
            constraints=constraints
            )

        
        # logger.info(f"query_prompt for router_decide_next_agent_id: {query_prompt}")

        response = self.get_llm_response(
            system_prompt=ROUTER_DECIDE_NEXT_AGENT_ID_SYSTEM_PROMPT, 
            query_prompt=query_prompt
            )
        
        
        # response_dict = parse_decision_text(response)
        response_dict = extract_content_as_dict(long_string=response, short_strings=["DECISION", "REASON", "NEXT_AGENT_ID"])
        decision = response_dict.get('DECISION', '').lower().strip()
        reason = response_dict.get('REASON', '').lower().strip()
        next_agent_id = None

        if decision in ["completed"]:
            # If Agent thinks it's completed
            pass
        elif decision in ["incompleted"]:
            # If Agent thinks it's not completed, need to extract next Agent ID
            try:
                next_agent_id = extract_agent_id(str(response_dict.get('NEXT_AGENT_ID', None)))
                
                if next_agent_id == None:
                    next_agent_id = self.find_best_alternative_agent(task.task_type, neighbors_info)
                else:
                    if next_agent_id in neighbors_info.keys():
                        # Check target Agent's capabilities, load, and connectivity
                        next_agent_id_info = neighbors_info[next_agent_id]
                        ability = get_average_abilities_from_task_type(task.task_type, next_agent_id_info['abilities'])
                        if (ability >= 0.5 and
                            next_agent_id_info['current_load'] < 3 and
                            (next_agent_id_info['is_outgoing'] or next_agent_id_info['is_incoming'])):
                            next_agent_id = next_agent_id
                        else:
                            next_agent_id = self.find_best_alternative_agent(task.task_type, neighbors_info)
                    else:
                        next_agent_id = self.find_best_alternative_agent(task.task_type, neighbors_info)

            except ValueError:
                next_agent_id = self.find_best_alternative_agent(task)



        execution_time = time.time() - start_time
        router_experience=RouterExperience(
            decision=decision,

            task_id=task.task_id,
            task_type=task.task_type,
            task_major_problem=task.major_problem,
            task_progress_text=task.progress_text,
            task_description=task.description,

            initial_agent_id=0,
            source_agent_id=self.agent_id,
            target_agent_id=next_agent_id,     

            route_history=task_history,             
            execution_time=execution_time,
            success=False,               
            timestamp=time.time()
        )

        # self.router_pool.add_experience(self_info["abilities"],router_experience)

        output_dict = {
            "response": response,
            "response_dict": response_dict,
            "query_prompt": query_prompt,
            "decision": decision,
            "reason": reason,
            "next_agent_id": next_agent_id,
            "execution_time": execution_time,
            "experience": router_experience,
        }


        return output_dict



    def router_make_prompt(self, experiences, task, decide_mode, self_info, neighbors_info):
        current_task_type = task.task_type
        if decide_mode == "Normal":
            prompt_template = ROUTER_PROMPT_FORMAT
        elif decide_mode == "Split_and_Execute":
            prompt_template = ROUTER_PROMPT_SPLIT_AND_EXECUTE_FORMAT
        
        experience_strs = []
        for exp in experiences:
            experience_str = ""
            experience_str += f"- Task ID: {exp.task_id}\n"
            experience_str += f"- Task Type: {exp.task_type}\n"
            experience_str += f"- Task Major Task: {exp.task_major_problem}\n"
            experience_str += f"- Task Progress: {exp.task_progress_text}\n"
            experience_str += f"- Task Decision: {exp.decision}\n"
            if exp.target_agent_id != None:
                experience_str += f"- Next Agent ID: {exp.target_agent_id}\n"
            experience_strs.append(experience_str)
        
        if len(experience_strs) == 0:
            total_experience_str = "None experience yet."
        else:
            total_experience_str = "\n".join(experience_strs)
            
        
        agent_info = "\nAvailable Agents:\n"
        for neighbors_id, single_neighbors_info in neighbors_info.items():
            if decide_mode == "Normal" and task.in_forward_history(single_neighbors_info['agent_id']):
                # Once task has been forwarded, don't forward again
                continue 
            
            processed_task_str = ""
            if len(single_neighbors_info['processed_tasks']) == 0:
                processed_task_str = "None processed task yet."
            else:
                for i, exp in enumerate(single_neighbors_info['processed_tasks']):
                    processed_task_str += "\n"
                    processed_task_str += f"- Example {i+1}\n"
                    processed_task_str += f"- Task Type: {exp.task_type}\n"
                    processed_task_str += f"- Task Major Task: {exp.task_major_problem}\n"
                    processed_task_str += f"- Task Progress: {exp.task_progress_text}\n"
                    processed_task_str += f"- Task Solution: {exp.result}\n"
            
            agent_info += f"\nAgent {single_neighbors_info['agent_id']}:"
            agent_info += f"\n- Abilities:"
            for ability_type, ability in single_neighbors_info['abilities'].items():
                agent_info += f"\n  * {ability_type}: {ability:.2f}"
            agent_info += f"\n- Current Load: {single_neighbors_info['current_load']}"


            if float(single_neighbors_info['task_type_success_rate']) != 0:
                agent_info += f"\n- Success Rate for This Task Type: {single_neighbors_info['task_type_success_rate']:.2%}"
            else:
                agent_info += f"\n- Success Rate for This Task Type: Not Succeed Yet."
            agent_info += f"\n- Processed Tasks: {processed_task_str}"
            agent_info += "\n"

        task.complexity = calculate_task_complexity(task)
        task_type_success_rate = float(self_info['success_rate'][current_task_type])
        if task_type_success_rate == 0:
            task_type_success_rate_str = "Not Succeed Yet"
        else:
            task_type_success_rate_str = f"{task_type_success_rate:.2%}"
    
        current_agent_info = ""
        current_agent_info += f"- Agent ID: {self_info['agent_id']}\n"
        current_agent_info += f"- Ability for this task type: \n"
        for ability_type in task_to_ability_map[current_task_type]:
        # for ability_type in task_to_ability_map[current_task_type]:
            current_agent_info += f"\t {ability_type}: {self_info['abilities'][ability_type]}\n"
        current_agent_info += f"- Success rate for this task type: {task_type_success_rate_str}\n"

        if task.progress_text.strip() == "":
            progress_text = "None progress yet."
        else:
            progress_text = task.progress_text
        
        
        query_prompt = prompt_template.format(
            major_problem=task.major_problem,
            experiences=total_experience_str,
            task_context=progress_text,
            current_agent_info=current_agent_info,
            task_type=current_task_type,
            task_description=task.description if task.description != "" else task.major_problem,
            # task_complexity=task.complexity,    
            # current_route="Not Yet",            
            # source_agent_id=self_info["agent_id"],
            agent_info=agent_info,
            )
        

        return query_prompt


    def router_make_prompt_for_next_agent_id(self, experiences, task, prompt_template, self_info, neighbors_info, constraints):
        current_task_type = task.task_type
        experience_strs = []
        for exp in experiences:
            experience_str = ""
            experience_str += f"- Task ID: {exp.task_id}\n"
            experience_str += f"- Task Type: {exp.task_type}\n"
            experience_str += f"- Task Major Task: {exp.task_major_problem}\n"
            experience_str += f"- Task Progress: {exp.task_progress_text}\n"
            experience_str += f"- Task Decision: {exp.decision}\n"
            if exp.target_agent_id != None:
                experience_str += f"- Next Agent ID: {exp.target_agent_id}\n"
            experience_strs.append(experience_str)

        if len(experience_strs) == 0:
            total_experience_str = "None experience yet."
        else:
            total_experience_str = "\n".join(experience_strs)

        agent_info = "\nAvailable Agents:\n"

        for neighbors_id, single_neighbors_info in neighbors_info.items():
            if task.in_forward_history(single_neighbors_info['agent_id']):
                # Once task has been forwarded, don't forward again
                continue 
            
            processed_task_str = ""
            for i, exp in enumerate(single_neighbors_info['processed_tasks']):
                processed_task_str += "\n"
                processed_task_str += f"- Example {i+1}\n"
                processed_task_str += f"- Task Type: {exp.task_type}\n"
                processed_task_str += f"- Task Major Task: {exp.task_major_problem}\n"
                processed_task_str += f"- Task Progress: {exp.task_progress_text}\n"
                processed_task_str += f"- Task Solution: {exp.result}\n"
            
            agent_info += f"\nAgent {single_neighbors_info['agent_id']}:"
            agent_info += f"\n- Abilities:"
            for ability_type, ability in single_neighbors_info['abilities'].items():
                agent_info += f"\n  * {ability_type}: {ability:.2f}"
            agent_info += f"\n- Current Load: {single_neighbors_info['current_load']}"
            if float(single_neighbors_info['task_type_success_rate']) != 0:
                agent_info += f"\n- Success Rate for This Task Type: {single_neighbors_info['task_type_success_rate']:.2%}"
            else:
                agent_info += f"\n- Success Rate for This Task Type: Not Succeed Yet."
            agent_info += f"\n- Processed Tasks: {processed_task_str}"
            agent_info += "\n"
        
        if task.progress_text.strip() == "":
            progress_text = "None progress yet."
        else:
            progress_text = task.progress_text
        
        # Calculate task complexity
        task.complexity = calculate_task_complexity(task)
        task_type_success_rate = float(self_info['success_rate'][current_task_type])
        if task_type_success_rate == 0:
            task_type_success_rate_str = "Not Succeed Yet"
        else:
            task_type_success_rate_str = f"{task_type_success_rate:.2%}"

        current_agent_info = ""
        current_agent_info += f"- Agent ID: {self_info['agent_id']}\n"
        current_agent_info += f"- Ability for this task type: \n"
        for ability_type in task_to_ability_map[current_task_type]:
            current_agent_info += f"\t {ability_type}: {self_info['abilities'][ability_type]}\n"
        current_agent_info += f"- Success rate for this task type: {task_type_success_rate_str}\n"


        query_prompt = prompt_template.format(
            major_problem=task.major_problem,
            task_context=progress_text,
            experiences=total_experience_str,
            current_agent_info=current_agent_info,
            agent_info=agent_info,
            constraints = constraints  # added
            )
        
        return query_prompt


    def find_best_alternative_agent(self, task_type, neighbors_info):
        """Find the most suitable alternative Agent"""
        candidates = []
        current_agent_id = self.agent_id
        for agent_id, info in neighbors_info.items():
            if agent_id != self.agent_id:
                # Basic condition check
                if not (info['is_incoming'] or info['is_outgoing']):
                    continue  # Skip non-neighbor nodes
    
                ability = get_average_abilities_from_task_type(task_type, info['abilities'])
                load = info['current_load']
                success_rate = info['success_rate'][task_type]
                
                # Calculate comprehensive score
                score = (
                    ability * 0.4 +  # Ability weight
                    (1 - load/3) * 0.3 +  # Load weight
                    success_rate * 0.2 +  # Success rate weight
                    (1.0 if info['is_outgoing'] else 0.5) * 0.1  # Connectivity weight
                )
                
                if ability >= 0.5 and load < 3:  # Basic condition filtering
                    candidates.append((score, agent_id))
        
        if candidates:
            candidates.sort(reverse=True)  # Sort by score in descending order
            return candidates[0][1]
        else:
            logger.warning(f"No suitable agent found for task type {task_type}")
            logger.info("We are going to choose a random agent from the neighbors")
            candidates = [agent_id for agent_id in neighbors_info.keys() if agent_id != self.agent_id]
            logger.info(f"The neighbors are {candidates}")
            if not candidates:
                logger.warning(f"No neighbors found for agent {self.agent_id}")
                return current_agent_id
            return random.choice(candidates)

    def get_newest_experiences(self, task_type, k=50):
        return self.router_pool.get_newest_experience(task_type, k)


    def add_experience_to_pool(self, agent_abilities, experience):
        self.router_pool.add_experience(agent_abilities, experience)
        return 0 


    def set_experience_success_state(self, task_id, task_type, success):
        self.router_pool.set_experience_success_state(task_id, task_type, success)


    def get_all_experiences(self):
        return self.router_pool.get_all_experiences()



class ExecutorModule:
    def __init__(self,  agent_id, memory_limit, llm, retrieval_num):
        self.agent_id = agent_id
        
        self.memory_limit = memory_limit
        self.executor_pool = ExecutorExperiencePool(capacity=memory_limit)
        self.llm = llm
        self.retrieval_num = retrieval_num
        self.get_response = llm_model_response_map[self.llm]

    def get_relevant_experiences(self, task):
        return self.executor_pool.get_relevant_experiences(task,top_k=self.retrieval_num)

    def get_relevant_experiences_by_thought(self, task):
        return self.executor_pool.get_relevant_experiences_by_thought(task,top_k=self.retrieval_num)

    def generate_thought(self, task, experiences):
        query_prompt = self.executor_make_prompt_for_thought(
            experiences=experiences, 
            task=task, 
            prompt_template=SPLIT_THOUGHT_PROMPT_FORMAT)

        response = self.get_response(
            system_prompt=THOUGHT_SYSTEM_PROMPT, 
            query_prompt=query_prompt
            )

        response_dict = extract_content_as_dict(long_string=response, short_strings=["RESULT"])
        thought = response_dict.get("RESULT", response) 
        # thought = response_dict["RESULT"]


        output = {
            "response": response,
            "response_dict": response_dict,
            "query_prompt": query_prompt,
            "thought": thought,
        }

        return output


    def executor_execute_task(self, abilities, task, experiences, constraints, few_shot, execution_choice):
        """Execute task"""
        start_time = time.time()
        if execution_choice == "execute":
            thought_query_prompt = self.executor_make_prompt_for_thought(experiences=experiences,
                                                                          task=task,
                                                                          prompt_template=EXECUTOR_THOUGHT_PROMPT_FORMAT)
            logger.info(f"thought query prompt for execute: {thought_query_prompt}")
            thought_response = self.get_response(
                system_prompt=THOUGHT_SYSTEM_PROMPT, 
                query_prompt=thought_query_prompt
                )
            
            thought_response_dict = extract_content_as_dict(long_string=thought_response, short_strings=["RESULT"])
            thought = thought_response_dict.get("RESULT", thought_response)

            logger.info(f"thought before execute: {thought}")

            query_prompt = self.executor_make_prompt(experiences=experiences, 
                                                     task=task, 
                                                     prompt_template=EXECUTOR_PROMPT_FORMAT, 
                                                     constraints=constraints,
                                                     few_shot=few_shot, 
                                                     execution_choice=execution_choice, 
                                                     thought=thought)
            # logger.info(f"query prompt for execute: {query_prompt}")
        elif execution_choice == "split":
            thought = task.thought
            logger.info(f"when making prompt, thought: {thought}")
            query_prompt = self.executor_make_prompt(experiences=experiences,
                                                    task=task, prompt_template=SPLIT_EXECUTOR_PROMPT_FORMAT,
                                                    constraints=constraints,
                                                    few_shot=few_shot, execution_choice=execution_choice,
                                                    thought=thought)

        response = self.get_response(
            system_prompt=EXECUTOR_SYSTEM_PROMPT, 
            query_prompt=query_prompt
            )
        
        response_dict = extract_content_as_dict(long_string=response, short_strings=["RESULT"])

        result = response_dict.get("RESULT", response)

        execution_time = time.time() - start_time
        executor_experience=ExecutorExperience(
            task_id=task.task_id,
            task_type=task.task_type,
            task_major_problem=task.major_problem,
            task_description=task.description,
            task_progress_text=task.progress_text,
            task_thought=task.thought,

            result=result,
            execution_time=execution_time, 
            success=False,                   
            error_type = None,              # error_type not added yet
            source_agent_id=self.agent_id,
            performance_metrics=None,       # performance_metrics not added yet
            timestamp=time.time()
        )

        # self.executor_pool.add_experience(abilities, executor_experience)

        output = {
            "response": response,
            "response_dict": response_dict,
            "query_prompt": query_prompt,
            "result": result,
            "execution_time": execution_time,
            "experience": executor_experience,
        }

        return output
    
    def executor_make_prompt_for_thought(self, experiences, task, prompt_template):
        
        experience_strs = []
        for exp in experiences:
            experience_str = ""
            experience_str += f"- Task ID: {exp.task_id}\n"
            experience_str += f"- Task Type: {exp.task_type}\n"
            experience_str += f"- Task Major Task: {exp.task_major_problem}\n"
            experience_str += f"- Task Progress: {exp.task_progress_text}\n"
            experience_str += f"- Task Description: {exp.task_description}\n"
            experience_str += f"- Task Thought: {exp.task_thought}\n"
            experience_str += f"- Task Result: {exp.result}\n"

        if len(experience_strs) == 0:
            total_experience_str = "None experience yet."
        else:
            total_experience_str = "\n".join(experience_strs)

        if task.progress_text.strip() == "":
            progress_text = "None progress yet."
        else:
            progress_text = task.progress_text
        
        query_prompt = prompt_template.format(
            major_problem = task.major_problem,
            experiences = total_experience_str,
            task_context = progress_text,
            task_description = task.description
        )

        return query_prompt



    def executor_make_prompt(self, experiences, task, prompt_template, constraints, few_shot, execution_choice, thought):
        
        # Build few-shot examples
    
        experience_strs = []
        logger.info(f"when making prompt, len(experiences): {len(experiences)}")
        if execution_choice == "execute":
            logger.info(f"when making prompt, execution_choice: {execution_choice}")
            for exp in experiences:
                experience_str = ""
                experience_str += f"- Task ID: {exp.task_id}\n"
                experience_str += f"- Task Type: {exp.task_type}\n"
                experience_str += f"- Task Major Task: {exp.task_major_problem}\n"
                experience_str += f"- Task Progress: {exp.task_progress_text}\n"
                experience_str += f"- Task Description: {exp.task_description}\n"
                experience_str += f"- Task Result: {exp.result}\n"
                experience_strs.append(experience_str)
        elif execution_choice == "split":
            logger.info(f"when making prompt, execution_choice: {execution_choice}")
            for exp in experiences:
                if exp.task_progress_text.strip() == "":
                    continue
                experience_str = ""
                experience_str += f"- Task ID: {exp.task_id}\n"
                experience_str += f"- Task Type: {exp.task_type}\n"
                experience_str += f"- Task Major Task: {exp.task_major_problem}\n"
                experience_str += f"- Task Progress: {exp.task_progress_text}\n"
                experience_strs.append(experience_str)

        if len(experience_strs) == 0:
            total_experience_str = "None experience yet."
        else:
            total_experience_str = "\n".join(experience_strs)


        if task.progress_text.strip() == "":
            progress_text = "None progress yet."
        else:
            progress_text = task.progress_text
        

        query_prompt = prompt_template.format(
            major_problem = task.major_problem,
            experiences = total_experience_str,
            task_context = progress_text,
            task_description = task.description,
            thought = thought,
            constraints = constraints,
            few_shot = few_shot,
        )

        return query_prompt


    def add_experience_to_pool(self, agent_abilities, experience):
        self.executor_pool.add_experience(agent_abilities, experience)
        return 0 

    def set_experience_success_state(self, task_id, task_type, success):
        self.executor_pool.set_experience_success_state(task_id, task_type, success)
        # self.smart_evivation
        return 0


    def get_all_experiences(self):
        return self.executor_pool.get_all_experiences()





class Agent:
    def __init__(self, agent_config, global_router_experience_flag):
        self.agent_config = agent_config
        self.agent_id = agent_config["id"]
        self.config = agent_config["config"]
        self.abilities = self.config["abilities"]
        self.llm = self.config["LLM"]
        print(self.llm)

        self.executor_retrieval_num = self.config["executor_retrieval_num"]
        self.router_retrieval_num = self.config["router_retrieval_num"]

        self.collect_neighbors_info = None
        
        self.current_load = 0
        self.task_correlations = {}
        
        self.success_rate = {}
        self.total_task_num = {}
        self.successful_task_num = {}
        for key, value in BASIC_PROBLEM_COMPLEXITY.items():
            self.success_rate[key] = 0
            self.total_task_num[key] = 0
            self.successful_task_num[key] = 0


        self.decay_rate = self.config["decay_rate"]
        self.decay_interval = self.config["decay_interval"]
        self.decay_count_down = {}

        self.router_module = RouterModule(agent_id=self.agent_id, memory_limit=self.config["router_memory_limit"], llm=self.llm, global_router_experience_flag=global_router_experience_flag, retrieval_num = self.router_retrieval_num)
        self.router_module.set_get_self_agent_info(self.get_self_info)

        self.executor_module = ExecutorModule(
            agent_id=self.agent_id, 
            memory_limit=self.config["executor_memory_limit"], 
            llm=self.llm,
            retrieval_num = self.executor_retrieval_num
            )

    # new ability to classify the task
    def decide_task_type(self, task_chain):
        current_task = task_chain.get_current_task()

        description = current_task.description if current_task.description != "" else "You need to finish this code task."
        # description = "You need to finish this code task."

        # Let current agent decide task type based on major_task and description
        query_prompt = ROUTER_CLASSIFY_PROMPT.format(major_problem=current_task.major_problem, description=description)
        logger.info(f'The prompt for router to decide task type is: \n{query_prompt}')
        response = llm_model_response_map[self.llm](system_prompt='You are a coding problem classifier.', query_prompt=query_prompt)

        # Extract answer from response, handle various possible cases
        try:
            extract_answer = extract_content_as_dict(response, ['CATEGORY'])
            logger.info(response)
            category = extract_answer['CATEGORY'].lower().strip().strip('[').strip(']').strip()   # Lowercase, remove possible outer spaces and []
            logger.info(f'Agent {self.agent_id} think the task belongs to [{category}].')  # May be multiple
            abilities = category.split(',') # May also have spaces after splitting
            for ability in abilities:
                ability = ability.strip()  # Remove spaces
                if ability in self.abilities.keys():
                    current_task.set_task_type(ability)
                    task_chain.set_task_type(ability)
                    logger.info(f"Finally! classify the task as [{ability}].")
                    break
        except:
            pass
        if task_chain.task_type == None:
            # If not None, can reuse the original one, which is more robust
            # If not found, uniformly set to code_generation
            # tmp_task_type = random.choice(list(self.abilities.keys()))
            tmp_task_type = 'others'
            current_task.set_task_type(tmp_task_type)
            task_chain.set_task_type(tmp_task_type)
            logger.info(f"Cannot find category, random choose [{tmp_task_type}] as the task type.")

    def set_collect_neighbors_info(self, collect_neighbors_info):
        self.collect_neighbors_info = collect_neighbors_info



    def decide_action(self, task_chain, forward_times):
        current_task = task_chain.get_current_task()

        # description = current_task.description if current_task.description != "" else "You need to finish this code task."
        # # Let current agent decide task type based on major_task and description
        # query_prompt = ROUTER_CLASSIFY_PROMPT.format(major_problem=current_task.major_problem, description=description)
        # logger.info(query_prompt)
        # response = llm_model_response_map[self.llm](system_prompt='You are a coding problem classifier.', query_prompt=query_prompt)
        # # Extract answer from response, handle various possible cases
        # extract_answer = extract_content_as_dict(response, ['CATEGORY'])
        # logger.info(response)
        # # logger.info(extract_answer['CATEGORY'])
        # category = extract_answer['CATEGORY'].lower().strip().strip('[').strip(']').strip()   # Lowercase, remove spaces
        # logger.info(f'Agent {self.agent_id} think the task belongs to [{category}].')
        # # category = extract_answers['CATEGORIES'].lower().strip().replace(' ', '_')
        # # abilities = extract_answers['ABILITIES'].lower().split(',')
        # if category in self.abilities.keys():
        #     current_task.task_type = category
        #     task_chain.task_type = category
        #     logger.info(f"Finally! classify the task as {category}.")
        # else:
        #     current_task.task_type = random.choice(list(self.abilities.keys()))
        #     logger.info(f"Cannot find {category}, random choose {current_task.task_type} as the task type.")
        self.decide_task_type(task_chain)   # Update current task type

        # Start task processing
        self_info = self.get_self_info()
        neighbors_info = self.collect_neighbors_info(agent_id=self.agent_id, task=current_task)

        # If forward times too low, change to forced execution; if no neighbors left, also change to forced execution
        decide_mode = "Normal"
        No_Neighbor_Flag = True
        for key, value in neighbors_info.items():
            if key not in current_task.forward_history:
                No_Neighbor_Flag = False
        if (No_Neighbor_Flag) or (forward_times <=0):
            decide_mode = "Split_and_Execute"
        
        # For new tasks, classify
        output = self.router_module.router_decide_action(
            task=current_task, 
            self_info=self_info, 
            neighbors_info=neighbors_info,
            task_history=task_chain.task_history,
            decide_mode=decide_mode,
            )
        
        newest_experiences = self.router_module.get_newest_experiences(task_type=current_task.task_type, k=50)
        self.update_task_correlations(task_type=current_task.task_type, newest_experiences=newest_experiences)
        
        return output


    def decide_next_agent_id(self, task_chain,constraints):
        current_task = task_chain.get_current_task()

        self_info = self.get_self_info()
        neighbors_info = self.collect_neighbors_info(agent_id = self.agent_id, task=current_task)

        output = self.router_module.router_decide_next_agent_id(
            task=current_task, 
            self_info=self_info, 
            neighbors_info=neighbors_info,
            task_history=task_chain.task_history,
            constraints=constraints,
            )
        
        newest_experiences = self.router_module.get_newest_experiences(task_type=current_task.task_type, k=50)
        self.update_task_correlations(task_type=current_task.task_type, newest_experiences=newest_experiences)

        return output



    def execute_task(self, task_chain, constraints, split_constraints, thought_constraints, few_shot, user_react=True, decision="execute"):
        # Use Agent's executor to execute task, return task execution result
        current_task = task_chain.get_current_task()
        logger.info(f"Agent {self.agent_id} is Executing Task {current_task.task_id}")

        self.current_load += 1
        self.total_task_num[current_task.task_type] += 1

        relevant_experiences = self.executor_module.get_relevant_experiences(current_task)

        if user_react == True:

            output = self.executor_module.generate_thought(current_task, relevant_experiences)
            current_task.set_thought(output["thought"])
            
            relevant_experiences_by_thought = self.executor_module.get_relevant_experiences_by_thought(current_task)
            relevant_experiences = relevant_experiences_by_thought
        
        logger.info(f"when execute task, len(relevant_experiences): {len(relevant_experiences)}")
        # Execute task based on Thought retrieval results
        if decision == "execute":
            output = self.executor_module.executor_execute_task(abilities=self.abilities, task=current_task, experiences=relevant_experiences, constraints=constraints, few_shot=few_shot, execution_choice="execute")
        elif decision == "split":   
            output = self.executor_module.executor_execute_task(abilities=self.abilities, task=current_task, experiences=relevant_experiences, constraints=split_constraints, few_shot=few_shot, execution_choice="split")

        self.current_load -= 1
        if current_task.task_type not in self.decay_count_down.keys():
            self.decay_count_down[current_task.task_type] = self.decay_interval
        return output


    # Here, when we retrieve, is it coarse-grained retrieval or retrieval by thought? This belongs to Router retrieval, we'll look at it later
    # After deleting this function
    def get_relevant_experence(self, task):
        return self.executor_module.executor_pool.get_relevant_experiences(task, top_k=self.executor_retrieval_num)


    def get_self_info(self):
        self_info = {
            "agent_id": self.agent_id,
            "current_load": self.current_load,
            "success_rate": self.success_rate, 
            "abilities": self.abilities,
        }
        return self_info


    def add_executor_experience(self, experience):
        self.executor_module.add_experience_to_pool(self.abilities, experience)
        

    def add_router_experience(self, experience):
        self.router_module.add_experience_to_pool(self.abilities, experience)
    




    def get_all_experiences(self):
        router_experiences = self.router_module.get_all_experiences()
        executor_experiences = self.executor_module.get_all_experiences()

        return {
            "router_experiences": router_experiences,
            "executor_experiences": executor_experiences,
        }



    # def update_abilities(self, task_type, complexity, execution_time, success):
    #     """Update ability values"""
    #     # If success is True, increase ability; otherwise, no change for now
    #     if success == True:

    #         self.successful_task_num[task_type] += 1
    #         self.success_rate[task_type] = self.successful_task_num[task_type] / self.total_task_num[task_type]

    #         # Basic ability improvement
    #         # Improve based on task difficulty
    #         ability_gain = 0.1 * complexity
            
    #         # Adjust based on execution time
    #         time_factor = 1.0 / (1 + execution_time / 10)  # Normalized time factor

    #         ability_gain *= time_factor
            
    #         # Update main abilities
    #         for ability_type in task_to_ability_map[task_type]:
    #             self.abilities[ability_type] = min(2.0, self.abilities[ability_type] + ability_gain)   # 2.0 is ability upper limit

    #         for related_type, correlation in self.task_correlations[task_type].items():
    #             if correlation > 0.3:  # Only update strongly correlated abilities
    #                 for ability_type in task_to_ability_map[related_type]:
    #                     self.abilities[ability_type] = min(2.0, self.abilities[ability_type] + ability_gain * correlation * 0.5)
    #     else:
    #         pass



    def update_abilities(self, task_type, complexity, execution_time, success, pass_rate):
        """Update ability values"""
        if pass_rate > 0:
            # If success is True, increase ability; otherwise, no change for now
            ability_gain = round(pass_rate * 0.04 * complexity,2)   # Complexity <1
            # Limit to two decimal places
            for ability_type in task_to_ability_map[task_type]:
                self.abilities[ability_type] = min(2.0, self.abilities[ability_type] + ability_gain)  # 2.0 is ability upper limit

        if success:  
            self.successful_task_num[task_type] += 1
            self.success_rate[task_type] = self.successful_task_num[task_type] / self.total_task_num[task_type]

            # Basic ability improvement
            ability_gain = 0.05  # When task succeeds, ability increases by fixed value (or you can customize this value)

            # Update main abilities
            for ability_type in task_to_ability_map[task_type]:
                self.abilities[ability_type] = min(2.0, self.abilities[ability_type] + ability_gain)  # 2.0 is ability upper limit

            # Update related abilities, boost ability based on task correlation
            for related_type, correlation in self.task_correlations[task_type].items():
                if correlation > 0.3:  # Only update strongly correlated abilities
                    for ability_type in task_to_ability_map[related_type]:
                        self.abilities[ability_type] = min(2.0, self.abilities[ability_type] + ability_gain * correlation * 0.5)

        else:  # Task failed, ability remains unchanged
            pass



    def update_task_correlations(self, task_type, newest_experiences) -> None:
        """Update task type correlations"""
        if task_type not in self.task_correlations:
            # If Agent hasn't encountered this task yet, expand it
            self.task_correlations[task_type] = {}
        
        for experience in newest_experiences:
            if experience.task_type != task_type:
                correlation = self.calculate_correlation(newest_experiences, task_type, experience.task_type)
                current_correlation = self.task_correlations[task_type].get(experience.task_type, 0)
                self.task_correlations[task_type][experience.task_type] = current_correlation * 0.9 + correlation * 0.1
                    



    def calculate_correlation(self, newest_history, task_type1, task_type2):
        """Calculate correlation between two task types"""
        # Based on recent history (at most the latest 100 histories)
        task_type1_count = sum(1 for entry in newest_history if entry['task_type'] == task_type1)
        task_type2_count = sum(1 for entry in newest_history if entry['task_type'] == task_type2)
        
        if task_type1_count == 0 or task_type2_count == 0:
            return 0.0
            
        cooccurrence = 0
        for i in range(len(newest_history)-1):
            if newest_history[i]['task_type'] == task_type1 and newest_history[i+1]['task_type'] == task_type2:
                cooccurrence += 1
                
        correlation = cooccurrence / min(task_type1_count, task_type2_count)
        return correlation



    def set_executor_pool_result(self, task_id, task_type, success):
        # If execution successful, reset corresponding counter
        for key, value in self.decay_count_down.items():
            self.decay_count_down[key] = value - 1
        if success:
            self.decay_count_down[task_type] = self.decay_interval

        for key, value in self.decay_count_down.items():
            if value <= 0:
                self.decay_abilities(task_type=key)
                self.decay_count_down[key] = self.decay_interval
            
        self.executor_module.set_experience_success_state(task_id, task_type, success)



    def set_router_pool_result(self, task_id, task_type, success):
        self.router_module.set_experience_success_state(task_id, task_type, success)


    
    def get_pool_info(self):
        """Get experience pool experience count"""
        return (self.router_module.router_pool,self.executor_module.executor_pool)
    

    def decay_abilities(self, task_type) -> None:
        """Ability decay"""
        for ability_type in task_to_ability_map[task_type]:
            self.abilities[ability_type] = max(0.1, self.abilities[ability_type] * (1 - self.decay_rate))

