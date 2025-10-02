

import sys
import numpy as np
import time
import random
from collections import defaultdict
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from FlagEmbedding import FlagModel
import logging
from config.setting import task_to_ability_map
logger = logging.getLogger(__name__)

encode_model = FlagModel("BAAI/bge-large-en-v1.5", 
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:") 


from .utils import calculate_cos_similarity_A_and_Batch_B, calculate_cos_similarity_A_and_B

MAX_SIZE = sys.maxsize


@dataclass
class RouterExperience:
    """Router的经验记录"""
    decision: str               # 'forward', 'split', 'execute'

    task_id: int
    task_type: str
    task_major_problem: str     # 任务的主要目标(原始的任务)
    task_progress_text: str     # 任务的进展
    task_description: str       # 任务的描述

    initial_agent_id: int       # 任务最初的分配给的Agent
    source_agent_id: int        # 任务当前所在的Agent
    target_agent_id: int        # 如果转发，目标agent

    route_history: List[Dict]   # 当前这条任务(TaskChain)的完整路由历史   
    execution_time: float       # 执行时间
    success: bool               # 决策是否成功, 默认失败
    timestamp: float = field(default_factory=time.time)  # 记录这条经验入库的时间

    embedding: np.ndarray = None  # 向量表示




@dataclass
class ExecutorExperience:
    """Executor的经验记录"""
    task_id: int
    task_type: str
    task_major_problem: str               # 任务的描述
    task_description: str               # 任务的描述
    # task_context: str               # 任务的上下文, 主要是前文的任务和进展
    task_progress_text: str               # 任务的上下文, 主要是前文的任务和进展
    
    task_thought: str

    result: str                         # 执行结果
    execution_time: float               # 执行时间
    success: bool                       # 是否成功              
    error_type: Optional[str] = None    # 错误类型(如果失败)     TODO 这条暂时还没写
    source_agent_id: int = None         # 任务来源Agent         
    performance_metrics: Dict[str, float] = field(default_factory=dict)  
    # 性能指标

    embedding: Optional[np.ndarray] = None  # 向量表示
    timestamp: float = field(default_factory=time.time)  # 记录时间




class BaseExperiencePool:
    """经验池基类"""
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.embedding_cache = {}
        
        self.success_experiences = defaultdict(list)
        self.failure_experiences = defaultdict(list)
        self.retrieval_experiences = defaultdict(list)



    def __len__(self):
        length = 0
        for key, value in self.retrieval_experiences.items():
            length = max(len(value), length)
        return length
    


    def compute_embedding(self, text: str) -> np.ndarray:
        """计算文本的向量表示"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        global encode_model
        embedding = encode_model.encode(text)

        self.embedding_cache[text] = embedding
        return embedding
    

class RouterExperiencePool(BaseExperiencePool):

    def add_experience(self, abilities, experience: RouterExperience) -> None:
        task_type = experience.task_type
        # ensure existence of experience embedding(task_major_problem + task_progress_text + task_description)
        if not experience.embedding:
            experience.embedding = self.compute_embedding(experience.task_major_problem + experience.task_progress_text + experience.task_description)
        
        if experience.success == False:
            self.failure_experiences[task_type].append(experience)
            return 0
        elif experience.success == True:
            self.success_experiences[task_type].append(experience)

            retrieval_experience_num = 0
            for key, value in self.retrieval_experiences.items():
                for exp in value:
                    retrieval_experience_num += 1
            
            if retrieval_experience_num >= self.capacity and self.capacity >= 0:
                self._smart_eviction(abilities, task_type, experience)
            else:
                self.retrieval_experiences[task_type].append(experience)
                
    # def _smart_eviction_rule_based(self, task_type: str, new_experience: RouterExperience) -> None:
    #     """Rule-based eviction strategy"""
    #     # experiences = self.retrieval_experiences[task_type]
    #     current_time = time.time()
        
    #     scores = []
    #     for existed_task_type, experiences in self.retrieval_experiences.items():
    #         for i, exp in enumerate(experiences):
    #             # 1. 时间衰减
    #             time_factor = np.exp(-(current_time - exp.timestamp) / 86400)
                
    #             # 2. 成功率影响
    #             success_factor = 1.5 if exp.success else 0.5
                
    #             # 3. 路由路径长度（更短的路径更valuable）
    #             path_length = len(exp.route_history)
    #             path_factor = 1.0 / (1 + path_length)
                
    #             # 4. 与新经验的相似度
    #             similarity = self.calculate_similarity(new_experience, exp)
                
    #             # 5. 来源匹配度
    #             source_factor = 1.2 if exp.source_agent_id == new_experience.source_agent_id else 1.0
                
    #             # 综合评分
    #             score = (0.25 * time_factor +
    #                     0.25 * success_factor +
    #                     0.2 * path_factor +
    #                     0.15 * similarity +
    #                     0.15 * source_factor)
                        
    #             scores.append((score, i, existed_task_type))
                
    #     # 淘汰最不重要的经验
    #     scores = sorted(scores)
    #     idx_to_remove = scores[0][1]
    #     task_type_to_remove = scores[0][2]
    #     self.retrieval_experiences[task_type_to_remove].pop(idx_to_remove)
    #     self.retrieval_experiences[task_type].append(new_experience)





    def _smart_eviction(self, abilities, task_type: str, new_experience: RouterExperience) -> None:
        """Smart eviction strategy, let the LLM Agent decide which trajectory to remove"""
        current_time = time.time()

        # Generate input for the agent: the new experience and the existing experiences in the pool
        trajectory_info = []
        for existed_task_type, experiences in self.retrieval_experiences.items():
            for exp in experiences:
                temp_ability_dict = {}
                for ability_name in task_to_ability_map[exp.task_type]:
                    temp_ability_dict[ability_name] = abilities[ability_name]
                # Add relevant information, including success rate and ability score
                trajectory_info.append({
                    'task_id': exp.task_id,
                    'task_type': exp.task_type,
                    'success': exp.success,
                    'major_problem': exp.task_major_problem,     # 任务的主要目标(原始的任务)
                    'progress_text': exp.task_progress_text,   # 任务的进展
                    'task_description': exp.task_description,  # 任务的描述
                    'agent_ability': temp_ability_dict,  # Agent's ability in this task type 
                    'timestamp': exp.timestamp,
                })

        # Current task information (new experience)
        new_task_info = {
            'task_id': new_experience.task_id,
            'task_type': new_experience.task_type,
            'major_problem': new_experience.task_major_problem,     # 任务的主要目标(原始的任务)
            'progress_text': new_experience.task_progress_text,   # 任务的进展
            'task_description': new_experience.task_description,  # 任务的描述
            'timestamp': new_experience.timestamp,
        }

        # Prepare the prompt for LLM Agent decision-making
        agent_input = self.prepare_agent_input(trajectory_info, new_task_info)

        # Generate the decision through LLM: return the task_id of the trajectory to remove
        trajectory_to_remove = self.make_agent_decision(agent_input)

        # Evict the selected trajectory based on agent's decision
        self.remove_selected_trajectory(task_type, trajectory_to_remove, new_experience)

    def prepare_agent_input(self, trajectory_info: List[Dict], new_task_info: Dict) -> str:
        """Prepare the input prompt for the LLM agent"""
        input_str = "Here is the list of historical router trajectories (task experiences):\n"
        for traj in trajectory_info:
            input_str += f"Task ID: {traj['task_id']}, Task Type: {traj['task_type']}, Agent's Ability in this task: {traj['agent_ability']}, Major Problem: {traj['major_problem']}, Progress Text: {traj['progress_text']}\n"

        input_str += "\nCurrent Task Information:\n"
        input_str += f"Task ID: {new_task_info['task_id']}, Task Type: {new_task_info['task_type']}, Major Problem: {new_task_info['major_problem']}, Progress Text: {new_task_info['progress_text']}\n"

        input_str += "\nPlease decide which trajectory would be most beneficial for improving your abilities. Choose the least valuable trajectory (either the new one or an existing one) to remove, and return the Task ID of the trajectory to remove.\n"

        input_str += "\nReturn the Task ID of the trajectory to evict.\n"

        return input_str

    def make_agent_decision(self, agent_input: str) -> str:
        """Generate decision based on the agent's evaluation using LLM"""
        # Feed input to LLM model (like encode_model) for decision-making
        decision_embedding = encode_model.encode(agent_input)
        
        # Interpret the output decision from LLM: which task ID to evict
        trajectory_to_remove = self.interpret_decision(decision_embedding)
        
        return trajectory_to_remove

    def interpret_decision(self, decision_embedding: np.ndarray) -> str:
        """Interpret the decision from the LLM model output"""
        # Assuming the model outputs an embedding that corresponds to the task ID of the trajectory to remove
        # For example, the model might output a high score for the most likely task ID to evict
        task_id_to_remove = np.argmax(decision_embedding)  # Assuming the model generates a vector for task IDs
        
        return str(task_id_to_remove)

    def remove_selected_trajectory(self, task_type: str, task_id_to_remove: str, new_experience: RouterExperience) -> None:
        """Remove the selected trajectory from the experience pool based on Task ID"""
        # Check if the trajectory to remove is the new experience
        if task_id_to_remove == new_experience.task_id:
            # If the new experience is chosen to be removed, don't add it
            print(f"Evicting new experience with Task ID: {task_id_to_remove}")
        else:
            # Remove the specified experience by Task ID
            for exp in self.retrieval_experiences[task_type]:
                if exp.task_id == task_id_to_remove:
                    self.retrieval_experiences[task_type].remove(exp)
                    print(f"Evicting existing experience with Task ID: {task_id_to_remove}")
                    break
    
    def calculate_similarity(self, exp1: RouterExperience, exp2: RouterExperience) -> float:
        # TODO
        # 如何计算路由经验可能需要进行修改
        """计算两个路由经验的相似度"""
        # 计算向量相似度
        vector_similarity = np.dot(exp1.embedding, exp2.embedding) / (np.linalg.norm(exp1.embedding) * np.linalg.norm(exp2.embedding))
        
        # 考虑路由决策相似度
        decision_similarity = 1.0 if exp1.decision == exp2.decision else 0.5
        
        # 考虑目标Agent相似度
        target_similarity = 1.0 if exp1.target_agent_id == exp2.target_agent_id else 0.7
        
        # 加权平均
        return 1 * vector_similarity + 0 * decision_similarity + 0 * target_similarity

    def get_relevant_experiences(self, task, source_agent_id, top_k, threshold=0.7, success_only=True):
        """获取相关的路由经验，增加阈值检查"""
        query_embedding = self.compute_embedding(task.major_problem + task.progress_text + task.description)
        
        all_scores = []
        for experiences in self.retrieval_experiences.values():
            for exp in experiences:
                if success_only and not exp.success:
                    continue
                # 文本的基础相似度，主要是任务的上下文和任务的描述之间的相似度
                similarity = np.dot(query_embedding, exp.embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(exp.embedding))
                all_scores.append((-similarity, exp))    

        # 按照相似度进行排序
        sorted_scores = sorted(all_scores, key=lambda x: x[0])
        score_list = []
        for score, exp in sorted_scores:
            score_list.append(score)
        # print(f"score_list: {score_list}")
        logger.info(f"score_list for {task.task_id}: {score_list}")
        logger.info(f"top_k: {top_k}")
        if len(sorted_scores) == 0:
            return []
        # 如果符合条件的经验数量少于top_k，返回所有符合条件的经验
        valid_experiences = [exp for _, exp in sorted_scores if -1 *_ >= threshold]
        if len(valid_experiences) < top_k:
            return valid_experiences

        return valid_experiences[:top_k]



    # def get_relevant_experiences(self, task, source_agent_id, top_k, threshold=0.7, success_only=True):
    #     """获取相关的路由经验，增加阈值检查并随机选择"""
    #     query_embedding = self.compute_embedding(task.major_problem + task.progress_text + task.description)
        
    #     all_scores = []
    #     for experiences in self.retrieval_experiences.values():
    #         for exp in experiences:
    #             if success_only and not exp.success:
    #                 continue
    #             # 计算经验与任务的相似度
    #             similarity = np.dot(query_embedding, exp.embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(exp.embedding))
    #             all_scores.append((similarity, exp))    

    #     # 按相似度阈值筛选符合条件的经验
    #     valid_experiences = [exp for similarity, exp in all_scores if similarity >= threshold]
        
    #     # 如果符合条件的经验数量少于top_k，返回所有符合条件的经验
    #     if len(valid_experiences) < top_k:
    #         return valid_experiences
        
    #     # 随机选择top_k个经验
    #     selected_experiences = random.sample(valid_experiences, top_k)
        
    #     return selected_experiences



    def get_newest_experience(self,task_type, k=50):
        experiences = self.retrieval_experiences[task_type]
        return experiences[:-k]

    def set_experience_success_state(self, task_id, task_type, success):
        for index in range(len(self.retrieval_experiences[task_type])):
            if self.retrieval_experiences[task_type][index].task_id == task_id:
                self.retrieval_experiences[task_type][index].success = success

    def get_all_experiences(self):

        return_list = []

        for task_type, experiences in self.retrieval_experiences.items():            
            for experience in experiences:
                temp_dict = {
                    "decision" : experience.decision,

                    "task_id" : experience.task_id,
                    "task_type" : experience.task_type,
                    "task_major_problem" : experience.task_major_problem,
                    "task_description" : experience.task_description,
                    "task_progress_text" : experience.task_progress_text,

                    "initial_agent_id" : experience.initial_agent_id,
                    "source_agent_id" : experience.source_agent_id,
                    "target_agent_id" : experience.target_agent_id,

                    "execution_time" : experience.execution_time,
                    "success" : experience.success,
                    "timestamp" : experience.timestamp,
                }
                return_list.append(temp_dict)

        return return_list


class ExecutorExperiencePool(BaseExperiencePool):
    def add_experience(self, abilities, experience: ExecutorExperience) -> None:

        task_type = experience.task_type
        # ensure existence of experience embedding(task_major_problem + task_progress_text + task_description)
        if not experience.embedding:
            experience.embedding = self.compute_embedding(experience.task_major_problem + experience.task_progress_text + experience.task_description)
        
        if experience.success == False:
            self.failure_experiences[task_type].append(experience)
            return 0
        elif experience.success == True:
            self.success_experiences[task_type].append(experience)

            retrieval_experience_num = 0
            for key, value in self.retrieval_experiences.items():
                for exp in value:
                    retrieval_experience_num += 1
            
            if retrieval_experience_num >= self.capacity and self.capacity >= 0:
                self._smart_eviction(abilities, task_type, experience)
            else:
                self.retrieval_experiences[task_type].append(experience)

    # def _smart_eviction_rule_based(self, abilities, task_type: str, new_experience: ExecutorExperience) -> None:
    #     """智能淘汰策略"""
    #     # experiences = self.retrieval_experiences[task_type]
    #     current_time = time.time()
        
    #     scores = []

    #     for existed_task_type, experiences in self.retrieval_experiences.items():
    #         for i, exp in enumerate(experiences):

    #             time_factor = np.exp(-(current_time - exp.timestamp) / 86400)
                
    #             # 2. 成功率影响
    #             success_factor = 1.5 if exp.success else 0.5
                
    #             # 3. 执行效率（更快的执行更valuable）
    #             # 尚未写好执行时间
    #             efficiency_factor = 1.0 / (1 + exp.execution_time / 60)  # 标准化到分钟级
                
    #             # 4. 与新经验的相似度
    #             similarity = self.calculate_similarity(new_experience, exp)
                
    #             # 综合评分
    #             score = (0.3 * time_factor +
    #                     0.3 * success_factor +
    #                     0.2 * efficiency_factor +
    #                     0.2 * similarity)
                        
    #             scores.append((score, i, existed_task_type))
                
    #     scores = sorted(scores)
    #     idx_to_remove = scores[0][1]
    #     task_type_to_remove = scores[0][2]

    #     self.retrieval_experiences[task_type_to_remove].pop(idx_to_remove)
    #     self.retrieval_experiences[task_type].append(new_experience)

    #     # for i, exp in enumerate(experiences):
    #     #     # 1. 时间衰减
    #     #     time_factor = np.exp(-(current_time - exp.timestamp) / 86400)
            
    #     #     # 2. 成功率影响
    #     #     success_factor = 1.5 if exp.success else 0.5
            
    #     #     # 3. 执行效率（更快的执行更valuable）
    #     #     # 尚未写好执行时间
    #     #     efficiency_factor = 1.0 / (1 + exp.execution_time / 60)  # 标准化到分钟级
            
    #     #     # 4. 与新经验的相似度
    #     #     similarity = self.calculate_similarity(new_experience, exp)
            
    #     #     # 综合评分
    #     #     score = (0.3 * time_factor +
    #     #             0.3 * success_factor +
    #     #             0.2 * efficiency_factor +
    #     #             0.2 * similarity)
                    
    #     #     scores.append((score, i))
            
    #     # 淘汰最不重要的经验
    #     # scores = sorted(scores)
    #     # idx_to_remove = scores[0][1]
    #     # self.retrieval_experiences[task_type].pop(idx_to_remove)
    #     # self.retrieval_experiences[task_type].append(new_experience)


    def _smart_eviction(self, abilities, task_type: str, new_experience: ExecutorExperience) -> None:
        """Smart eviction strategy, let the LLM Agent decide which trajectory to remove"""
        current_time = time.time()

        # Generate input for the agent: the new experience and the existing experiences in the pool
        trajectory_info = []
        for existed_task_type, experiences in self.retrieval_experiences.items():
            for exp in experiences:
                # Add relevant information, including success rate and ability score
                temp_ability_dict = {}
                for ability_name in task_to_ability_map[exp.task_type]:
                    temp_ability_dict[ability_name] = abilities[ability_name]
                trajectory_info.append({
                    'task_id': exp.task_id,
                    'task_type': exp.task_type,
                    'success': exp.success,
                    'major_problem': exp.task_major_problem,     # 任务的主要目标(原始的任务)
                    'progress_text': exp.task_progress_text,   # 任务的进展
                    'task_description': exp.task_description,  # 任务的描述
                    'progress_text': exp.task_progress_text,               # 任务的上下文, 主要是前文的任务和进展
                    'task_thought': exp.task_thought,
                    'result': exp.result,
                    'agent_ability': temp_ability_dict,  # Agent's ability in this task type 
                    'timestamp': exp.timestamp,
                })


        # Current task information (new experience)
        temp_ability_dict = {}
        for ability_name in task_to_ability_map[exp.task_type]:
            temp_ability_dict[ability_name] = abilities[ability_name]
        new_task_info = {
            'task_id': new_experience.task_id,
            'task_type': new_experience.task_type,
            'success': new_experience.success,
            'major_problem': new_experience.task_major_problem,     # 任务的主要目标(原始的任务)
            'progress_text': new_experience.task_progress_text,   # 任务的进展
            'task_description': new_experience.task_description,  # 任务的描述
            'progress_text': new_experience.task_progress_text,               # 任务的上下文, 主要是前文的任务和进展
            'task_thought': new_experience.task_thought,
            'result': new_experience.result,
            'agent_ability': temp_ability_dict,  # Agent's ability in this task type 4
            'timestamp': new_experience.timestamp,
        }

        # Prepare the prompt for LLM Agent decision-making
        agent_input = self.prepare_agent_input(trajectory_info, new_task_info, abilities)

        # Generate the decision through LLM: return the task_id of the trajectory to remove
        trajectory_to_remove = self.make_agent_decision(agent_input)

        # Evict the selected trajectory based on agent's decision
        self.remove_selected_trajectory(task_type, trajectory_to_remove, new_experience)

    def prepare_agent_input(self, trajectory_info: List[Dict], new_task_info: Dict, abilities) -> str:
        """Prepare the input prompt for the LLM agent"""
        input_str = "Here is the list of historical trajectories (task experiences):\n"
        for traj in trajectory_info:
            input_str += f"Task ID: {traj['task_id']}, Task Type: {traj['task_type']}, Agent's Ability in this task: {traj['agent_ability']}, Major Problem: {traj['major_problem']}, Progress Text: {traj['progress_text']}\n"

        input_str += "\nCurrent Task Information:\n"
        input_str += f"Task ID: {new_task_info['task_id']}, Task Type: {new_task_info['task_type']}, Major Problem: {new_task_info['major_problem']}, Progress Text: {new_task_info['progress_text']}\n"
        # abilities = str(abilities)
        # input_str += f"Your overall abilities are {abilities}"
        input_str += "\nPlease decide which trajectory would be most beneficial for improving your abilities. Choose the least valuable trajectory (either the new one or an existing one) to remove, and return the Task ID of the trajectory to remove.\n"

        input_str += "\nReturn the Task ID of the trajectory to evict.\n"

        return input_str

    def make_agent_decision(self, agent_input: str) -> str:
        """Generate decision based on the agent's evaluation using LLM"""
        # Feed input to LLM model (like encode_model) for decision-making
        decision_embedding = encode_model.encode(agent_input)
        
        # Interpret the output decision from LLM: which task ID to evict
        trajectory_to_remove = self.interpret_decision(decision_embedding)
        
        return trajectory_to_remove

    def interpret_decision(self, decision_embedding: np.ndarray) -> str:
        """Interpret the decision from the LLM model output"""
        # Assuming the model outputs an embedding that corresponds to the task ID of the trajectory to remove
        # For example, the model might output a high score for the most likely task ID to evict
        task_id_to_remove = np.argmax(decision_embedding)  # Assuming the model generates a vector for task IDs
        
        return str(task_id_to_remove)

    def remove_selected_trajectory(self, task_type: str, task_id_to_remove: str, new_experience: RouterExperience) -> None:
        """Remove the selected trajectory from the experience pool based on Task ID"""
        # Check if the trajectory to remove is the new experience
        if task_id_to_remove == new_experience.task_id:
            # If the new experience is chosen to be removed, don't add it
            print(f"Evicting new experience with Task ID: {task_id_to_remove}")
        else:
            # Remove the specified experience by Task ID
            for exp in self.retrieval_experiences[task_type]:
                if exp.task_id == task_id_to_remove:
                    self.retrieval_experiences[task_type].remove(exp)
                    print(f"Evicting existing experience with Task ID: {task_id_to_remove}")
                    break
    

    def calculate_similarity(self, exp1: ExecutorExperience, exp2: ExecutorExperience) -> float:
        """计算两个执行经验的相似度"""
        # 向量相似度
        A_embedding = encode_model.encode(exp1.task_major_problem + exp1.task_progress_text + exp1.task_description)
        B_embedding = encode_model.encode(exp2.task_major_problem + exp2.task_progress_text + exp2.task_description)

        cos_similarity = calculate_cos_similarity_A_and_B(A_embedding, B_embedding)

        # 执行时间相似度
        time_diff = abs(exp1.execution_time - exp2.execution_time)
        time_similarity = 1.0 / (1 + time_diff / 60)  # 标准化到分钟级
        return 0.7 * cos_similarity + 0.3 * time_similarity
    

    def get_relevant_experiences(self, task, success_only=True, top_k=0, threshold=0.7):
        """获取相关的执行经验，基于任务上下文文本，增加阈值检查"""
        query_embedding = encode_model.encode(task.major_problem + task.progress_text + task.description).reshape(1, -1)

        all_texts = []
        all_efficiency_bonus = []
        all_experiences = []

        for experiences in self.retrieval_experiences.values():
            for exp in experiences:
                if success_only and not exp.success:
                    continue
                all_texts.append(exp.task_major_problem + exp.task_progress_text + exp.task_description)
                all_efficiency_bonus.append(1.0 / (1 + exp.execution_time / 60))
                all_experiences.append(exp)

        if len(all_experiences) == 0:
            return []
        logger.info(f"len(all_experiences) {task.task_id}: {len(all_experiences)}")
        # 获取所有文本的嵌入
        all_texts_embeddings = encode_model.encode(all_texts, batch_size=512).reshape(-1, 1024)
        cos_similarity = calculate_cos_similarity_A_and_Batch_B(query_embedding, all_texts_embeddings)
        all_efficiency_bonus = np.array(all_efficiency_bonus)
        logger.info(f"top_k: {top_k}")
        logger.info(f"executor cos_similarity: {cos_similarity}")
        logger.info(f"executor all_efficiency_bonus: {all_efficiency_bonus}")

        # 计算加权得分
        scores = cos_similarity * (1 + 0.2 * all_efficiency_bonus)
        logger.info(f"executor scores: {scores}")
        # 如果 score 是数组，转换为标量进行比较\
        valid_experiences = []
        for score, exp in zip(scores[0], all_experiences):
            if score >= threshold:
                valid_experiences.append((score, exp))
        # valid_experiences = [(score[0] if isinstance(score, np.ndarray) else score, exp) for score, exp in zip(scores, all_experiences) if (score[0] if isinstance(score, np.ndarray) else score) >= threshold]
        # 如果符合条件的经验数量少于 top_k，返回所有符合条件的经验
        logger.info(f"executor len(valid_experiences): {len(valid_experiences)}")
        if len(valid_experiences) < top_k:
            return [exp for _, exp in valid_experiences]

        # 按得分排序
        sorted_scores = sorted(valid_experiences, key=lambda x: x[0], reverse=True)

        # 返回 top_k 个经验
        return [exp for _, exp in sorted_scores[:top_k]]

    def get_relevant_experiences_by_thought(self, task, success_only=True, top_k=0, threshold=0.7):
        """根据经验的Thought来获取相关的执行经验，增加阈值检查"""
        query_embedding = encode_model.encode(task.thought).reshape(1,-1)

        all_thoughts = []
        all_efficiency_bonus = []
        all_experiences = []
        
        # 遍历所有经验
        for experiences in self.retrieval_experiences.values():
            for exp in experiences:
                if success_only and not exp.success:
                    continue
                all_thoughts.append(exp.task_thought)
                all_efficiency_bonus.append(1.0 / (1 + exp.execution_time / 60))
                all_experiences.append(exp)

        if len(all_experiences) == 0:
            return []

        # 获取所有思想的嵌入
        all_thoughts_embeddings = encode_model.encode(all_thoughts, batch_size=512).reshape(-1, 1024)
        cos_similarity = calculate_cos_similarity_A_and_Batch_B(query_embedding, all_thoughts_embeddings)
        all_efficiency_bonus = np.array(all_efficiency_bonus)

        # 计算加权得分
        scores = cos_similarity * (1 + 0.2 * all_efficiency_bonus)

        # 如果 score 是数组，转换为标量进行比较
        valid_experiences = []
        for score, exp in zip(scores[0], all_experiences):
            if score >= threshold:
                valid_experiences.append((score, exp))
        
        logger.info(f"top_k: {top_k}")
        logger.info(f"executor get_relevant_experiences_by_thought len(valid_experiences): {len(valid_experiences)}")

        # 如果符合条件的经验数量少于 top_k，返回所有符合条件的经验
        if len(valid_experiences) < top_k:
            experiences = [exp for _, exp in valid_experiences]
            logger.info(f"executor get_relevant_experiences_by_thought len(experiences): {len(experiences)}")
            return experiences

        # 按得分排序
        sorted_scores = sorted(valid_experiences, key=lambda x: x[0], reverse=True)

        # 返回 top_k 个经验
        experiences = [exp for _, exp in sorted_scores[:top_k]]
        logger.info(f"executor get_relevant_experiences_by_thought len(experiences): {len(experiences)}")
        return experiences


    def set_experience_success_state(self, task_id, task_type, success):
        for index in range(len(self.retrieval_experiences[task_type])):
            if self.retrieval_experiences[task_type][index].task_id == task_id:
                self.retrieval_experiences[task_type][index].success = success
                
                

    def get_all_experiences(self):

        return_list = []

        for task_type, experiences in self.retrieval_experiences.items():            
            for experience in experiences:
                temp_dict = {
                    "task_id" : experience.task_id,
                    "task_type" : experience.task_type,
                    "task_major_problem" : experience.task_major_problem,
                    "task_description" : experience.task_description,
                    "task_progress_text" : experience.task_progress_text,
                    "task_thought" : experience.task_thought,
                    "result" : experience.result,
                    "execution_time" : experience.execution_time,
                    "success" : experience.success,
                    "timestamp" : experience.timestamp,
                }
                return_list.append(temp_dict)

        return return_list