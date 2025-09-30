

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
    """Router experience record"""
    decision: str               # 'forward', 'split', 'execute'

    task_id: int
    task_type: str
    task_major_problem: str     # Main objective of the task (original task)
    task_progress_text: str     # Task progress
    task_description: str       # Task description

    initial_agent_id: int       # Initial Agent assigned to the task
    source_agent_id: int        # Current Agent where the task is located
    target_agent_id: int        # Target agent if forwarding

    route_history: List[Dict]   # Complete routing history of current TaskChain
    execution_time: float       # Execution time
    success: bool               # Whether decision was successful, default is failure
    timestamp: float = field(default_factory=time.time)  # Time when this experience was recorded

    embedding: np.ndarray = None  # Vector representation




@dataclass
class ExecutorExperience:
    """Executor experience record"""
    task_id: int
    task_type: str
    task_major_problem: str               # Task description
    task_description: str               # Task description
    # task_context: str               # Task context, mainly previous tasks and progress
    task_progress_text: str               # Task context, mainly previous tasks and progress
    
    task_thought: str

    result: str                         # Execution result
    execution_time: float               # Execution time
    success: bool                       # Whether successful
    error_type: Optional[str] = None    # Error type (if failed)
    source_agent_id: int = None         # Task source Agent
    performance_metrics: Dict[str, float] = field(default_factory=dict)  
    # Performance metrics

    embedding: Optional[np.ndarray] = None  # Vector representation
    timestamp: float = field(default_factory=time.time)  # Recording time




class BaseExperiencePool:
    """Base class for experience pool"""
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
        """Compute vector representation of text"""
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
    #             # 1. Time decay
    #             time_factor = np.exp(-(current_time - exp.timestamp) / 86400)
                
    #             # 2. Success rate impact
    #             success_factor = 1.5 if exp.success else 0.5
                
    #             # 3. Routing path length (shorter paths are more valuable)
    #             path_length = len(exp.route_history)
    #             path_factor = 1.0 / (1 + path_length)
                
    #             # 4. Similarity with new experience
    #             similarity = self.calculate_similarity(new_experience, exp)
                
    #             # 5. Source matching degree
    #             source_factor = 1.2 if exp.source_agent_id == new_experience.source_agent_id else 1.0
                
    #             # Comprehensive score
    #             score = (0.25 * time_factor +
    #                     0.25 * success_factor +
    #                     0.2 * path_factor +
    #                     0.15 * similarity +
    #                     0.15 * source_factor)
                        
    #             scores.append((score, i, existed_task_type))
                
    #     # Evict the least important experience
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
                    'major_problem': exp.task_major_problem,     # Main objective of the task (original task)
                    'progress_text': exp.task_progress_text,   # Task progress
                    'task_description': exp.task_description,  # Task description
                    'agent_ability': temp_ability_dict,  # Agent's ability in this task type 
                    'timestamp': exp.timestamp,
                })

        # Current task information (new experience)
        new_task_info = {
            'task_id': new_experience.task_id,
            'task_type': new_experience.task_type,
            'major_problem': new_experience.task_major_problem,     # Main objective of the task (original task)
            'progress_text': new_experience.task_progress_text,   # Task progress
            'task_description': new_experience.task_description,  # Task description
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
        """Calculate similarity between two routing experiences"""
        # Calculate vector similarity
        vector_similarity = np.dot(exp1.embedding, exp2.embedding) / (np.linalg.norm(exp1.embedding) * np.linalg.norm(exp2.embedding))
        
        # Consider routing decision similarity
        decision_similarity = 1.0 if exp1.decision == exp2.decision else 0.5
        
        # Consider target agent similarity
        target_similarity = 1.0 if exp1.target_agent_id == exp2.target_agent_id else 0.7
        
        # Weighted average
        return 1 * vector_similarity + 0 * decision_similarity + 0 * target_similarity

    def get_relevant_experiences(self, task, source_agent_id, top_k, threshold=0.7, success_only=True):
        """Get relevant routing experiences, add threshold check"""
        query_embedding = self.compute_embedding(task.major_problem + task.progress_text + task.description)
        
        all_scores = []
        for experiences in self.retrieval_experiences.values():
            for exp in experiences:
                if success_only and not exp.success:
                    continue
                # Basic similarity, mainly similarity between task context and task description
                similarity = np.dot(query_embedding, exp.embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(exp.embedding))
                all_scores.append((-similarity, exp))    

        # Sort by similarity
        sorted_scores = sorted(all_scores, key=lambda x: x[0])
        score_list = []
        for score, exp in sorted_scores:
            score_list.append(score)
        # print(f"score_list: {score_list}")
        logger.info(f"score_list for {task.task_id}: {score_list}")
        logger.info(f"top_k: {top_k}")
        if len(sorted_scores) == 0:
            return []
        # If the number of valid experiences is less than top_k, return all valid experiences
        valid_experiences = [exp for _, exp in sorted_scores if -1 *_ >= threshold]
        if len(valid_experiences) < top_k:
            return valid_experiences

        return valid_experiences[:top_k]



    # def get_relevant_experiences(self, task, source_agent_id, top_k, threshold=0.7, success_only=True):
    #     """Get relevant routing experiences, add threshold check and random selection"""
    #     query_embedding = self.compute_embedding(task.major_problem + task.progress_text + task.description)
        
    #     all_scores = []
    #     for experiences in self.retrieval_experiences.values():
    #         for exp in experiences:
    #             if success_only and not exp.success:
    #                 continue
    #             # Calculate similarity between experience and task
    #             similarity = np.dot(query_embedding, exp.embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(exp.embedding))
    #             all_scores.append((similarity, exp))    

    #     # Filter experiences that meet the similarity threshold
    #     valid_experiences = [exp for similarity, exp in all_scores if similarity >= threshold]
        
    #     # If the number of valid experiences is less than top_k, return all valid experiences
    #     if len(valid_experiences) < top_k:
    #         return valid_experiences
        
    #     # Randomly select top_k experiences
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
    #     """Smart eviction strategy"""
    #     # experiences = self.retrieval_experiences[task_type]
    #     current_time = time.time()
        
    #     scores = []

    #     for existed_task_type, experiences in self.retrieval_experiences.items():
    #         for i, exp in enumerate(experiences):

    #             time_factor = np.exp(-(current_time - exp.timestamp) / 86400)
                
    #             # 2. Success rate impact
    #             success_factor = 1.5 if exp.success else 0.5
                
    #             # 3. Execution efficiency (faster execution is more valuable)
    #             # Execution time not yet implemented
    #             efficiency_factor = 1.0 / (1 + exp.execution_time / 60)  # Normalized to minutes
                
    #             # 4. Similarity with new experience
    #             similarity = self.calculate_similarity(new_experience, exp)
                
    #             # Comprehensive score
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
    #     #     # 1. Time decay
    #     #     time_factor = np.exp(-(current_time - exp.timestamp) / 86400)
            
    #     #     # 2. Success rate impact
    #     #     success_factor = 1.5 if exp.success else 0.5
            
    #     #     # 3. Execution efficiency (faster execution is more valuable)
    #     #     # Execution time not yet implemented
    #     #     efficiency_factor = 1.0 / (1 + exp.execution_time / 60)  # Normalized to minutes
            
    #     #     # 4. Similarity with new experience
    #     #     similarity = self.calculate_similarity(new_experience, exp)
            
    #     #     # Comprehensive score
    #     #     score = (0.3 * time_factor +
    #     #             0.3 * success_factor +
    #     #             0.2 * efficiency_factor +
    #     #             0.2 * similarity)
                    
    #     #     scores.append((score, i))
            
    #     # Evict the least important experience
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
                    'major_problem': exp.task_major_problem,     # Main objective of the task (original task)
                    'progress_text': exp.task_progress_text,   # Task progress
                    'task_description': exp.task_description,  # Task description
                    'progress_text': exp.task_progress_text,               # Task context, mainly previous tasks and progress
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
            'major_problem': new_experience.task_major_problem,     # Main objective of the task (original task)
            'progress_text': new_experience.task_progress_text,   # Task progress
            'task_description': new_experience.task_description,  # Task description
            'progress_text': new_experience.task_progress_text,               # Task context, mainly previous tasks and progress
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
        """Calculate similarity between two execution experiences"""
        # Vector similarity
        A_embedding = encode_model.encode(exp1.task_major_problem + exp1.task_progress_text + exp1.task_description)
        B_embedding = encode_model.encode(exp2.task_major_problem + exp2.task_progress_text + exp2.task_description)

        cos_similarity = calculate_cos_similarity_A_and_B(A_embedding, B_embedding)

        # Execution time similarity
        time_diff = abs(exp1.execution_time - exp2.execution_time)
        time_similarity = 1.0 / (1 + time_diff / 60)  # Normalized to minutes
        return 0.7 * cos_similarity + 0.3 * time_similarity
    

    def get_relevant_experiences(self, task, success_only=True, top_k=0, threshold=0.7):
        """Get relevant execution experiences, based on task context text, add threshold check"""
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
        # Get embeddings for all texts
        all_texts_embeddings = encode_model.encode(all_texts, batch_size=512).reshape(-1, 1024)
        cos_similarity = calculate_cos_similarity_A_and_Batch_B(query_embedding, all_texts_embeddings)
        all_efficiency_bonus = np.array(all_efficiency_bonus)
        logger.info(f"top_k: {top_k}")
        logger.info(f"executor cos_similarity: {cos_similarity}")
        logger.info(f"executor all_efficiency_bonus: {all_efficiency_bonus}")

        # Calculate weighted scores
        scores = cos_similarity * (1 + 0.2 * all_efficiency_bonus)
        logger.info(f"executor scores: {scores}")
        # If score is an array, convert to scalar for comparison
        valid_experiences = []
        for score, exp in zip(scores[0], all_experiences):
            if score >= threshold:
                valid_experiences.append((score, exp))
        # If the number of valid experiences is less than top_k, return all valid experiences
        logger.info(f"executor len(valid_experiences): {len(valid_experiences)}")
        if len(valid_experiences) < top_k:
            return [exp for _, exp in valid_experiences]

        # Sort by score
        sorted_scores = sorted(valid_experiences, key=lambda x: x[0], reverse=True)

        # Return top_k experiences
        return [exp for _, exp in sorted_scores[:top_k]]

    def get_relevant_experiences_by_thought(self, task, success_only=True, top_k=0, threshold=0.7):
        """Get relevant execution experiences based on the Thought of the experience, add threshold check"""
        query_embedding = encode_model.encode(task.thought).reshape(1,-1)

        all_thoughts = []
        all_efficiency_bonus = []
        all_experiences = []
        
        # Iterate through all experiences
        for experiences in self.retrieval_experiences.values():
            for exp in experiences:
                if success_only and not exp.success:
                    continue
                all_thoughts.append(exp.task_thought)
                all_efficiency_bonus.append(1.0 / (1 + exp.execution_time / 60))
                all_experiences.append(exp)

        if len(all_experiences) == 0:
            return []

        # Get embeddings for all thoughts
        all_thoughts_embeddings = encode_model.encode(all_thoughts, batch_size=512).reshape(-1, 1024)
        cos_similarity = calculate_cos_similarity_A_and_Batch_B(query_embedding, all_thoughts_embeddings)
        all_efficiency_bonus = np.array(all_efficiency_bonus)

        # Calculate weighted scores
        scores = cos_similarity * (1 + 0.2 * all_efficiency_bonus)

        # If score is an array, convert to scalar for comparison
        valid_experiences = []
        for score, exp in zip(scores[0], all_experiences):
            if score >= threshold:
                valid_experiences.append((score, exp))
        
        logger.info(f"top_k: {top_k}")
        logger.info(f"executor get_relevant_experiences_by_thought len(valid_experiences): {len(valid_experiences)}")

        # If the number of valid experiences is less than top_k, return all valid experiences
        if len(valid_experiences) < top_k:
            experiences = [exp for _, exp in valid_experiences]
            logger.info(f"executor get_relevant_experiences_by_thought len(experiences): {len(experiences)}")
            return experiences

        # Sort by score
        sorted_scores = sorted(valid_experiences, key=lambda x: x[0], reverse=True)

        # Return top_k experiences
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