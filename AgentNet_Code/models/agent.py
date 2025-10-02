from collections import defaultdict
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
from dataclasses import dataclass
import heapq
from models.experience_pool import ExperiencePool, Experience
import openai

class Agent:
    """Enhanced Agent with improved experience accumulation"""
    def __init__(self, agent_id: int, abilities: Dict[str, float] = None,
                 memory_limit: int = 50, decay_rate: float = 0.01):
        self.agent_id = agent_id
        self.abilities = abilities or {
            'math_problem': 1.0,
            'code_generation': 1.0,
            'translation': 1.0,
            'summary': 1.0,
            'story': 1.0,
            'combined_task': 1.0,
        }
        self.weight = 1.0
        self.experience_pool = ExperiencePool(capacity=memory_limit)
        self.processed_tasks = 0
        self.neighbors = []
        self.current_load = 0
        self.decay_rate = decay_rate
        
        # 任务关联图：记录任务类型之间的关联强度
        self.task_correlation = defaultdict(lambda: defaultdict(float))

    def call_openai_api(self, full_prompt: str) -> Tuple[str, bool]:
        """调用 OpenAI API 处理任务"""
        try:
            # 设置 API 密钥
            openai.api_key = "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" ## api key for this project

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Chat model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=100,
                temperature=0.7,
            )

            # Extract the generated text
            generated_text = response.choices[0].message['content'].strip()
            return generated_text, True

        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "", False
        
    def calculate_task_complexity(self, task) -> float:
        """计算任务复杂度"""
        base_complexity = {
            'math_problem': 3.0,
            'code_generation': 2.5,
            'translation': 1.5,
            'summary': 1.0,
            'story': 1.0,
            'combined_task': 3.5
        }
        
        # 基础复杂度
        complexity = base_complexity.get(task.task_type, 1.0)
        
        # 根据提示长度调整
        length_factor = len(task.prompt) / 100  # 标准化长度
        complexity *= (1 + 0.1 * length_factor)
        
        # 考虑任务优先级
        complexity *= (1 + 0.2 * task.priority)
        
        # 如果是组合任务，增加复杂度
        if 'combined' in task.task_type.lower():
            complexity *= 1.5
            
        return min(5.0, complexity)  # 将复杂度限制在5以内
        
    def process_task(self, task) -> Tuple[Optional[str], bool]:
        """处理任务并累积经验"""
        self.current_load += 1
        start_time = time.time()
        
        # 获取相关经验
        relevant_experiences = self.experience_pool.get_relevant_experiences(task)
        few_shot_examples = "\n".join([
            f"Example {i+1} ({exp.task_type}): {exp.prompt} -> {exp.result}"
            for i, exp in enumerate(relevant_experiences)
        ])
        
        # 构建完整提示
        full_prompt = f"{few_shot_examples}\nNow, solve: {task.prompt}"
        
        # 调用API处理任务
        response, success = self.call_openai_api(full_prompt)
        execution_time = time.time() - start_time
        
        # 记录经验
        difficulty = self.calculate_task_complexity(task)
        experience = Experience(
            task_id=task.task_id,
            task_type=task.task_type,
            prompt=task.prompt,
            execution_time=execution_time,
            success=success,
            result=response if success else "",
            timestamp=time.time(),
            difficulty=difficulty
        )
        self.experience_pool.add_experience(experience)
        
        # 更新能力值
        if success:
            self._update_abilities(task, execution_time, difficulty)
            self._update_task_correlations(task, relevant_experiences)
            
        self.processed_tasks += 1
        self.current_load -= 1
        
        return response, success
        
    def _update_abilities(self, task: 'Task', execution_time: float, 
                         difficulty: float) -> None:
        """更新能力值"""
        # 基础能力提升
        ability_gain = 0.1 * difficulty
        # 根据执行时间调整
        time_factor = 1.0 / (1 + execution_time / 10)  # 归一化时间因子
        ability_gain *= time_factor
        
        # 更新主要能力
        self.abilities[task.task_type] = min(
            2.0,  # 能力上限
            self.abilities[task.task_type] + ability_gain
        )
        # 更新相关能力
        for related_type, correlation in self.task_correlation[task.task_type].items():
            if correlation > 0.3:  # 只更新强相关的能力
                self.abilities[related_type] = min(
                    2.0,
                    self.abilities[related_type] + ability_gain * correlation * 0.5
                )
                
    def _update_task_correlations(self, current_task: 'Task', 
                                relevant_experiences: List[Experience]) -> None:
        """更新任务类型之间的关联强度"""
        for exp in relevant_experiences:
            if exp.task_type != current_task.task_type:
                # 增加双向关联强度
                current_correlation = self.task_correlation[current_task.task_type][exp.task_type]
                new_correlation = current_correlation * 0.9 + 0.1  # 渐进增强
                self.task_correlation[current_task.task_type][exp.task_type] = new_correlation
                self.task_correlation[exp.task_type][current_task.task_type] = new_correlation
                
    def decide_to_process(self, task: 'Task') -> bool:
        """决定是否处理任务"""
        # 计算处理概率
        base_ability = self.abilities.get(task.task_type, 0)
        
        # 考虑相关任务类型的能力
        related_abilities = sum(
            self.abilities.get(related_type, 0) * correlation
            for related_type, correlation in self.task_correlation[task.task_type].items()
        )
        
        # 综合能力评分
        total_ability = base_ability * 0.7 + related_abilities * 0.3
        # 根据当前负载调整概率
        load_factor = max(0.2, 1 - self.current_load / 3)
        # 计算最终处理概率
        processing_probability = min(1.0, total_ability * load_factor)
        
        return random.random() < processing_probability
        
    def decay_abilities(self) -> None:
        """能力衰减"""
        for task_type in self.abilities:
            self.abilities[task_type] = max(
                0.1,
                self.abilities[task_type] * (1 - self.decay_rate)
            )
            
    def get_status(self) -> Dict:
        """获取Agent状态信息"""
        return {
            'agent_id': self.agent_id,
            'abilities': self.abilities,
            'current_load': self.current_load,
            'processed_tasks': self.processed_tasks,
            'task_correlations': dict(self.task_correlation),
            'experience_count': self.experience_pool.get_experience_count()
        }
    