from collections import defaultdict
import numpy as np
from typing import List, Dict
import time
import difflib
from dataclasses import dataclass
import heapq
import openai

@dataclass
class Experience:
    """存储任务执行经验的数据结构"""
    task_id: int
    task_type: str
    prompt: str
    execution_time: float
    success: bool
    result: str
    timestamp: float
    difficulty: float
    embeddings: np.ndarray = None
    related_tasks: List[int] = None

class ExperiencePool:
    """智能经验池，管理经验存储和检索"""
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.experiences: Dict[str, List[Experience]] = defaultdict(list)
        self.task_vectors = {}
        openai.api_key = "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" ## api key for this project
        
    def add_experience(self, experience: Experience) -> None:
        """添加新经验到经验池"""
        task_type = experience.task_type
        
        # 计算任务的向量表示
        if not experience.embeddings:
            experience.embeddings = self._compute_embeddings(experience.prompt)
        print("experience.embeddings", experience.embeddings)

        # 确保为新任务类型初始化经验池 (initialize experience pool for new task types)
        if task_type not in self.experiences:
            self.experiences[task_type] = []
        # 如果没有相关任务，跳过相关任务查找 (skip related task lookup if no prior experiences)
        if len(self.experiences[task_type]) == 0:
            experience.related_tasks = []
        else:
            # 找到相关任务 (find related tasks based on existing experiences)
            experience.related_tasks = self._find_related_tasks(experience)
        
        # 如果达到容量限制，执行智能淘汰
        if len(self.experiences[task_type]) >= self.capacity:
            self._smart_eviction(task_type, experience)
        else:
            self.experiences[task_type].append(experience)
            

    def _compute_embeddings(self, prompt: str) -> np.ndarray:
      """使用 OpenAI API 计算任务的嵌入表示"""
      try:
          # 调用 OpenAI 的 embedding 模型
          response = openai.Embedding.create(
              model="text-embedding-ada-002",  # 使用合适的 OpenAI 嵌入模型
              input=prompt
          )
          # 提取嵌入并返回
          embeddings = response['data'][0]['embedding']
          return np.array(embeddings)
      except Exception as e:
          print(f"OpenAI API 请求失败，错误信息: {e}")
          # 如果 OpenAI 请求失败，仍然降级为简单的词袋模型
          words = set(prompt.lower().split())
          vector = np.zeros(100)
          for i, word in enumerate(words):
              vector[hash(word) % 100] = 1
          return vector 

    def _calculate_similarity(self, exp1: Experience, exp2: Experience) -> float:
        """计算两个经验之间的相似度"""
        # 结合多个相似度指标
        text_similarity = difflib.SequenceMatcher(
            None, 
            exp1.prompt.lower(), 
            exp2.prompt.lower()
        ).ratio()
        # print("exp1",exp1,"exp2",exp2)
        vector_similarity = np.dot(exp1.embeddings, exp2.embeddings) / \
                          (np.linalg.norm(exp1.embeddings) * np.linalg.norm(exp2.embeddings))
                          
        type_similarity = 1.0 if exp1.task_type == exp2.task_type else 0.0
        
        difficulty_similarity = 1 - abs(exp1.difficulty - exp2.difficulty) / max(exp1.difficulty, exp2.difficulty)
        
        # 加权组合
        weights = [0.4, 0.3, 0.2, 0.1]  # 文本相似度权重最高
        return sum([
            weights[0] * text_similarity,
            weights[1] * vector_similarity,
            weights[2] * type_similarity,
            weights[3] * difficulty_similarity
        ])
        
    def _find_related_tasks(self, experience: Experience, threshold: float = 0.6) -> List[int]:
        """找到与给定经验相关的任务"""
        related = []
        for task_type, experiences in self.experiences.items():
            for exp in experiences:
                similarity = self._calculate_similarity(experience, exp)
                if similarity > threshold:
                    related.append(exp.task_id)
        return related
        
    def _smart_eviction(self, task_type: str, new_experience: Experience) -> None:
        """智能经验淘汰策略"""
        experiences = self.experiences[task_type]
        
        # 计算每个经验的重要性分数
        scores = []
        current_time = time.time()
        
        for i, exp in enumerate(experiences):
            # 1. 时间衰减
            time_factor = np.exp(-(current_time - exp.timestamp) / 86400)  # 24小时的衰减
            
            # 2. 成功率影响
            success_factor = 1.5 if exp.success else 0.5
            
            # 3. 相关性影响
            similarity = self._calculate_similarity(new_experience, exp)
            
            # 4. 任务难度
            difficulty_factor = exp.difficulty / 5.0  # 假设最高难度为5
            
            # 5. 关联任务数量
            connectivity = len(exp.related_tasks) / len(experiences)
            
            # 综合评分
            score = (0.3 * time_factor + 
                    0.2 * success_factor + 
                    0.2 * similarity +
                    0.15 * difficulty_factor +
                    0.15 * connectivity)
                    
            scores.append((score, i))
            
        # 淘汰最不重要的经验
        scores.sort()  # 按分数升序排序
        idx_to_remove = scores[0][1]
        self.experiences[task_type].pop(idx_to_remove)
        self.experiences[task_type].append(new_experience)

    def get_relevant_experiences(self, task, top_k: int = 3) -> List[Experience]:
        """获取与给定任务最相关的经验"""
        # 计算任务的向量表示
        task_vector = self._compute_embeddings(task.prompt)

        # 检查是否有任何经验，如果没有，直接返回空列表
        if not any(self.experiences.values()):
            return []  # 如果没有经验，返回空列表

        # 构建包含相似度分数的优先队列
        scores = []
        for experiences in self.experiences.values():
            for exp in experiences:
                similarity = self._calculate_similarity(
                    Experience(
                        task_id=task.task_id,
                        task_type=task.task_type,
                        prompt=task.prompt,
                        execution_time=0,
                        success=False,
                        result="",
                        timestamp=time.time(),
                        difficulty=0,
                        embeddings=task_vector
                    ),
                    exp
                )
                # 把相似度、时间戳和 Experience 对象一起推入堆中，使用负的相似度来实现最大堆
                heapq.heappush(scores, (-similarity, exp.timestamp, exp))

        # 返回最相关的 top_k 个经验
        return [heapq.heappop(scores)[2] for _ in range(min(top_k, len(scores)))]


    def get_experience_count(self) -> int:
        """获取经验池中的总经验数量"""
        return sum(len(exps) for exps in self.experiences.values())