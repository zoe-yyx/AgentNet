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
    """Data structure for storing task execution experience"""
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
    """Intelligent experience pool for managing experience storage and retrieval"""
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.experiences: Dict[str, List[Experience]] = defaultdict(list)
        self.task_vectors = {}
        
    def add_experience(self, experience: Experience) -> None:
        """Add new experience to the experience pool"""
        task_type = experience.task_type
        
        # Calculate vector representation of the task
        if not experience.embeddings:
            experience.embeddings = self._compute_embeddings(experience.prompt)
        print("experience.embeddings", experience.embeddings)

        # Ensure experience pool is initialized for new task types
        if task_type not in self.experiences:
            self.experiences[task_type] = []
        # If no related tasks, skip related task lookup
        if len(self.experiences[task_type]) == 0:
            experience.related_tasks = []
        else:
            # Find related tasks based on existing experiences
            experience.related_tasks = self._find_related_tasks(experience)
        
        # If capacity limit is reached, perform smart eviction
        if len(self.experiences[task_type]) >= self.capacity:
            self._smart_eviction(task_type, experience)
        else:
            self.experiences[task_type].append(experience)
            

    def _compute_embeddings(self, prompt: str) -> np.ndarray:
      """Use OpenAI API to compute task embedding representation"""
      try:
          # Call OpenAI's embedding model
          response = openai.Embedding.create(
              model="text-embedding-ada-002",  # Use appropriate OpenAI embedding model
              input=prompt
          )
          # Extract embedding and return
          embeddings = response['data'][0]['embedding']
          return np.array(embeddings)
      except Exception as e:
          print(f"OpenAI API request failed, error message: {e}")
          # If OpenAI request fails, fallback to simple bag-of-words model
          words = set(prompt.lower().split())
          vector = np.zeros(100)
          for i, word in enumerate(words):
              vector[hash(word) % 100] = 1
          return vector 

    def _calculate_similarity(self, exp1: Experience, exp2: Experience) -> float:
        """Calculate similarity between two experiences"""
        # Combine multiple similarity metrics
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
        
        # Weighted combination
        weights = [0.4, 0.3, 0.2, 0.1]  # Text similarity has highest weight
        return sum([
            weights[0] * text_similarity,
            weights[1] * vector_similarity,
            weights[2] * type_similarity,
            weights[3] * difficulty_similarity
        ])
        
    def _find_related_tasks(self, experience: Experience, threshold: float = 0.6) -> List[int]:
        """Find tasks related to the given experience"""
        related = []
        for task_type, experiences in self.experiences.items():
            for exp in experiences:
                similarity = self._calculate_similarity(experience, exp)
                if similarity > threshold:
                    related.append(exp.task_id)
        return related
        
    def _smart_eviction(self, task_type: str, new_experience: Experience) -> None:
        """Smart experience eviction strategy"""
        experiences = self.experiences[task_type]
        
        # Calculate importance score for each experience
        scores = []
        current_time = time.time()
        
        for i, exp in enumerate(experiences):
            # 1. Time decay
            time_factor = np.exp(-(current_time - exp.timestamp) / 86400)  # 24-hour decay
            
            # 2. Success rate impact
            success_factor = 1.5 if exp.success else 0.5
            
            # 3. Relevance impact
            similarity = self._calculate_similarity(new_experience, exp)
            
            # 4. Task difficulty
            difficulty_factor = exp.difficulty / 5.0  # Assume max difficulty is 5
            
            # 5. Number of related tasks
            connectivity = len(exp.related_tasks) / len(experiences)
            
            # Comprehensive score
            score = (0.3 * time_factor + 
                    0.2 * success_factor + 
                    0.2 * similarity +
                    0.15 * difficulty_factor +
                    0.15 * connectivity)
                    
            scores.append((score, i))
            
        # Evict the least important experience
        scores.sort()  # Sort by score in ascending order
        idx_to_remove = scores[0][1]
        self.experiences[task_type].pop(idx_to_remove)
        self.experiences[task_type].append(new_experience)

    def get_relevant_experiences(self, task, top_k: int = 3) -> List[Experience]:
        """Get the most relevant experiences for the given task"""
        # Calculate vector representation of the task
        task_vector = self._compute_embeddings(task.prompt)

        # Check if there are any experiences, if not, return empty list directly
        if not any(self.experiences.values()):
            return []  # If no experiences, return empty list

        # Build priority queue containing similarity scores
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
                # Push similarity, timestamp and Experience object into heap, use negative similarity for max heap
                heapq.heappush(scores, (-similarity, exp.timestamp, exp))

        # Return the most relevant top_k experiences
        return [heapq.heappop(scores)[2] for _ in range(min(top_k, len(scores)))]


    def get_experience_count(self) -> int:
        """Get total number of experiences in the experience pool"""
        return sum(len(exps) for exps in self.experiences.values())