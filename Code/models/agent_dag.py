from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple
import heapq
import random


class AgentDAG:
    def __init__(self):
        """
        Initialize an enhanced DAG structure with edge weights and cycle detection.
        """
        self.agents = {}  # 存储所有 agent
        self.edges = defaultdict(dict)  # 存储边和权重: {from_id: {to_id: weight}}
        self.task_history = defaultdict(list)  # 存储任务执行历史
        self.edge_success_rate = defaultdict(lambda: defaultdict(lambda: 1.0))  # 边的成功率统计
        self.topological_order = []  # 存储拓扑排序结果
        self.max_path_length = 5  # 最大路径长度限制
        self.neighbors = defaultdict(set)  # Store neighbors for each agent

    def add_agent(self, agent) -> None:
        """
        Add an agent to the DAG.
        """
        self.agents[agent.agent_id] = agent
        self._update_topological_order()

    def add_neighbor(self, from_agent_id: int, to_agent: 'Agent') -> None:
        """
        Add a neighbor to an agent's neighbor set.
        
        Args:
            from_agent_id: ID of the agent to add neighbor to
            to_agent: Agent object to be added as neighbor
        """
        self.neighbors[from_agent_id].add(to_agent.agent_id)

    def add_edge(self, from_agent_id: int, to_agent_id: int, initial_weight: float = 1.0) -> bool:
        """
        Add a weighted edge between two agents with cycle detection.
        
        Args:
            from_agent_id: Source agent ID
            to_agent_id: Target agent ID
            initial_weight: Initial edge weight (default: 1.0)
            
        Returns:
            bool: True if edge was added successfully, False if it would create a cycle
        """
        if from_agent_id not in self.agents or to_agent_id not in self.agents:
            raise ValueError("One or both agents do not exist in the DAG")

        # 检查是否会形成环
        if self._would_create_cycle(from_agent_id, to_agent_id):
            print(f"Warning: Edge from {from_agent_id} to {to_agent_id} would create a cycle. Skipping.")
            return False

        self.edges[from_agent_id][to_agent_id] = initial_weight
        self.add_neighbor(from_agent_id, self.agents[to_agent_id])
        self._update_topological_order()
        return True
    
    def _would_create_cycle(self, from_agent_id: int, to_agent_id: int) -> bool:
        """
        Check if adding an edge would create a cycle in the DAG.
        """
        if from_agent_id == to_agent_id:
            return True

        visited = set()
        def dfs(current_id: int) -> bool:
            if current_id == from_agent_id:
                return True
            visited.add(current_id)
            for next_id in self.edges[current_id]:
                if next_id not in visited and dfs(next_id):
                    return True
            visited.remove(current_id)
            return False

        return dfs(to_agent_id)

    def _update_topological_order(self) -> None:
        """
        Update the topological ordering of the DAG.
        """
        in_degree = defaultdict(int)
        for from_id in self.edges:
            for to_id in self.edges[from_id]:
                in_degree[to_id] += 1

        # 使用优先队列来确保稳定的拓扑排序
        queue = [(0, agent_id) for agent_id in self.agents if in_degree[agent_id] == 0]
        heapq.heapify(queue)
        
        self.topological_order = []
        while queue:
            _, agent_id = heapq.heappop(queue)
            self.topological_order.append(agent_id)
            
            for to_id in self.edges[agent_id]:
                in_degree[to_id] -= 1
                if in_degree[to_id] == 0:
                    heapq.heappush(queue, (to_id, to_id))

    def update_edge_weight(self, from_agent_id: int, to_agent_id: int, 
                         success: bool, execution_time: float) -> None:
        """
        Update edge weight based on task execution results.
        
        Args:
            from_agent_id: Source agent ID
            to_agent_id: Target agent ID
            success: Whether the task was successful
            execution_time: Time taken to execute the task
        """
        if from_agent_id in self.edges and to_agent_id in self.edges[from_agent_id]:
            # 更新成功率
            history = self.edge_success_rate[from_agent_id][to_agent_id]
            self.edge_success_rate[from_agent_id][to_agent_id] = history * 0.9 + (1.0 if success else 0.0) * 0.1

            # 更新边权重
            current_weight = self.edges[from_agent_id][to_agent_id]
            success_factor = 1.1 if success else 0.9
            time_factor = min(1.0, 1.0 / execution_time) if execution_time > 0 else 1.0
            new_weight = current_weight * success_factor * time_factor
            self.edges[from_agent_id][to_agent_id] = max(0.1, min(2.0, new_weight))

    def find_optimal_path(self, start_agent_id: int, task) -> List[int]:
        """
        Find the optimal path for task execution using weighted shortest path.
        
        Args:
            start_agent_id: Starting agent ID
            task: Task to be executed
            
        Returns:
            List of agent IDs representing the optimal path
        """
        distances = {agent_id: float('inf') for agent_id in self.agents}
        distances[start_agent_id] = 0
        predecessors = {agent_id: None for agent_id in self.agents}
        
        # 使用 Dijkstra 算法找最优路径
        pq = [(0, start_agent_id)]
        visited = set()
        
        while pq:
            current_distance, current_id = heapq.heappop(pq)
            
            if current_id in visited:
                continue
                
            visited.add(current_id)
            
            for next_id in self.edges[current_id]:
                edge_weight = self.edges[current_id][next_id]
                success_rate = self.edge_success_rate[current_id][next_id]
                agent_ability = self.agents[next_id].abilities.get(task.task_type, 0)
                
                # 综合考虑边权重、成功率和agent能力
                weight = edge_weight * (2 - success_rate) / (agent_ability + 0.1)
                distance = current_distance + weight
                
                if distance < distances[next_id]:
                    distances[next_id] = distance
                    predecessors[next_id] = current_id
                    heapq.heappush(pq, (distance, next_id))
        
        # 构建最优路径
        best_target = min(distances.keys(), key=lambda x: distances[x])
        path = []
        current = best_target
        
        while current is not None:
            path.append(current)
            current = predecessors[current]
            
        return list(reversed(path))

    def route_task(self, task, max_attempts: int = 3) -> Optional[str]:
        """
        Enhanced task routing with optimal path finding and load balancing.
        
        Args:
            task: Task to be routed
            max_attempts: Maximum number of routing attempts
            
        Returns:
            Optional[str]: Task execution result if successful, None otherwise
        """
        # 选择起始节点：优先选择空闲且能力匹配的agent
        available_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.current_load < 3 and agent.abilities.get(task.task_type, 0) > 0.5
        ]
        
        start_agent_id = random.choice(available_agents) if available_agents else \
                        random.choice(list(self.agents.keys()))
        
        # 找到最优执行路径
        optimal_path = self.find_optimal_path(start_agent_id, task)
        attempts = 0
        
        while attempts < max_attempts and optimal_path:
            current_agent_id = optimal_path.pop(0)
            current_agent = self.agents[current_agent_id]
            
            print(f"Attempting task {task.task_id} with Agent {current_agent_id}")
            
            if current_agent.decide_to_process(task):
                response, success = current_agent.process_task(task)
                
                # 更新路径权重
                if len(optimal_path) > 0:
                    next_agent_id = optimal_path[0]
                    self.update_edge_weight(
                        current_agent_id, 
                        next_agent_id,
                        success,
                        current_agent.current_load
                    )
                
                if success:
                    self.task_history[task.task_type].append({
                        'path': [current_agent_id],
                        'success': True,
                        'response': response
                    })
                    return response
                
            attempts += 1
            
            # 如果当前路径失败，重新计算路径
            if attempts < max_attempts and optimal_path:
                optimal_path = self.find_optimal_path(optimal_path[0], task)
        
        # 如果所有尝试都失败，选择负载最小的agent强制执行
        least_busy_agent = min(self.agents.values(), key=lambda a: a.current_load)
        response, success = least_busy_agent.process_task(task)
        
        self.task_history[task.task_type].append({
            'path': [least_busy_agent.agent_id],
            'success': success,
            'response': response if success else None
        })
        
        return response if success else None

    def get_agent_statistics(self) -> Dict:
        """
        Get statistics about agents and their performance.
        """
        stats = {
            'agent_loads': {agent_id: agent.current_load for agent_id, agent in self.agents.items()},
            'edge_weights': dict(self.edges),
            'success_rates': dict(self.edge_success_rate),
            'topological_order': self.topological_order
        }
        return stats