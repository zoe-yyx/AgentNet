'''
此文件包括了AgentGraph类
AgentGraph类，是我们的核心，是一个由不同的Agent组成的图，这个图目前仅仅保留拓扑信息和节点信息，对于任务调用和转发等，我们
都通过Experiment类和Agent类去进行，Experiment类负责控制，Agent类负责执行
'''

import random
import networkx as nx
import copy
import logging

from .agent import Agent
from config.setting import task_to_ability_map


class AgentGraph:
    def __init__(self, agent_config, agent_graph_config):
        # 很符合常理的思维，先初始化Agent，再初始化图，先有Agent再有图，我们还是把Agents放进了AgentGraph中了      
        self.agent_config = agent_config
        self.agent_graph_config = agent_graph_config
        self.edge_weight = {}
        self.edge_success_rate = {}

        self.agents = self.initilize_agents(agent_config, agent_graph_config["global_router_experience"])
        self.agent_neighbor_dict = self.initilize_graph(agent_graph_config)


    def initilize_agents(self, agent_config, global_router_experience_flag):
        # 初始化每一个Agent
        agents = {}
        for single_agent_config in agent_config:
            single_agent_config_copy = copy.deepcopy(single_agent_config)
            agents[single_agent_config_copy["id"]] = Agent(single_agent_config_copy, global_router_experience_flag)

        for agent_id, agent in agents.items():
            agent.set_collect_neighbors_info(self.collect_neighbors_info)

        # 初始化边权重
        for source_agent_id, agent in agents.items():
            if source_agent_id not in self.edge_weight.keys():
                self.edge_weight[source_agent_id] = {}
                self.edge_success_rate[source_agent_id] = {}
            for target_agent_id, agent in agents.items():
                if source_agent_id != target_agent_id:
                    self.edge_weight[source_agent_id][target_agent_id] = 1.0
                    self.edge_success_rate[source_agent_id][target_agent_id] = 0.0

        return agents


    def initilize_graph(self, agent_graph_config):
        agent_neighbor_dict = {

        }
        if agent_graph_config["graph_type"] == "complete":
            # 如果是完全图，那么我们就需要将所有的Agent的Incoming和Outcoming领居都设置为其他的所有领居
            for source_agent_id, source_agent in self.agents.items():
                if source_agent_id not in agent_neighbor_dict.keys():
                    agent_neighbor_dict[source_agent_id] = {
                        "incoming_agent_id": [],
                        "outcoming_agent_id": [],
                    }
                for target_agent_id, target_agent in self.agents.items():
                    if source_agent_id == target_agent_id:
                        continue
                    else:
                        agent_neighbor_dict[source_agent_id]["incoming_agent_id"].append(target_agent_id)
                        agent_neighbor_dict[source_agent_id]["outcoming_agent_id"].append(target_agent_id)
        
        return agent_neighbor_dict

    def sample_an_agent(self):
        # 该任务返回一个随机的agent id，具体随机方式我们暂时先定为完全的随机
        agent_id = random.choice(list(self.agents.keys())) 
        return agent_id


    def select_an_agent(self, task_type):
        outcoming_agents_id = self.agents.keys()      
        logging.info(f"outcoming_agents_id = {outcoming_agents_id}") 
        neighbors_info = {}
        
        # 遍历所有邻居，选择最合适的 agent
        for agent_id in outcoming_agents_id:
            agent = self.agents[agent_id]
            
            agent_info = agent.get_self_info()  
            success_rate = agent_info["success_rate"]
            abilities = agent_info["abilities"]

            if task_type not in success_rate:
                continue
        
            ability_names = task_to_ability_map[task_type]
            total_value, ability_num = 0, 0
            for name in ability_names:
                ability_num += 1
                total_value += abilities[name]
            average_ability_value = total_value / ability_num

            neighbors_info[agent_id] = {
                "agent_info": agent_info,
                "average_ability_value": average_ability_value
            }
            logging.info(f"neighbors_info[agent_id]: {agent_id}, agent_info: {agent_info}, average_ability_value: {average_ability_value}")

        if not neighbors_info:
            logging.info(f"没有找到合适的 agent 来执行任务 {task_type}, 随机选择一个 agent")
            return self.sample_an_agent()
        
        # 获取所有最大的 average_ability_value
        max_value = max(neighbors_info[agent_id]["average_ability_value"] for agent_id in neighbors_info)
        
        # 过滤出拥有最大值的所有 agent
        best_agents = [agent_id for agent_id, info in neighbors_info.items() if info["average_ability_value"] == max_value]
        
        # 如果有多个 agent，随机选择一个
        best_agent_id = random.choice(best_agents)

        logging.info(f"选定 agent {best_agent_id} 来执行任务 {task_type}，平均能力值: {neighbors_info[best_agent_id]['average_ability_value']}")
        
        return best_agent_id



    def collect_neighbors_info(self, agent_id, task):
        outcoming_neighbors_id  = self.agent_neighbor_dict[agent_id]["outcoming_agent_id"]
        neighbors_info = {}
        for neighbor_id in outcoming_neighbors_id:
            if self.edge_weight[agent_id][neighbor_id] <= 0.3:
                # 如果边权重过低，则舍去这个边，在传递Agent周围信息的时候不再接受
                continue
            neighbor_agent = self.agents[neighbor_id]
            neighbor_agent_info = neighbor_agent.get_self_info()

            neighbor_agent_info["processed_tasks"] = neighbor_agent.get_relevant_experence(task)
            neighbor_agent_info["success_rate"] = neighbor_agent_info["success_rate"]
            neighbor_agent_info["task_type_success_rate"] = neighbor_agent_info["success_rate"][task.task_type]
            neighbor_agent_info["is_incoming"] = False
            neighbor_agent_info["is_outgoing"] = True
            neighbors_info[neighbor_id]= neighbor_agent_info
        return neighbors_info


    def update_edge_weight(self, source_agent_id, target_agent_id, execution_time, success):
        
        current_rate = self.edge_success_rate[source_agent_id].get(target_agent_id, 0.6)
        self.edge_success_rate[source_agent_id][target_agent_id] = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
            
        # 更新边权重
        current_weight = self.edge_weight[source_agent_id].get(target_agent_id, 1.0)
        success_factor = 1.1 if success else 0.9
        time_factor = min(1.0, 1.0 / (execution_time * 0.1)) if execution_time > 0 else 1.0

        logging.info(f"更新边权重: {source_agent_id} -> {target_agent_id}, 成功率: {current_rate}, 权重: {current_weight}, 成功因子: {success_factor}, 时间因子: {time_factor}")

        new_weight = current_weight * success_factor * time_factor
        
        # 确保权重在合理范围内
        self.edge_weight[source_agent_id][target_agent_id] = max(0.1, min(2.0, new_weight))
        logging.info(f"更新边权重: {source_agent_id} -> {target_agent_id}, 权重: {self.edge_weight[source_agent_id][target_agent_id]}")
        logging.info(f"邻居列表: {self.agent_neighbor_dict[source_agent_id]['outcoming_agent_id']}")
        if self.edge_weight[source_agent_id][target_agent_id] <= 0.3 and self.agent_neighbor_dict[source_agent_id]["outcoming_agent_id"].count(target_agent_id) > 0:
            # 权重过小直接删除
            self.agent_neighbor_dict[source_agent_id]["outcoming_agent_id"].remove(target_agent_id)

    def get_all_agent_info(self):
        all_agent_info = { }
        for agent_id, agent in self.agents.items():
            all_agent_info[agent_id] = agent.get_self_info()
        return all_agent_info


    def get_all_agent_experiences(self):
        all_agent_experiences = { }
        for agent_id, agent in self.agents.items():
            all_agent_experiences[agent_id] = agent.get_all_experiences()
        return all_agent_experiences
