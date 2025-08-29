'''
This file contains the AgentGraph class.
AgentGraph class is our core component, which is a graph composed of different Agents. This graph currently only preserves 
topology information and node information. For task calling and forwarding, we handle them through the Experiment class 
and Agent class, where the Experiment class is responsible for control and the Agent class is responsible for execution.
'''

import random
import networkx as nx
import copy
import logging

from .task import TaskChain
from .agent import Agent
from config.setting import task_to_ability_map


class AgentGraph:
    def __init__(self, agent_config, agent_graph_config):
        # Logical thinking: first initialize Agent, then initialize graph. We put Agents into AgentGraph.      
        self.agent_config = agent_config
        self.agent_graph_config = agent_graph_config
        self.edge_weight = {}
        self.edge_success_rate = {}

        self.agents = self.initilize_agents(agent_config, agent_graph_config["global_router_experience"])
        self.agent_neighbor_dict = self.initilize_graph(agent_graph_config)


    def initilize_agents(self, agent_config, global_router_experience_flag):
        # Initialize each Agent
        agents = {}
        for single_agent_config in agent_config:
            single_agent_config_copy = copy.deepcopy(single_agent_config)
            agents[single_agent_config_copy["id"]] = Agent(single_agent_config_copy, global_router_experience_flag)

        for agent_id, agent in agents.items():
            agent.set_collect_neighbors_info(self.collect_neighbors_info)

        # Initialize edge weights
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
            # If it's a complete graph, we need to set all Agent's incoming and outgoing neighbors to all other neighbors
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
        # This function returns a random agent id, the specific random method is currently set as completely random
        agent_id = random.choice(list(self.agents.keys())) 
        return agent_id


    def select_an_agent(self, task_type):
    # def select_an_agent(self, task_chain: TaskChain):
        # task_type = task_chain.task_type
        outcoming_agents_id = self.agents.keys()    # Get all agents   
        logging.info(f"outcoming_agents_id = {outcoming_agents_id}") 
        neighbors_info = {}
        if not task_type:
            # task_type is None, randomly select an agent to start, otherwise select the most suitable agent
            agent_id = random.choice(list(outcoming_agents_id))
            # Agent decides what to do (only the first time the effect is not good, might need to vary by different cases)
            # self.agents[agent_id].decide_task_type(task_chain)
            return agent_id
        
        # Traverse all agents on the graph
        for agent_id in outcoming_agents_id:
            agent = self.agents[agent_id]
            
            agent_info = agent.get_self_info()  
            success_rate = agent_info["success_rate"]
            abilities = agent_info["abilities"]

            if task_type not in success_rate:
                continue
            # Start modification
            ability_names = task_to_ability_map[task_type]
            # ability_names = task_type   # Can pass in many, list, each is an ability
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
            logging.info(f"No suitable agent found to execute task {task_type}, randomly selecting an agent")
            return self.sample_an_agent()
        
        # Get all maximum average_ability_values
        max_value = max(neighbors_info[agent_id]["average_ability_value"] for agent_id in neighbors_info)
        
        # Filter out all agents with maximum value
        best_agents = [agent_id for agent_id, info in neighbors_info.items() if info["average_ability_value"] == max_value]
        
        # If there are multiple agents, randomly select one
        best_agent_id = random.choice(best_agents)

        logging.info(f"Selected agent {best_agent_id} to execute task {task_type}, average ability value: {neighbors_info[best_agent_id]['average_ability_value']}")
        
        return best_agent_id



    def collect_neighbors_info(self, agent_id, task):
        outcoming_neighbors_id  = self.agent_neighbor_dict[agent_id]["outcoming_agent_id"]
        neighbors_info = {}
        for neighbor_id in outcoming_neighbors_id:
            if self.edge_weight[agent_id][neighbor_id] <= 0.3:
                # If edge weight is too low, discard this edge and no longer accept it when passing Agent neighbor information
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
            
        # Update edge weights
        current_weight = self.edge_weight[source_agent_id].get(target_agent_id, 1.0)
        success_factor = 1.1 if success else 0.97  # 0.9
        time_factor = min(1.0, 1.0 / (execution_time * 0.1)) if execution_time > 0 else 1.0

        logging.info(f"Update edge weight: {source_agent_id} -> {target_agent_id}, success rate: {current_rate}, weight: {current_weight}, success factor: {success_factor}, time factor: {time_factor}")

        new_weight = current_weight * success_factor * time_factor
        
        # Ensure weights are within reasonable range
        self.edge_weight[source_agent_id][target_agent_id] = max(0.1, min(2.0, new_weight))
        logging.info(f"Update edge weight: {source_agent_id} -> {target_agent_id}, weight: {self.edge_weight[source_agent_id][target_agent_id]}")
        logging.info(f"Neighbor list: {self.agent_neighbor_dict[source_agent_id]['outcoming_agent_id']}")
        if self.edge_weight[source_agent_id][target_agent_id] <= 0.3 and self.agent_neighbor_dict[source_agent_id]["outcoming_agent_id"].count(target_agent_id) > 0:
            # Delete directly if weight is too small
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
