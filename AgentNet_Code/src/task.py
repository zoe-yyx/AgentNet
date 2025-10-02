'''
此文件包括了Task类，TaskTree类。

Task类是基本的用来存储task的类，包括task的id, description, 其他基本信息(type等), state(是否完成), 需要的话也需要记录这条task的经历

TaskTree类的目的是可以很好地split任务，当任务被split的时候，TaskTree类就可以据此延伸叶子节点，并且依靠中序遍历来维持目前的task的
运行序列，同时需要保存这个root task (最原始的task)的经历(转发执行等)

'''

import random
from typing import Optional, List
from dataclasses import dataclass, field
import copy
import  time






# 定义复杂度因子，可以根据实际情况设置
complexity_factors = {
    "math_problem": 0.2,
    "code_generation": 0.5,
    "translation": 0.2,
    "summary": 0.3,
    "story": 0.3,
    "combined_task": 0.2
}


def _generate_prompt(task_type: str, task_templates: dict) -> str:
    """根据任务类型生成具体提示"""
    template = random.choice(task_templates[task_type])
    
    if task_type == "math_problem":
        return template.format(
            a=random.randint(1, 100),
            b=random.randint(1, 100),
            c=random.randint(1, 10),
            expr=f"x^{random.randint(2,4)} + {random.randint(1,10)}x + {random.randint(1,100)}"
        )
        
    elif task_type == "code_generation":
        actions = [
            "implement bubble sort",
            "create a linked list",
            "build a binary tree",
            "implement a stack"
        ]
        return template.format(
            action=random.choice(actions),
            feature="data structure",
            algorithm=random.choice(["sorting", "searching", "graph traversal"])
        )
        
    elif task_type == "translation":
        languages = ["English", "Chinese", "Spanish", "French"]
        source = random.choice(languages)
        target = random.choice([l for l in languages if l != source])
        return template.format(
            source=source,
            target=target,
            text="The quick brown fox jumps over the lazy dog."
        )
        
    elif task_type == "summary":
        texts = [
            "An article about artificial intelligence and its impact on society.",
            "A research paper on climate change and global warming.",
            "A news article about recent technological advancements."
        ]
        return template.format(text=random.choice(texts))
        
    elif task_type == "story":
        topics = ["space exploration", "time travel", "magical discovery"]
        characters = ["scientist", "detective", "adventurer"]
        return template.format(
            topic=random.choice(topics),
            character=random.choice(characters)
        )
        
    elif task_type == "combined_task":
        task_types = [
            "math_problem",
            "code_generation",
            "translation",
            "summary",
            "story",
            "combined_task"
        ]
        task1 = _generate_prompt(random.choice(task_types), task_templates)
        task2 = _generate_prompt(random.choice(task_types), task_templates)
        return template.format(task1=task1, task2=task2)
        
    return "Default task prompt"



def generate_random_task(task_id: Optional[int] = None, task_type=None):
        
    """生成随机任务"""
    task_counter = 0
    task_types = [
        "math_problem",
        "code_generation",
        "translation",
        "summary",
        "story",
        "combined_task"
    ]
    
    task_templates = {
        "math_problem": [
            "Calculate: {a} + {b} * {c}",
            "Solve the equation: {a}x + {b} = {c}",
            "Find the derivative of: {expr}"
        ],
        "code_generation": [
            "Write a Python function to {action}",
            "Create a class that implements {feature}",
            "Implement a {algorithm} algorithm"
        ],
        "translation": [
            "Translate from {source} to {target}: {text}",
            "Provide a {target} translation for: {text}"
        ],
        "summary": [
            "Summarize the following text: {text}",
            "Create a brief summary of: {text}"
        ],
        "story": [
            "Write a story about {topic}",
            "Create a narrative involving {character}"
        ],
        "combined_task": [
            "First {task1}, then {task2}",
            "Combine the results of {task1} and {task2}"
        ]
    }

    task_counter += 1
    
    if task_id is not None:
        task_counter = task_id - 1

    if task_type == None:
        task_type = random.choice(task_types)
    
    complexity = complexity_factors[task_type]  # 随便设置的
    prompt = _generate_prompt(task_type, task_templates)

    generated_task = Task(
        task_id=f"{task_id}_0",
        task_type=task_type,
        complexity=complexity,

        major_problem=prompt,
        description=prompt,

    )

    return generated_task


def generate_batch_tasks(count: int, task_type=None):
    """生成一批任务"""
    
    return [generate_random_task(_, task_type) for _ in range(count)]




class Task:
    def __init__(self,
                 task_id,
                 task_type,
                 complexity,
                 priority=1.0,
                 major_problem="",
                 context="",
                 progress_text="",
                 description="",
                 thought="",
                 test_cases=[],
                 result="",
                 state="Incompleted",
                 correct_answer="",
                 ):
        
        # 对于task的id, 我们用str来存，基本格式"xx-xx", 表示的是第几个大任务的第几个小任务
        self.task_id: str = task_id
        self.task_type = task_type                              
        self.complexity = complexity                            
        self.priority = priority

        self.major_problem = major_problem
        self.context = context                                  

        self.progress_text = progress_text 
        self.description = description                          
        self.thought = thought                                  
        
        
        self.result = result 
        self.test_cases = test_cases
        self.state = state                                      
        
        self.forward_history = []                               
        self.correct_answer = correct_answer                    # 正确答案

    def __str__(self):
        return f"Task {self.task_id} is a {self.task_type} problem. Its major problem is '{self.major_problem}'. Its context is {self.context}"

    def set_state(self, state):
        self.state = state

    def set_result(self, result):
        self.result = result

    def set_context(self, context):
        self.context = context

    def set_thought(self, thought):
        self.thought = thought

    def get_context(self):
        return self.context

    def add_forward_history(self, agent_id):
        self.forward_history.append(agent_id)

    def in_forward_history(self, agent_id):
        return agent_id in self.forward_history



class TaskChain:
    def __init__(self, task: Task):
        self.original_task = task                       # 初始化的任务，它的ID应该是str(xx)的形式，也是task_chain的ID
        self.task_chain_id = task.task_id               # 初始化的task_chain的ID，
        self.task_id_count = 1                          # 每次多加一个新任务，就多加一


        self.progress_list = []
        self.task_type = task.task_type                 # 初始任务的任务类型
        self.complexity = task.complexity               # 初始任务的复杂度
        self.priority = task.priority                   # 初始任务的优先级      
        # TODO 这三个可能需要调节，它们的目的只是为了传给下一个但是下一个任务可能不是这样计算的

        self.major_problem = task.major_problem         # 始终谨记主要任务

        self.state = "incompleted"                      # 状态全小写吧
        
        self.final_result = ""                        # 这条任务链的最终回复,

        self.current_task = task                        # 当前的任务
        self.task_history = []                          # 总的任务历史, 包括每一次决策执行和寻找下一个Agent的历史
        self.task_chain = []                            # 一条task的chain，仅仅包括每次的task


    def create_next_task(self):
        self.task_id_count += 1

        # progress = {
        #     "agent_id": agent_id,
        #     "step_id": self.task_id_count,
        #     "description": self.current_task.description,
        #     "result": result
        # }
        # self.progress_list.append(progress)

        context = self.current_task.context + f"\nDescription:\n{self.current_task.description}.\nResult:\n{self.current_task.result}"

        progress_text = ""
        for progress_dict in self.progress_list:
            progress_text += f"Agent {progress_dict['agent_id']} executed following tasks\n"
            progress_text += f"{progress_dict['description']}\n"
            progress_text += f"Agent {progress_dict['agent_id']} output the following answers\n"
            progress_text += f"{progress_dict['result']}\n"

            

        next_task = Task(
            task_id=f"{self.task_chain_id}_{self.task_id_count}",
            task_type=self.task_type,           
            complexity=self.complexity,         
            priority=self.priority,
            major_problem=self.major_problem,
            context=context,      
            progress_text=progress_text,          
            description=""        
        )

        self.task_chain.append(copy.deepcopy(self.current_task))
        self.current_task = next_task



    def set_current_task_description(self, task_description):
        self.current_task.description = task_description
        

    def set_current_task_result(self, agent_id, result):
        progress = {
            "agent_id": agent_id,
            "step_id": self.task_id_count,
            "description": self.current_task.description,
            "result": result
        }
        self.progress_list.append(progress)
        
        self.current_task.result = result
        self.final_result = result


    def add_task_history(self, 
                         format_response, 
                         original_response, 
                         experience,
                         agent_id, 
                         execution_time, 
                         mode):
        
        task_history = TaskHistory(
            task=copy.deepcopy(self.current_task),
            format_response=format_response,
            original_response=original_response,
            experience=experience,
            current_agent_id=agent_id,
            execution_time=execution_time,
            mode=mode
        )
        self.task_history.append(task_history)

    def get_current_state(self):
        return self.state.lower().strip()

    def set_current_state(self, state):
        self.state = state.lower().strip()

    def get_current_task(self):
        return self.current_task

    def get_current_task_id(self):
        return self.current_task.task_id


    def set_final_result(self):
        self.final_result = self.current_task.context 
    
    def get_final_result(self):
        return self.final_result



class TaskHistory:
    def __init__(self, 
                 task,
                 format_response,
                 experience,
                 original_response,
                 current_agent_id,
                 execution_time,
                 mode,
                 ):

        self.task = copy.deepcopy(task)

        self.format_response = format_response
        self.current_agent_id = current_agent_id           
        self.original_response = original_response        
        self.experience = experience
        self.execution_time = execution_time             
        self.timestamp = time.time()             
        self.mode=mode       
