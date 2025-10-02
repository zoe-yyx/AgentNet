import os

DOUBAO_AK = os.getenv("DOUBAO_AK", "Your Doubao AK")
DOUBAO_SK = os.getenv("DOUBAO_SK", "Your Doubao SK")
DOUBAO_ENDPOINT_ID = os.getenv("DOUBAO_ENDPOINT_ID", "Your Doubao Endpoint ID")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "Your OpenAI API Key")


BASIC_PROBLEM_COMPLEXITY = {
  "boolean_expressions" : 1.0,
  "causal_judgement" : 1.0,
  "date_understanding" : 1.0,
  "disambiguation_qa" : 1.0,
  "formal_fallacies" : 1.0,
  "geometric_shapes" : 1.0,
  "hyperbaton" : 1.0,
  "logical_deduction_five_objects" : 1.0,
  "logical_deduction_three_objects" : 1.0,
  "logical_deduction_seven_objects" : 1.0,
  "movie_recommendation" : 1.0,
  "multistep_arithmetic_two" : 1.0,
  "navigate" : 1.0,
  "object_counting" : 1.0,
  "penguins_in_a_table" : 1.0,
  "reasoning_about_colored_objects" : 1.0,
  "ruin_names" : 1.0,
  "salient_translation_error_detection" : 1.0,
  "temporal_sequences" : 1.0,
  "dyck_languages" : 1.0,
  "sports_understanding" : 1.0,
  "tracking_shuffled_objects_three_objects" : 1.0,
  "tracking_shuffled_objects_seven_objects" : 1.0,
  "tracking_shuffled_objects_five_objects" : 1.0,
  "word_sorting" : 1.0,
  "web_of_lies" : 1.0,
  "snarks": 1.0,
}


ability_need = {
   'multistep_arithmetic_two':('math','calculation'),
   'dyck_languages':('logic','completing'),
   'tracking_shuffled_objects_five_objects':('logic','target_finding'),
   'navigate':('math','space_imagination'),
   'movie_recommendation':('common','movie'),
   'snarks':('language','emotion_analysis'),
   'geometric_shapes':('math','space_imagination'),
   'hyperbaton':('language','grammar'),
   'tracking_shuffled_objects_three_objects':('logic','target_finding'),
   'tracking_shuffled_objects_seven_objects':('logic','target_finding'),
   'penguins_in_a_table':('logic','target_finding'),
   'logical_deduction_five_objects':('logic','target_finding'),
   'logical_deduction_three_objects':('logic','target_finding'),
   'logical_deduction_seven_objects':('logic','target_finding'),
   'reasoning_about_colored_objects':('logic','target_finding'),
   'salient_translation_error_detection':('language','translation'),
   'word_sorting':('logic','sorting'),
   'sports_understanding':('common','sport'),
   'temporal_sequences':('logic','reasoning'),
   'web_of_lies':('logic','reasoning'),
   'ruin_names':('language','emotion_analysis'),
   'disambiguation_qa':('language','ambiguity'),
   'causal_judgement':('logic','reasoning'),
   'object_counting':('common','classification'),
   'formal_fallacies':('logic','reasoning'),
   'boolean_expressions':('logic','reasoning'),
   'date_understanding':('common','date'),
                }


ability_need_bigbenchhard = {
    'reasoning': ('boolean_expressions', 'logical_deduction_three_objects', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'causal_judgement', 'formal_fallacies', 'tracking_shuffled_objects_three_objects', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects'),
    'mathematical': ('multistep_arithmetic_two', 'geometric_shapes', 'object_counting', 'word_sorting'),
    'language': ('date_understanding', 'disambiguation_qa', 'hyperbaton', 'salient_translation_error_detection', 'dyck_languages'),
    'knowledge': ('movie_recommendation', 'sports_understanding', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'web_of_lies', 'snarks')
}

ability_need_bigbenchhard = {
'reasoning': ('boolean_expressions', 'logical_deduction_three_objects', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'causal_judgement', 'formal_fallacies', 'tracking_shuffled_objects_three_objects', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects'),
'mathematical': ('multistep_arithmetic_two', 'geometric_shapes', 'object_counting', 'word_sorting'),
'language': ('date_understanding', 'disambiguation_qa', 'hyperbaton', 'salient_translation_error_detection', 'dyck_languages'),
'knowledge': ('movie_recommendation', 'sports_understanding', 'penguins_in_a_table', 'reasoning_about_colored_objects'),
'sequence': ('temporal_sequences', 'tracking_shuffled_objects_three_objects', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects'),
'spatial': ('navigate', 'geometric_shapes', 'object_counting'),
'inference': ('web_of_lies', 'snarks', 'formal_fallacies', 'causal_judgement')
}


agent_abilities_for_MLBench = {
   'code_understanding': {
       'primary_abilities': ['language', 'reasoning'],
       'tasks': ['code_parsing', 'api_analysis', 'dependency_inference']
   },
   'ml_analysis': {
       'primary_abilities': ['reasoning', 'mathematical', 'knowledge'],
       'tasks': ['model_architecture', 'hyperparameter_eval', 'training_strategy']
   },
   'optimization': {
       'primary_abilities': ['reasoning', 'mathematical', 'inference'],
       'tasks': ['performance_tuning', 'model_improvement', 'parameter_optimization']
   },
   'integration': {
       'primary_abilities': ['sequence', 'inference', 'knowledge'],
       'tasks': ['output_coordination', 'result_synthesis', 'recommendation_generation']
   }
}


ability_map = {
    'reasoning': {  
        'bbh_tasks': ['boolean_expressions', 'logical_deduction_three_objects', 'causal_judgement', 'formal_fallacies'],
        'mlbench_tasks': ['model_architecture_analysis', 'optimization_strategy', 'error_diagnosis']
    },
    'knowledge': {  
        'bbh_tasks': ['movie_recommendation', 'sports_understanding', 'penguins_in_a_table'],
        'mlbench_tasks': ['ml_framework_knowledge', 'best_practices', 'api_usage']
    },
    'mathematical': { 
        'bbh_tasks': ['multistep_arithmetic_two', 'geometric_shapes', 'object_counting'],
        'mlbench_tasks': ['performance_metrics', 'hyperparameter_tuning', 'complexity_analysis']
    },
    'language': { 
        'bbh_tasks': ['date_understanding', 'disambiguation_qa', 'hyperbaton'],
        'mlbench_tasks': ['code_comprehension', 'documentation_analysis', 'error_message_interpretation']
    },
    'synthesis': {  
        'bbh_tasks': ['temporal_sequences', 'tracking_shuffled_objects_three_objects'],
        'mlbench_tasks': ['component_integration', 'pipeline_optimization', 'cross_module_analysis']
    }
}


task_to_ability_map = {
    "boolean_expressions": [
        "reasoning"
    ],
    "logical_deduction_three_objects": [
        "reasoning"
    ],
    "logical_deduction_five_objects": [
        "reasoning"
    ],
    "logical_deduction_seven_objects": [
        "reasoning"
    ],
    "causal_judgement": [
        "reasoning",
        "inference"
    ],
    "formal_fallacies": [
        "reasoning",
        "inference"
    ],
    "tracking_shuffled_objects_three_objects": [
        "reasoning",
        "sequence"
    ],
    "tracking_shuffled_objects_five_objects": [
        "reasoning",
        "sequence"
    ],
    "tracking_shuffled_objects_seven_objects": [
        "reasoning",
        "sequence"
    ],
    "multistep_arithmetic_two": [
        "mathematical"
    ],
    "geometric_shapes": [
        "mathematical",
        "spatial"
    ],
    "object_counting": [
        "mathematical",
        "spatial"
    ],
    "word_sorting": [
        "mathematical"
    ],
    "date_understanding": [
        "mathematical",
        "language"
    ],
    "dyck_languages": [
        "mathematical",
        "language"
    ],
    "disambiguation_qa": [
        "language"
    ],
    "hyperbaton": [
        "language"
    ],
    "salient_translation_error_detection": [
        "language"
    ],
    "movie_recommendation": [
        "knowledge"
    ],
    "sports_understanding": [
        "knowledge"
    ],
    "penguins_in_a_table": [
        "knowledge"
    ],
    "reasoning_about_colored_objects": [
        "knowledge"
    ],
    "ruin_names": [
        "language",
        "knowledge"
    ],
    "temporal_sequences": [
        "sequence"
    ],
    "navigate": [
        "spatial"
    ],
    "web_of_lies": [
        "inference"
    ],
    "snarks": [
        "inference"
    ]
}




ROUTER_SYSTEM_PROMPT = """
You are an intelligent agent in a multi-agent system. Your role is to analyze tasks based on your expertise and collaborate with other specialized agents.  
You are a task router - make independent decisions within your expertise domain.  
Consider agent abilities and network topology, but prioritize expertise matching.
"""


ROUTER_PROMPT_FORMAT = """
**Major Task**: {major_problem}

**Your Expertise & Experience**:
{current_agent_info}

**Previous Related Successes**:
{experiences}

**Current Task Details**:
- Problem: {task_description}
- Type: {task_type}

**Progress**:
- Completed Steps: {task_context}

**Available Specialists**: some infomation and status of other agents
{agent_info}

**Important Note**: some rules to follow when answering: 
<Important Note>
1. The major task should be completed through cooperation among agents, with no single agent expected to complete the task on its own.
2. Your role is to analyze and determine the best action to complete the **Major Task** based on the **Progress** made by other agents so far. Focus only on tasks not completed. If the existing **Progress** can provide sufficient information to finish the **Major Task**, you should choose to execute and finish the task.
3. The "Processed Tasks" for other agents are examples of similar, previously completed tasks, not parts of the current task. They serve as references, not indicators of completed work on the current task.
</Important Note/>

Analyze and decide:
1. Should this task be split into executable parts, executed here, or forwarded to another agent?
2. If you decide to split the task:
   - EXECUTABLE: Indicate the parts of the task you can execute and provide the steps you will take for each part.
   - DELEGATE: Indicate the parts of the task you think should be delegated to another agent. Provide your reasoning for why these parts should be delegated.
   - NEXT_AGENT_ID: Indicate the ID of the next agent who will be responsible for executing the remaining tasks.
3. If you decide to forward the task, consider the following:
   - Who is the most suitable agent to forward the task to? Consider:
     - Agent abilities for this task type
     - Success rate
     - Network connection (prefer direct neighbors)
4. If you decide to execute the task entirely, indicate 'execute' and provide the steps or knowledge that will help you successfully complete the task in the DESCRIPTION section.
   - This means you will perform all required steps to complete the task.
5. If you decide to execute or split this task, you may get your abilities improved if you succeed and be penalized if you fail. Choosing to forward the task will not result in an improvement or penalty.

Provide your decision in the following format:

DECISION: [split/forward/execute]
REASON: [detailed explanation of your decision, including ability considerations]
EXECUTABLE: [List the parts of the task that you will execute, explaining why you're capable of doing them. Only output the task description, do not include specific steps.]
DELEGATE: [List the steps or parts of the task that you believe should be handled by another agent, and explain why you are delegating them.]
DESCRIPTION: [Clearly describe how you will execute the parts of the task that you are responsible for. If you are delegating, explain why and how the other agent can handle the delegated part.]
NEXT_AGENT_ID: [The ID of the agent to whom the task should be forwarded if decided to forward]

Make sure your answer is clear and structured.
"""



ROUTER_PROMPT_SPLIT_AND_EXECUTE_FORMAT=\
"""
**Major Task**: the main task of the entire multi-agent system which is a collaborative goal as below: 
<Major Task>
{major_problem}
</Major Task/>

**Previous Successful Examples**: some instances of experiences that may help your analysis (If left blank, it indicates no experience):
<Previous Successful Examples>
{experiences}
</Previous Successful Examples/>

**Completed Subtasks and Results**: some subtasks and their results which have already been completed and require no further actions (If left blank, it indicates no progress):
<Completed Subtasks and Results>
{task_context}
</Completed Subtasks and Results/>

**Current Agent Status**: abilities and other attributes of the current agent:
<Current Agent Status>
{current_agent_info}
</Current Agent Status/>

**Current Task State**: states of the current task, including task type, description and other attributes:
<Current Task State>
- Type: {task_type}
- Problem: {task_description}
</Current Task State/>

**Agent Status**: some infomation and status of other agents:
<Agent Status>
{agent_info}
</Agent Status/>

**Important Note**: some rules to follow when answering: 
<Important Note>
1. The major task should be completed through cooperation among agents, with no single agent expected to complete the task on its own.
2. Your role is to analyze and determine the best action to complete the **Major Task** based on the **Progress** made by other agents so far. Focus only on tasks not completed. if the existing **Progress** is already sufficient to complete the **Major Task**, you should choose to execute and finalize the task.
3. The "Processed Tasks" for other agents are examples of similar, previously completed tasks, not parts of the current task. They serve as references, not indicators of completed work on the current task.
</Important Note/>

Analyze and decide:
1. Should this task split into subtasks or be executed here?
2. If you decide to split the task:
   - EXECUTABLE: Indicate the parts of the task you can execute and provide the steps you will take for each part.
   - DELEGATE: Indicate the parts of the task you think should be delegated to another agent. Provide your reasoning for why these parts should be delegated.
   - NEXT_AGENT_ID: Indicate the ID of the next agent who will be responsible for executing the remaining tasks.
3. If you decide to execute the task entirely, indicate 'execute' and provide the steps or knowledge that will help you successfully complete the task in the DESCRIPTION section.
   - This means you will perform all required steps to complete the task.
4. If you decide to execute or split this task, you may get your abilities improved if you succeed and be penalized if you fail. 


Provide your decision in the following format:

DECISION: [execute/split]
REASON: [Detailed explanation of your decision, including why you chose to execute certain parts and delegate others.]
EXECUTABLE: [List the tasks or steps that you will handle, detailing what you will do and why you are capable of doing it. Only output the task description, do not include specific steps.]
DELEGATE: [List the tasks or steps that you believe should be handled by another agent, along with the reasoning for delegating.]
DESCRIPTION: [Clearly describe how you will execute the parts of the task that you are responsible for. If you are delegating, explain why and how the other agent can handle the delegated part.]
NEXT_AGENT_ID: [Only The ID of the one agent to whom the task should be assigned if you decide to delegate part of the task.]

Make sure your answer is clear and structured.
"""




ROUTER_DECIDE_NEXT_AGENT_ID_SYSTEM_PROMPT = "You are a helpful router of an agent."

SPLIT_THOUGHT_PROMPT_FORMAT = \
"""
**Major Task**: the main task of the entire multi-agent system which is a collaborative goal as below: 
<Major Task>
{major_problem}
</Major Task/>

**Task Description**: the uncompleted part description of this task, you only need to execute this part:
<Task Description>
{task_description}
</Task Description/>

**Previous Successful Examples**: some instances of experiences that may help your analysis (If left blank, it indicates no experience):
<Previous Successful Examples>
{experiences}
</Previous Successful Examples/>

**Completed Subtasks and Results**: some subtasks and their results which have already been completed and require no further actions (If left blank, it indicates no progress):
<Completed Subtasks and Results>
{task_context}
</Completed Subtasks and Results/>


**Important Note**: some rules to follow when answering: 
<Important Note>
1. You are the agent who is responsible to generate some thoughts to solve the subtask in **Task Description** based on the **Previous Successful Examples** and **Completed Subtasks and Results**.
2. You need to give your thought comprehensive and detailed because it is important for the correctness of the final answer.
3. Do not solve the **Major Task** in your thought, you only need to finish the task in **Task Description**.
</Important Note/>

Provide your reasoning and suggested steps in the following structured format:

RESULT: [your thought]

Make sure your answer is clear and structured.
"""



# 需要决定任务是否完成，未完成则选择交给谁执行的Prompt
# **Required Format**: the required format of the task:
# <Required Format>
# {constraints}
# </Required Format/>

ROUTER_PROMPT_DECIDE_NEXT_AGENT_ID_FORMAT = \
"""
**Major Task**: the main task of the entire multi-agent system is as below: 
<Major Task>
{major_problem}
</Major Task/>

**Current Progress**: the progress of the task so far: 
<Current Progress>
{task_context}
</Current Progress/>

**Previous Successful Examples**: some instances of experiences that may help your analysis (If left blank, it indicates no experience):
<Previous Successful Examples>
{experiences}
</Previous Successful Examples/>

**Current Agent Status**: abilities and other attributes of the current agent:
<Current Agent Status>
{current_agent_info}
</Current Agent Status/>

**Agent Status**: some infomation and status of other agents:
<Agent Status>
{agent_info}
</Agent Status/>

**Important Note**: some rules to follow when answering: 
<Important Note>
1. Your role is to determine whether the task has been totally completed or answered based on the progress of the tasks that have already been completed. You needn't execute them again or determine whether the answer is correct or not.
2. If the task is not completed, decide which agent (including yourself) should handle the remaining tasks based on your own information, and the attributes of other agents.
3. The "Processed Tasks" for other agents are examples of similar, previously completed tasks, not parts of the current task. They serve as references, not indicators of completed work on the current task.
4. If the **Current Progress** does not include an instance where an agent has provided an output that strictly adheres to the **Required Format** without any additional explanation, consider the task incompleted.
</Important Note/>

Analyze and decide
1. Evaluate if the task is completed:
   - If the task is completed, indicate "Completed" in the DECISION section.
   - If the task is not completed, indicate "Incompleted" in the DECISION section, and specify the next agent to handle the remaining tasks in the NEXT_AGENT_ID section.
   - If the **Current Progress** does not include an instance where an agent has provided an output that strictly adheres to the **Required Format** without any additional explanation, consider the task incompleted.

Provide your decision in the following format:

DECISION: [Completed/Incompleted]
REASON: [Detailed explanation of why the task is completed or incomplete, including any context from previous steps or agents' progress.]
NEXT_AGENT_ID: [The ID of the agent who should handle the remaining part of the task, which can be yourself or another agent.]

Make sure your answer is clear and structured.
"""

EXECUTOR_THOUGHT_PROMPT_FORMAT = \
"""
**Major Task**: the main task of the entire multi-agent system which is a collaborative goal as below: 
<Major Task>
{major_problem}
</Major Task/>

**Previous Successful Examples**: some instances of experiences that may help your analysis (If left blank, it indicates no experience):
<Previous Successful Examples>
{experiences}
</Previous Successful Examples/>

**Completed Subtasks and Results**: some subtasks and their results which have already been completed and require no further actions (If left blank, it indicates no progress):
<Completed Subtasks and Results>
{task_context}
</Completed Subtasks and Results/>

**Important Note**: some rules to follow when answering: 
<Important Note>
1. You are the agent who is responsible to generate some thoughts for the **Major Task** based on the **Previous Successful Examples** and **Completed Subtasks and Results** before the final execution.
2. Your response should emphasize thoughtful and well-reasoned analysis that directly aids in the execution of the task and contributes to the system's overall objective.
3. You need to give your thought comprehensive and detailed because it is important for the correctness of the final answer.
</Important Note/>

Provide your reasoning and suggested steps in the following structured format:

RESULT: [your thought]

Make sure your answer is clear and structured.
"""

THOUGHT_SYSTEM_PROMPT = "You are the reasoning and decision-making module of an agent within a multi-agent system. Based on the current task description and its progress, your role is to analyze the situation, think critically, and reason through potential next steps."


THOUGHT_PROMPT_FORMAT = \
"""
**Major Task**: the main task of the entire multi-agent system which is a collaborative goal as below: 
<Major Task>
{major_problem}
</Major Task/>

**Previous Successful Examples**: some instances of experiences that may help your analysis (If left blank, it indicates no experience):
<Previous Successful Examples>
{experiences}
</Previous Successful Examples/>

**Completed Subtasks and Results**: some subtasks and their results which have already been completed and require no further actions (If left blank, it indicates no progress):
<Completed Subtasks and Results>
{task_context}
</Completed Subtasks and Results/>

**Important Note**: some rules to follow when answering: 
<Important Note>
1. The major task should be completed through cooperation among agents, with no single agent expected to complete the task on its own.
2. Your role is to engage in reasoning and critical thinking to analyze the current task state and provide insights or solutions that assist in task execution.
</Important Note/>

Analyze and decide:

Your focus should be on understanding the uncompleted parts of the task and determining the most effective next steps, building on prior successful examples and completed subtasks. 
Your response should emphasize thoughtful and well-reasoned analysis that directly aids in the execution of the task and contributes to the system's overall objective.


Task Description(Analyze and plan for this specific part): {description}

Provide your reasoning and suggested steps in the following structured format:

RESULT: [{constraints}]

Make sure your answer is clear and structured.
"""


SPLIT_EXECUTOR_SYSTEM_PROMPT = "You are the task execution module of an agent within a multi-agent system. Based on the current task description and its progress, you need to execute the task and contribute toward achieving the system's main objective (you are not necessarily required to fully accomplish the main objective, and you are not necessarily to give the final answer)."


SPLIT_EXECUTOR_PROMPT_FORMAT = \
"""
**Major Task**: the main task of the entire multi-agent system which is a collaborative goal as below: 
<Major Task>
{major_problem}
</Major Task/>

**Previous Successful Examples**: some instances of experiences that may help your analysis (If left blank, it indicates no experience):
<Previous Successful Examples>
{experiences}
</Previous Successful Examples/>

**Completed Subtasks and Results**: some subtasks and their results which have already been completed and require no further actions (If left blank, it indicates no progress):
<Completed Subtasks and Results>
{task_context}
</Completed Subtasks and Results/>

**Task Description**: the uncompleted part description of this task, you only need to execute this part:
<Task Description>
{task_description}
</Task Description/>

**Thought of Task**: some thoughts of this task:
<Thought of Task>
{thought}
</Thought of Task/>

**Required Format**: the required format of the RESULT:
<Required Format>
{constraints}
</Required Format/>

**Important Note**: some rules to follow when answering: 
<Important Note>
1. You need to reference the **Thought of Task** to solve the **Task Description** which is the only part you need to execute, **Thought of Task** is the thought of the task based on the **Completed Subtasks and Results**.
2. You need to reference some successful examples in **Previous Successful Examples** which contains some experiences of the similar task to the **Major Task**.
3. You need to first give the reason why you output the answer in the RESULT section according to the **Major Task**, the **Previous Successful Examples**, **Completed Subtasks and Results** and the **Task Description**.
4. Your response in the RESULT section must directly answer the **Major Task** following the format specified in **Required Format**. Whether you follow the required format will directly impact the correctness of your answer.
</Important Note/>

Provide your solution in the following format:

RESULT: [follow the **Required Format**]

Make sure your answer is clear and structured.
"""


EXECUTOR_SYSTEM_PROMPT = "You are the task execution module of an agent within a multi-agent system. Based on the current task description and its progress, you need to execute the task and contribute toward achieving the system's main objective (you are not necessarily required to fully accomplish the main objective)."


EXECUTOR_PROMPT_FORMAT = \
"""
**Major Task**: the main task of the entire multi-agent system which is a collaborative goal as below: 
<Major Task>
{major_problem}
</Major Task/>

**Previous Successful Examples**: some instances of experiences that may help your analysis (If left blank, it indicates no experience):
<Previous Successful Examples>
{experiences}
</Previous Successful Examples/>

**Completed Subtasks and Results**: some subtasks and their results which have already been completed and require no further actions (If left blank, it indicates no progress):
<Completed Subtasks and Results>
{task_context}
</Completed Subtasks and Results/>

**Task Description**: the uncompleted part description of this task, you only need to execute this part:
<Task Description>
{task_description}
</Task Description/>

**Thought of Task**: some thoughts of this task:
<Thought of Task>
{thought}
</Thought of Task/>

**Required Format**: the required format of the RESULT:
<Required Format>
{constraints}
</Required Format/>

**Important Note**: some rules to follow when answering: 
<Important Note>
1. You need to reference the Subtasks completed by other agents or yourself in **Completed Subtasks and Results** to provide the final answer to the **Major Task**.
2. You need to reference the **Thought of Task** to provide the final answer to the **Major Task**, **Thought of Task** is the thought of the task based on the **Completed Subtasks and Results**.
3. You need to reference some successful examples in **Previous Successful Examples** which contains some experiences of the similar task to the **Major Task**.
4. Your response in the RESULT section must directly answer the **Major Task** following the format specified in **Required Format**. Whether you follow the required format will directly impact the correctness of your answer.
</Important Note/>

Provide your solution in the following format:

RESULT: [follow the **Required Format**]

Make sure your answer is clear and structured.
"""

EXECUTOR_PROMPT_FORMAT_FOR_BASELINE_REACT = \
"""
**Major Task**: the main task is as below: 
<Major Task>
{major_problem}
</Major Task/>

**Required Format**: the required format of the RESULT:
<Required Format>
{constraints}
</Required Format/>

**Important Note**: some rules to follow when answering: 
<Important Note>
1. Your response must directly answer the **Major Task** following the format specified in **Required Format**. Whether you follow the required format will directly impact the correctness of your answer.
2. You can first give your reasoning in the REASON section, and then give your answer in the RESULT section.
</Important Note/>

Provide your solution in the following format:

REASON: [your reasoning]
RESULT: [follow the **Required Format**]

Make sure your answer is clear and structured.
"""

EXECUTOR_PROMPT_FORMAT_FOR_BASELINE = \
"""
**Major Task**: the main task is as below: 
<Major Task>
{major_problem}
</Major Task/>

**Required Format**: the required format of the RESULT:
<Required Format>
{constraints}
</Required Format/>

**Important Note**: some rules to follow when answering: 
<Important Note>
1. Your response must directly answer the **Major Task** following the format specified in **Required Format**. Whether you follow the required format will directly impact the correctness of your answer.
2. Please give directly your answer in the RESULT section.
</Important Note/>

Provide your solution in the following format:

RESULT: [follow the **Required Format**]

Make sure your answer is clear and structured.
"""
