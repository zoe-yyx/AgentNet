import random

class Task:
    def __init__(self, task_id, prompt, task_type, priority=1):
        """Initialize a task with given parameters.
        
        Args:
            task_id: Unique identifier for the task
            prompt: The actual task content/instruction
            task_type: Type of task (e.g., 'math_problem', 'code_generation')
            priority: Task priority (higher number = higher priority)
        """
        self.task_id = task_id
        self.prompt = prompt
        self.task_type = task_type
        self.priority = priority

def generate_random_task(task_id):
    """Generate a random task with given task_id."""
    task_types = [
        'math_problem',
        'code_generation',
        'translation',
        'summary',
        'story',
        'combined_task'
    ]

    task_prompts = {
        'math_problem': f"Calculate: {random.randint(1, 100)} + {random.randint(1, 100)} * {random.randint(1, 10)}",
        'code_generation': "Write a Python function that takes a number as input and returns its factorial.",
        'translation': "Please translate the following text into French: 'The quick brown fox jumps over the lazy dog'.",
        'summary': "Summarize the main points of an article on climate change and its impact on global economies.",
        'story': "Write a detailed story about a scientist who invents a time machine and accidentally travels to the age of dinosaurs.",
        'combined_task': f"First calculate {random.randint(1, 50)} + {random.randint(1, 50)}, then write a Python function to print this result."
    }

    task_type = random.choice(task_types)
    prompt = task_prompts[task_type]
    priority_map = {
        'math_problem': 1,
        'code_generation': 2,
        'translation': 1.5,
        'summary': 2,
        'story': 3,
        'combined_task': 3.5
    }
    priority = priority_map.get(task_type, 1)
    return Task(task_id=task_id, prompt=prompt, task_type=task_type, priority=priority)


