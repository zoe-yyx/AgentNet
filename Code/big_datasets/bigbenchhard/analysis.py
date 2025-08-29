import json

# 读取 JSONL 文件并提取 "task_type" 字段为集合
def read_task_types(file_path):
    task_types = set()
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            task_types.add(data['task_type'])  # 将 task_type 添加到集合中
    print(len(task_types))
    return task_types

# 比较两个文件的 task_type
def compare_task_types(file1_path, file2_path):
    # 从两个文件中读取 task_type 并返回集合
    task_types_file1 = read_task_types(file1_path)
    task_types_file2 = read_task_types(file2_path)

    # 比较两个集合的差异
    extra_task_types_file1 = task_types_file1 - task_types_file2
    extra_task_types_file2 = task_types_file2 - task_types_file1

    return extra_task_types_file1, extra_task_types_file2

# 使用示例
file1_path = '/hdd/yxyang/AgentNet-Experiments_debug/big_datasets/bigbenchhard/bigbenchhard_test.jsonl'  # 替换为实际文件路径
file2_path = '/hdd/yxyang/AgentNet-Experiments_debug/big_datasets/bigbenchhard/bigbenchhard_train.jsonl'  # 替换为实际文件路径

# 获取差异
extra_task_types_file1, extra_task_types_file2 = compare_task_types(file1_path, file2_path)

# 输出差异
print(f"File 1 has extra task types: {extra_task_types_file1}")
print(f"File 2 has extra task types: {extra_task_types_file2}")
