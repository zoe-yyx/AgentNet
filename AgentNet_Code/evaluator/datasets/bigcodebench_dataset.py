import glob
import json
import sys
import os
import pandas as pd
from typing import Union, List, Literal
import numpy as np

from evaluator.datasets.base_dataset import BaseDataset
# 将项目根目录添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.task import Task

class BigCodeBenchDataset(BaseDataset):
    def __init__(self,
        # split: Union[Literal['dev'], Literal['val'], Literal['test']],
        ) -> None:

        self.chunk_size = 100
        self.num_standard_batches = 65

        data_path = f"big_datasets/bigcodebench/data_jsonl/"
        # print("data_path: ", data_path)
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'bigcodebench'
    
    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        data_list = []
        with open(data_path + f"train-00000-of-00001.jsonl", 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                data_list.append(data)
        total_df = pd.DataFrame(data_list)

        total_df = total_df.reset_index(drop=True)

        # Pseudorandom shuffle
        rng = np.random.default_rng(888)
        total_df = total_df.reindex(rng.permutation(total_df.index))

        print("Total number of tasks: ", len(total_df))

        return total_df

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record
    
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:      
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        return answer

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['target']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer

    def generate_mini_batch(self, batch_size: int, seed: int = 888) -> List[Task]:
        records = self._total_df.sample(batch_size, random_state=seed)
        records = records.reset_index(drop=True)
        tasks = [Task(
                task_id=index,
                task_type='code_generation',
                complexity=0.5,
                major_problem=record['complete_prompt'],
                test_cases=record['test'],
                ) for index, record in records.iterrows()]
        return tasks
    
    def generate_full_batch(self) -> List[Task]:
        return self.generate_mini_batch(len(self._total_df))
    
    
