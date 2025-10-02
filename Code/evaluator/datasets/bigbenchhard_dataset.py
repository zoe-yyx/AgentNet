import glob
import json
import sys
import os
import pandas as pd
from typing import Union, List, Literal
import numpy as np

from evaluator.datasets.base_dataset import BaseDataset, ProblemInput

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from big_datasets.bigbenchhard.bbh_types import bbh_types
from src.task import Task

class BigBenchHardDataset(BaseDataset):
    def __init__(self,
        # split: Union[Literal['dev'], Literal['val'], Literal['test']],
        ) -> None:

        # self._split = split

        data_path = f"big_datasets/bigbenchhard/bbh/"
        self.test_data_path = f"big_datasets/bigbenchhard/bbh_test.json"
        # print("data_path: ", data_path)
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'bigbenchhard'
    
    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        # Load all JSON files from the data_path
        # print("json_paths: ", json_paths)
        total_df = pd.DataFrame()
        for task_type in bbh_types:
            with open(data_path + f"{task_type}.json", 'r') as f:
                data = json.load(f)
                # Assuming each JSON file has a key 'examples' which is a list of dictionaries
                df = pd.DataFrame(data['examples'])
                df['task_type'] = task_type
                total_df = pd.concat([total_df, df], ignore_index=True)

        total_df = total_df.reset_index(drop=True)

        # Pseudorandom shuffle
        rng = np.random.default_rng(888)
        total_df = total_df.reindex(rng.permutation(total_df.index))

        print("Total number of examples: ", len(total_df))

        return total_df
    
    @staticmethod
    def _load_a_specific_jsonl(data_path: str) -> pd.DataFrame:
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
            df = pd.DataFrame(data)
            return df
    
    @staticmethod
    def _load_data_chunks(data_path: str, chunk_index: int) -> pd.DataFrame:
        with open(data_path + f"chunk_{chunk_index}.json", 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            return df

    # @property
    # def split(self) -> str:
    #     return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record
    
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        # print("\033[94m" + "answer: " + str(answer) + " type: " + str(type(answer)) + "\033[0m")        
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

    def generate_standard_batch(self, chunk_index: List) -> List[Task]:
        data_path = f"big_datasets/bigbenchhard/data_chunks/"
        all_records = []
        for idx in chunk_index:
            records = self._load_data_chunks(data_path, idx)
            all_records.append(records)
        
        combined_records = pd.concat(all_records, ignore_index=True)
        tasks = [Task(
                task_id=index,
                task_type=record['task_type'],
                complexity=0.3,
                major_problem=record['input'],
                correct_answer=record['target'],
                ) for index, record in combined_records.iterrows()]
        return tasks
    
    def generate_morphagent_tasks(self) -> List[Task]:
        data_path = "/hdd/yxyang/AgentNet-Experiments_0226/big_datasets/bigbenchhard/shuffled_sampled_bigbenchhard.jsonl"
        df = self._load_a_specific_jsonl(data_path)
        tasks = [Task(
                task_id=index,
                task_type=record['task_type'],
                complexity=0.3,
                major_problem=record['question'],
                correct_answer=record['correct_answer'],
                ) for index, record in df.iterrows()]
        return tasks



    def generate_tasks_by_file_path(self, dataset_file_path) -> List[Task]:
        df = self._load_a_specific_jsonl(dataset_file_path)
        tasks = [Task(
                task_id=index,
                task_type=record['task_type'],
                complexity=0.3,
                major_problem=record['question'],
                correct_answer=record['correct_answer'],
                ) for index, record in df.iterrows()]
        return tasks




    def generate_mini_batch(self, batch_size: int, seed: int = 888) -> List[Task]:
        records = self._total_df.sample(batch_size, random_state=seed)
        records = records.reset_index(drop=True)
        tasks = [Task(
                task_id=index,
                task_type=record['task_type'],
                complexity=0.3,
                major_problem=record['input'],
                correct_answer=record['target'],
                ) for index, record in records.iterrows()]
        return tasks
    
    def generate_full_batch(self) -> List[Task]:
        return self.generate_mini_batch(len(self._total_df))

    def generate_test_batch(self) -> List[Task]:
        with open(self.test_data_path, 'r') as f:
            data = json.load(f)
            tasks = [Task(
                task_id=index,
                task_type=record['task_type'],
                complexity=0.3,
                major_problem=record['input'],
                correct_answer=record['target'],
                ) for index, record in enumerate(data)]
        return tasks
    

    
