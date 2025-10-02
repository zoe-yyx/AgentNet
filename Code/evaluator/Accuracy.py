class Accuracy:
    def __init__(self):
        self.correct_count = 0
        self.total_count = 0

    def update(self, is_correct):
        self.total_count += 1
        if is_correct:
            self.correct_count += 1

    def get_accuracy(self):
        return self.correct_count / self.total_count if self.total_count > 0 else 0




# import numpy as np

# class Accuracy:
#     """
#     Given the results evaluated against the testcases we output some statistics.

#     >>> print_results(EXAMPLE_RESULTS, EXAMPLE_ARGS)
#     number of compile errors = 1 avg = 0.2
#     number of runtime errors = 1 avg = 0.2
#     number of test cases run = 5
#     Test Case Average (average accuracy over problems) = 0.3
#     Strict Accuracy (all test cases passed / total problems) = 0.2
#     """
#     def __init__(self):
#         self.res = []
#         self.per_prob_res = []
#         self.all_correct = []
#         self.compile_errors = 0
#         self.runtime_errors = 0
#         self.total_testcases = 0

#     def update(self, success):
#         problem_results = np.asarray(success)
#         self.res.extend(problem_results)
#         self.per_prob_res.append(np.mean(problem_results > 0))
#         self.all_correct.append(np.all(problem_results > 0))
#         self.compile_errors = len([e for e in self.res if -2 in e])
#         self.runtime_errors = len([e for e in self.res if -1 in e])
#         self.total_testcases = len(self.res)


#     def get_accuracy(self):
#         """
#         Test Case Average (average accuracy over problems)
#         Strict Accuracy (all test cases passed / total problems) 
#         """
#         return np.mean(self.per_prob_res), np.mean(self.all_correct)

#     def print_results(self):
#         print(f"number of compile errors = {self.compile_errors} avg = {self.compile_errors / self.total_testcases }")
#         print(f"number of runtime errors = {self.runtime_errors} avg = {self.runtime_errors / self.total_testcases}")
#         print(f"number of test cases run = {self.total_testcases}")
#         print(f"Test Case Average (average accuracy over problems) = {np.mean(self.per_prob_res)}")
#         print(f"Strict Accuracy (all test cases passed / total problems) = {np.mean(self.all_correct)}")


