import unittest

from utils.Executor.executor import Executor, ExecuteResult
from utils.Executor.executor_utils import function_with_timeout
from typing import List



class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 5, verbose: bool = True) -> ExecuteResult:
        # Combine function code and test cases
        # imports = 'from typing import *'
        # cleaned_code = func.replace("```python\n", "").replace("```", "")
        # combined_code = f'{imports}\n{cleaned_code}'

        # Execute the generated function code
        try:
            exec(func, globals())
        except Exception as e:
            print('\033[91m' + f"Failed to execute function code. Error: {e}" + '\033[0m')
            print('\033[91m' + f"func: {func}" + '\033[0m')
            return False, f"Failed to execute function code. Error: {e}", (False,)

        # Execute the test cases
        class_name = "TestCases"
        test_code = tests
        try:
            exec(test_code, globals())
        except Exception as e:
            print('\033[91m' + f"Failed to execute test code. Error: {e}" + '\033[0m')
            print('\033[91m' + f"test_code: {test_code}" + '\033[0m')
            return False, f"Failed to execute test code. Error: {e}", (False,)
        try:
            # Load the TestCases class from the globals
            TestTaskFunc = globals()[class_name]
        except KeyError:
            print('\033[91m' + f"TestCases class not found in the provided code." + '\033[0m')
            return False, "TestCases class not found in the provided code.", (False,)

        # Run the tests using unittest
        def run_tests(test_class):
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            runner = unittest.TextTestRunner()
            result = runner.run(suite)
            return result

        # Run tests and collect the results
        result = run_tests(TestTaskFunc)
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        passed = total_tests - failures - errors
        pass_rate = (passed / total_tests) * 100 if total_tests > 0 else 0

        # Print pass rate
        print(f"Pass rate: {pass_rate:.2f}%")

        # Collect success and failure information
        success_tests = [test for test in result.successes] if hasattr(result, 'successes') else []
        failed_tests = [f"{fail[0]} # output: {fail[1]}" for fail in result.failures]
        is_passing = (failures == 0 and errors == 0)

        state = [test not in failed_tests for test in tests]

        feedback = "Tests passed:\n" + "\n".join(success_tests) + "\n\nTests failed:"
        feedback += "\n" + "\n".join(failed_tests)
        return is_passing, feedback, tuple(state), pass_rate


    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        
        code = f"""{func}

{test}

check({name})
    """
        try:
            function_with_timeout(exec, (code, globals()), timeout)
            return True
        except Exception:
            return False
        