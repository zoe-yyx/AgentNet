#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Any

from prompt.prompt_set import PromptSet


class DefaultPromptSet(PromptSet):
    """
    Default prompt set for the multi-type question answering.
    """

    @staticmethod
    def get_role():
        return "an helpful AI assistant"

    @staticmethod
    def get_constraint():
        return """provide your answer in a clear and structured format."""
    
    @staticmethod
    def get_split_constraint():
        return """Provide a concise response that includes a brief acknowledgment of the task you are executing which is in the **Task Description** and then the result. Ensure clarity by focusing on the specific part described in the **Task Description**. Avoid including unrelated explanations or details about parts of the task that you are not responsible for. Respond only with the necessary acknowledgment and result for your part."""

    @staticmethod
    def get_thought_constraint():
        # return """You are now in the reasoning phase. Provide detailed thought processes, including possible steps, considerations, and strategies for solving the task. Avoid providing the final answer or directly responding in the specified answer format. Focus on reasoning through the task, generating possible insights, and preparing for the next steps."""
        return """You are now in the reasoning phase. Focus on providing concise and actionable thoughts, outlining possible steps and key considerations for solving the task. Avoid delivering the final answer or adhering to a specific format. Prioritize generating insights and preparing for efficient task execution."""

    @staticmethod
    def get_format():
        return "provide your answer in a clear and structured format."

    @staticmethod
    def get_answer_prompt(question):
        return f"""Here is your question: {question} Provide your answer as instructed by the question type."""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Answer a lie to the following question: {question}. """


    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        pass
