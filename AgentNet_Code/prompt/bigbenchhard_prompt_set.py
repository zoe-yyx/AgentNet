#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Any

from prompt.prompt_set import PromptSet

class BigBenchHardPromptSet(PromptSet):
    """
    BigBenchHard prompt set for the multi-type question answering.
    """

    @staticmethod
    def get_role():
        return "a knowlegable expert in question answering"

    @staticmethod
    def get_constraint():
        return """Your response should directly answer the **major task** as instructed by the question type:
                - For multiple-choice questions, respond only with the letter or number of the correct answer (e.g., (A)).
                - For true/false questions, respond only with "True" or "False".
                - For ordering tasks, respond only with the sequence of elements (e.g., C B A).
                - For questions that require completing brackets or parentheses, respond only with the exact missing part needed to complete the sequence without any additional context.
                - For fill-in-the-blank questions, respond only with the word or phrase that completes the sentence.
                - For yes/no questions, respond only with "yes" or "no" as appropriate.
                - For open-ended questions that require a short answer, respond with the specific word, phrase, or number that directly answers the question.
                Avoid repeating the question, providing explanations, or adding any context. Always respond only with the answer.
                """

    @staticmethod
    def get_split_constraint():
        return """Provide a concise response that includes a brief acknowledgment of the task you are executing which is in the **Task Description** and then the result. Ensure clarity by focusing on the specific part described in the **Task Description**. Avoid including unrelated explanations or details about parts of the task that you are not responsible for. Respond only with the necessary acknowledgment and result for your part."""
    @staticmethod
    def get_thought_constraint():
        return """You are now in the reasoning phase. Provide detailed thought processes, including possible steps, considerations, and strategies for solving the task. Avoid providing the final answer or directly responding in the specified answer format. Focus on reasoning through the task, generating possible insights, and preparing for the next steps."""


    @staticmethod
    def get_format():
        return "Respond according to the question type (e.g., letter/number for multiple choice, value for calculation, etc.)"

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
        raise NotImplementedError

