# Math Self-Refiner Solver

A small Llama-3 powered tutor that solves GSM8K problems, critiques its own output, and retries until it’s correct.

## Features

- Two-stage self-evaluation loop
- 30-sample GSM8K benchmark
- Plug-and-play with any local Ollama model

## Concept Explanation

Chain-of-Thought: It is a reasoning technique where a model writes out step-by-step intermediate thoughts. In this method, the model divides the task into small, easier subtasks, resolves these subtasks, and in the process of dividing and conquering, makes it easier for the model to obtain the correct answer by combining the partial solutions.

Self-reflection: It is a refinement strategy used by the model in this project, this method helps the model reflect over the problems that gets wrong in the following way, the model sees that the wrong answer, critiques the answer in a strucutred manner and feeds the critique back in the model with a refine structure, this process help improve the final answer.
