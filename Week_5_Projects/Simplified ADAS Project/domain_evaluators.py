"""
Domain-Specific Evaluators for ADAS

This module demonstrates how the ADAS framework can work across different domains:
- Mathematical problem solving
- Logical reasoning
- Simple coding tasks

Each evaluator represents a different type of task that agents can be optimized for.
"""

import re
import random
import math
from typing import List, Dict, Any
from meta_agent_search import TaskEvaluator


class MathProblemEvaluator(TaskEvaluator):
    """Evaluator for mathematical problem-solving tasks"""
    
    def __init__(self):
        # Sample math problems with known solutions
        self.problems = [
            {
                "problem": "What is 15% of 240?",
                "answer": 36,
                "type": "percentage"
            },
            {
                "problem": "If a rectangle has length 8 and width 5, what is its area?",
                "answer": 40,
                "type": "geometry"
            },
            {
                "problem": "Solve for x: 2x + 5 = 17",
                "answer": 6,
                "type": "algebra"
            },
            {
                "problem": "What is the sum of first 10 natural numbers?",
                "answer": 55,
                "type": "arithmetic"
            },
            {
                "problem": "If a circle has radius 3, what is its circumference? (use π ≈ 3.14)",
                "answer": 18.84,
                "type": "geometry"
            }
        ]
    
    def evaluate(self, agent_code: str) -> float:
        """Evaluate agent on mathematical problems"""
        correct_count = 0
        total_problems = len(self.problems)
        
        for problem_data in self.problems:
            try:
                # Simulate running the agent code on the problem
                result = self._simulate_agent_execution(agent_code, problem_data)
                
                expected = problem_data["answer"]
                if isinstance(expected, float):
                    # Allow some tolerance for floating point answers
                    if abs(result - expected) < 0.1:
                        correct_count += 1
                else:
                    if result == expected:
                        correct_count += 1
                        
            except Exception:
                # Agent failed on this problem
                continue
        
        return correct_count / total_problems
    
    def _simulate_agent_execution(self, agent_code: str, problem_data: Dict) -> float:
        """Simulate how well the agent would perform based on its code structure"""
        problem = problem_data["problem"]
        answer = problem_data["answer"]
        problem_type = problem_data["type"]
        
        # Analyze agent code for relevant approaches
        code_lower = agent_code.lower()
        
        # Base probability of success
        success_prob = 0.3  # Increased base probability
        
        # Boost probability based on relevant keywords and patterns
        if problem_type == "percentage":
            if any(word in code_lower for word in ["percent", "fraction", "multiply", "divide"]):
                success_prob += 0.4
            if "0.01" in code_lower or "/100" in code_lower:
                success_prob += 0.3
                
        elif problem_type == "geometry":
            if any(word in code_lower for word in ["area", "rectangle", "circle", "radius", "π", "pi"]):
                success_prob += 0.4
            if "*" in code_lower:  # Multiplication for area
                success_prob += 0.2
                
        elif problem_type == "algebra":
            if any(word in code_lower for word in ["solve", "equation", "variable", "isolate"]):
                success_prob += 0.4
            if any(op in code_lower for op in ["subtract", "add", "divide"]):
                success_prob += 0.2
                
        elif problem_type == "arithmetic":
            if any(word in code_lower for word in ["sum", "sequence", "range", "loop"]):
                success_prob += 0.4
            if "n*(n+1)/2" in code_lower.replace(" ", ""):
                success_prob += 0.4  # Knows the formula
        
        # General problem-solving indicators
        if any(word in code_lower for word in ["analyze", "step", "method", "approach"]):
            success_prob += 0.1
        
        if any(word in code_lower for word in ["verify", "check", "validate"]):
            success_prob += 0.1
        
        # Cap probability and determine success
        success_prob = min(0.9, success_prob)
        
        if random.random() < success_prob:
            # Add some noise to the correct answer
            noise = random.uniform(-0.1, 0.1) * abs(answer) if answer != 0 else 0
            return answer + noise
        else:
            # Return a wrong answer
            return answer * random.uniform(0.5, 1.5) + random.uniform(-10, 10)
    
    def get_task_description(self) -> str:
        return "Mathematical problem solving across algebra, geometry, and arithmetic"


class LogicalReasoningEvaluator(TaskEvaluator):
    """Evaluator for logical reasoning tasks"""
    
    def __init__(self):
        self.reasoning_problems = [
            {
                "premise": "All birds can fly. Penguins are birds.",
                "question": "Can penguins fly?",
                "correct_reasoning": "logical_contradiction",  # This tests handling of exceptions
                "difficulty": "medium"
            },
            {
                "premise": "If it rains, then the ground is wet. The ground is wet.",
                "question": "Did it rain?",
                "correct_reasoning": "possible_but_not_certain",  # Affirming consequent fallacy
                "difficulty": "hard"
            },
            {
                "premise": "Some cats are black. All black things absorb heat well.",
                "question": "Do some cats absorb heat well?",
                "correct_reasoning": "yes",
                "difficulty": "easy"
            },
            {
                "premise": "No reptiles are mammals. All snakes are reptiles.",
                "question": "Are any snakes mammals?",
                "correct_reasoning": "no",
                "difficulty": "easy"
            }
        ]
    
    def evaluate(self, agent_code: str) -> float:
        """Evaluate agent on logical reasoning tasks"""
        correct_count = 0
        
        for problem in self.reasoning_problems:
            if self._evaluate_reasoning_approach(agent_code, problem):
                correct_count += 1
        
        return correct_count / len(self.reasoning_problems)
    
    def _evaluate_reasoning_approach(self, agent_code: str, problem: Dict) -> bool:
        """Evaluate if agent's approach would handle this reasoning problem well"""
        code_lower = agent_code.lower()
        difficulty = problem["difficulty"]
        correct_reasoning = problem["correct_reasoning"]
        
        # Base success probability
        base_prob = {"easy": 0.3, "medium": 0.2, "hard": 0.1}[difficulty]
        success_prob = base_prob
        
        # Look for logical reasoning indicators
        logical_keywords = [
            "premise", "conclusion", "logical", "valid", "invalid",
            "contradiction", "syllogism", "deduction", "induction"
        ]
        
        for keyword in logical_keywords:
            if keyword in code_lower:
                success_prob += 0.15
        
        # Look for careful analysis patterns
        careful_keywords = [
            "analyze", "examine", "consider", "evaluate", "assess",
            "step by step", "systematic", "methodical"
        ]
        
        for keyword in careful_keywords:
            if keyword in code_lower:
                success_prob += 0.1
        
        # Look for exception handling (important for the penguin problem)
        if correct_reasoning == "logical_contradiction":
            if any(word in code_lower for word in ["exception", "special case", "however", "but"]):
                success_prob += 0.3
        
        # Look for uncertainty handling
        if correct_reasoning == "possible_but_not_certain":
            if any(word in code_lower for word in ["possible", "maybe", "uncertain", "not certain", "could be"]):
                success_prob += 0.3
        
        success_prob = min(0.8, success_prob)
        return random.random() < success_prob
    
    def get_task_description(self) -> str:
        return "Logical reasoning and critical thinking problems"


class CodingTaskEvaluator(TaskEvaluator):
    """Evaluator for simple coding/algorithmic tasks"""
    
    def __init__(self):
        self.coding_tasks = [
            {
                "task": "Write a function to find the maximum element in a list",
                "complexity": "easy",
                "concepts": ["iteration", "comparison", "variables"]
            },
            {
                "task": "Implement a function to reverse a string",
                "complexity": "easy", 
                "concepts": ["string manipulation", "indexing", "loops"]
            },
            {
                "task": "Create a function to check if a number is prime",
                "complexity": "medium",
                "concepts": ["mathematics", "loops", "optimization", "early_termination"]
            },
            {
                "task": "Write a function to sort a list using bubble sort",
                "complexity": "medium",
                "concepts": ["nested_loops", "swapping", "algorithms", "comparison"]
            },
            {
                "task": "Implement binary search on a sorted array",
                "complexity": "hard",
                "concepts": ["binary_search", "divide_conquer", "recursion", "efficiency"]
            }
        ]
    
    def evaluate(self, agent_code: str) -> float:
        """Evaluate agent on coding tasks"""
        total_score = 0
        
        for task in self.coding_tasks:
            task_score = self._evaluate_coding_approach(agent_code, task)
            total_score += task_score
        
        return total_score / len(self.coding_tasks)
    
    def _evaluate_coding_approach(self, agent_code: str, task: Dict) -> float:
        """Evaluate if agent's approach would work for this coding task"""
        code_lower = agent_code.lower()
        complexity = task["complexity"]
        concepts = task["concepts"]
        
        # Base success rates by complexity
        base_rates = {"easy": 0.4, "medium": 0.25, "hard": 0.1}
        success_prob = base_rates[complexity]
        
        # Check for relevant programming concepts
        concept_keywords = {
            "iteration": ["loop", "for", "while", "iterate"],
            "comparison": ["compare", "greater", "less", "equal", "max", "min"],
            "variables": ["variable", "store", "assign", "value"],
            "string manipulation": ["string", "char", "reverse", "substring"],
            "indexing": ["index", "position", "array", "list"],
            "mathematics": ["math", "number", "calculate", "formula"],
            "optimization": ["optimize", "efficient", "fast", "improve"],
            "early_termination": ["break", "return early", "exit", "terminate"],
            "nested_loops": ["nested", "inner loop", "outer loop", "double loop"],
            "swapping": ["swap", "exchange", "temp", "temporary"],
            "algorithms": ["algorithm", "method", "approach", "technique"],
            "recursion": ["recursive", "recursion", "call itself", "base case"],
            "divide_conquer": ["divide", "conquer", "split", "half"],
            "binary_search": ["binary", "search", "sorted", "middle"],
            "efficiency": ["efficient", "complexity", "performance", "optimal"]
        }
        
        # Boost probability for each relevant concept found
        for concept in concepts:
            if concept in concept_keywords:
                keywords = concept_keywords[concept]
                if any(keyword in code_lower for keyword in keywords):
                    success_prob += 0.15
        
        # General programming indicators
        programming_keywords = [
            "function", "def", "method", "implement", "code",
            "algorithm", "solution", "logic", "structure"
        ]
        
        for keyword in programming_keywords:
            if keyword in code_lower:
                success_prob += 0.05
        
        # Systematic approach indicators
        if any(word in code_lower for word in ["step", "systematic", "plan", "design"]):
            success_prob += 0.1
        
        # Error handling and testing
        if any(word in code_lower for word in ["test", "verify", "check", "validate", "debug"]):
            success_prob += 0.1
        
        return min(0.85, success_prob)
    
    def get_task_description(self) -> str:
        return "Programming and algorithmic problem solving"


class MultiDomainEvaluator(TaskEvaluator):
    """Composite evaluator that tests agents across multiple domains"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.evaluators = {
            "math": MathProblemEvaluator(),
            "reasoning": LogicalReasoningEvaluator(), 
            "coding": CodingTaskEvaluator()
        }
        
        # Default equal weighting
        self.weights = weights or {"math": 1/3, "reasoning": 1/3, "coding": 1/3}
    
    def evaluate(self, agent_code: str) -> float:
        """Evaluate agent across all domains with weighted average"""
        total_score = 0
        
        for domain, evaluator in self.evaluators.items():
            domain_score = evaluator.evaluate(agent_code)
            weighted_score = domain_score * self.weights[domain]
            total_score += weighted_score
        
        return total_score
    
    def get_detailed_evaluation(self, agent_code: str) -> Dict[str, float]:
        """Get detailed breakdown of performance by domain"""
        results = {}
        
        for domain, evaluator in self.evaluators.items():
            results[domain] = evaluator.evaluate(agent_code)
        
        results["overall"] = self.evaluate(agent_code)
        return results
    
    def get_task_description(self) -> str:
        return "Multi-domain evaluation across mathematics, reasoning, and coding"


# Example usage and demonstration
def demonstrate_evaluators():
    """Demonstrate different evaluators with sample agent codes"""
    
    print("🧮 ADAS Domain Evaluators Demonstration")
    print("=" * 50)
    
    # Sample agent codes representing different approaches
    sample_agents = [
        {
            "name": "Basic Problem Solver",
            "code": '''def solve_task(problem):
    """Basic approach to problem solving"""
    # Try to understand the problem
    analysis = analyze_problem(problem)
    
    # Apply direct solution if possible
    if is_simple(problem):
        return direct_solve(problem)
    else:
        return step_by_step_solve(problem)'''
        },
        {
            "name": "Mathematical Specialist", 
            "code": '''def solve_task(problem):
    """Specialized for mathematical problems"""
    # Check problem type
    if "percent" in problem or "%" in problem:
        return solve_percentage(problem)
    elif "area" in problem or "rectangle" in problem:
        return calculate_area(problem)
    elif "equation" in problem or "solve for" in problem:
        return solve_algebra(problem)
    elif "sum" in problem:
        # Use formula for arithmetic series: n*(n+1)/2
        return arithmetic_sum(problem)
    
    # Verify result
    result = compute_solution(problem)
    return verify_result(result, problem)'''
        },
        {
            "name": "Logical Reasoner",
            "code": '''def solve_task(problem):
    """Focuses on logical reasoning"""
    # Identify premises and conclusion
    premises = extract_premises(problem)
    conclusion = extract_conclusion(problem)
    
    # Check for logical contradictions
    if has_contradiction(premises):
        return handle_exception_case(premises)
    
    # Apply deductive reasoning
    if can_deduce(premises, conclusion):
        return "yes" if is_valid_deduction(premises, conclusion) else "no"
    
    # Handle uncertainty
    if is_uncertain(premises, conclusion):
        return "possible_but_not_certain"
    
    return systematic_analysis(premises, conclusion)'''
        },
        {
            "name": "Programming Expert",
            "code": '''def solve_task(problem):
    """Algorithmic and programming approach"""
    # Analyze the algorithm needed
    if "maximum" in problem or "max" in problem:
        return implement_max_finder(problem)
    elif "reverse" in problem:
        return implement_string_reversal(problem)
    elif "prime" in problem:
        return implement_prime_check_optimized(problem)
    elif "sort" in problem:
        return implement_sorting_algorithm(problem)
    elif "binary search" in problem:
        return implement_binary_search(problem)
    
    # Use systematic programming approach
    solution = design_algorithm(problem)
    optimized = optimize_solution(solution)
    tested = test_solution(optimized)
    
    return tested'''
        }
    ]
    
    # Test each agent on different evaluators
    evaluators = {
        "Math": MathProblemEvaluator(),
        "Reasoning": LogicalReasoningEvaluator(),
        "Coding": CodingTaskEvaluator(),
        "Multi-Domain": MultiDomainEvaluator()
    }
    
    print("\n📊 EVALUATION RESULTS")
    print("=" * 60)
    
    for agent in sample_agents:
        print(f"\n🤖 Agent: {agent['name']}")
        print("-" * 40)
        
        for eval_name, evaluator in evaluators.items():
            score = evaluator.evaluate(agent['code'])
            print(f"{eval_name:12}: {score:.3f}")
    
    # Demonstrate detailed multi-domain evaluation
    print(f"\n🔍 DETAILED MULTI-DOMAIN ANALYSIS")
    print("=" * 40)
    
    multi_evaluator = MultiDomainEvaluator()
    for agent in sample_agents[:2]:  # Just show first two for brevity
        print(f"\n{agent['name']}:")
        detailed = multi_evaluator.get_detailed_evaluation(agent['code'])
        for domain, score in detailed.items():
            print(f"  {domain:9}: {score:.3f}")


if __name__ == "__main__":
    demonstrate_evaluators() 