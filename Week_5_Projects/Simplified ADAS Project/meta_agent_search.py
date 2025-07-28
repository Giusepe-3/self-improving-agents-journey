"""
Simplified ADAS (Automated Design of Agentic Systems) Implementation
Core Meta Agent Search Algorithm

This implements the key ideas from the ADAS paper:
- Meta agent that automatically generates new agents
- Code-based agent definitions
- Archive of discovered agents
- Iterative improvement through feedback
"""

import json
import random
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod


@dataclass
class Agent:
    """Represents an agent with its code, performance metrics, and metadata"""
    id: str
    name: str
    code: str
    performance_history: List[float]
    creation_time: float
    parent_ids: List[str]
    domain: str
    description: str
    
    @property
    def average_performance(self) -> float:
        return sum(self.performance_history) / len(self.performance_history) if self.performance_history else 0.0
    
    @property
    def best_performance(self) -> float:
        return max(self.performance_history) if self.performance_history else 0.0


class AgentArchive:
    """Archive system for storing and retrieving discovered agents"""
    
    def __init__(self, max_size: int = 100):
        self.agents: Dict[str, Agent] = {}
        self.max_size = max_size
        
    def add_agent(self, agent: Agent) -> None:
        """Add agent to archive, removing worst performers if at capacity"""
        self.agents[agent.id] = agent
        
        # Prune if over capacity
        if len(self.agents) > self.max_size:
            self._prune_archive()
    
    def _prune_archive(self) -> None:
        """Remove worst performing agents to maintain archive size"""
        sorted_agents = sorted(
            self.agents.values(), 
            key=lambda a: a.average_performance, 
            reverse=True
        )
        
        # Keep only the best performers
        keep_agents = sorted_agents[:self.max_size // 2]
        self.agents = {agent.id: agent for agent in keep_agents}
    
    def get_top_agents(self, n: int = 5) -> List[Agent]:
        """Get top N performing agents"""
        sorted_agents = sorted(
            self.agents.values(), 
            key=lambda a: a.average_performance, 
            reverse=True
        )
        return sorted_agents[:n]
    
    def get_random_agents(self, n: int = 3) -> List[Agent]:
        """Get random agents for diversity"""
        available_agents = list(self.agents.values())
        return random.sample(available_agents, min(n, len(available_agents)))
    
    def save_to_file(self, filepath: str) -> None:
        """Save archive to JSON file"""
        archive_data = {
            agent_id: {
                **asdict(agent),
                'creation_time': agent.creation_time
            }
            for agent_id, agent in self.agents.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(archive_data, f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load archive from JSON file"""
        try:
            with open(filepath, 'r') as f:
                archive_data = json.load(f)
            
            self.agents = {}
            for agent_id, agent_data in archive_data.items():
                agent = Agent(**agent_data)
                self.agents[agent_id] = agent
        except FileNotFoundError:
            print(f"Archive file {filepath} not found. Starting with empty archive.")


class TaskEvaluator(ABC):
    """Abstract base class for evaluating agents on specific tasks"""
    
    @abstractmethod
    def evaluate(self, agent_code: str) -> float:
        """Evaluate agent code and return performance score (0-1)"""
        pass
    
    @abstractmethod
    def get_task_description(self) -> str:
        """Get description of the evaluation task"""
        pass


class MetaAgentSearch:
    """
    Core ADAS algorithm: Meta Agent Search
    
    This implements the key innovation from the paper - a meta agent that
    automatically discovers new agents by programming them in code.
    """
    
    def __init__(self, archive: AgentArchive, evaluator: TaskEvaluator):
        self.archive = archive
        self.evaluator = evaluator
        self.generation_count = 0
    
    def search(self, iterations: int = 10, verbose: bool = True) -> List[Agent]:
        """
        Run Meta Agent Search for specified iterations
        
        Returns list of newly discovered agents
        """
        discovered_agents = []
        
        for i in range(iterations):
            if verbose:
                print(f"\n=== Generation {self.generation_count + 1} ===")
            
            # Generate new agent
            new_agent = self._generate_new_agent()
            
            if new_agent:
                # Evaluate the new agent
                performance = self._evaluate_agent(new_agent)
                new_agent.performance_history.append(performance)
                
                # Add to archive if good enough
                if self._should_keep_agent(new_agent):
                    self.archive.add_agent(new_agent)
                    discovered_agents.append(new_agent)
                    
                    if verbose:
                        print(f"✓ Discovered agent '{new_agent.name}' with performance {performance:.3f}")
                else:
                    if verbose:
                        print(f"✗ Agent '{new_agent.name}' performance {performance:.3f} too low")
            
            self.generation_count += 1
        
        return discovered_agents
    
    def _generate_new_agent(self) -> Optional[Agent]:
        """Generate a new agent based on archive and exploration strategies"""
        
        # Different generation strategies
        strategies = [
            self._mutate_existing_agent,
            self._combine_agents,
            self._generate_random_agent
        ]
        
        # Weight strategies based on archive size
        if len(self.archive.agents) < 3:
            # Early exploration - more random generation
            weights = [0.2, 0.1, 0.7]
        else:
            # Later exploration - more exploitation of good agents
            weights = [0.5, 0.3, 0.2]
        
        strategy = random.choices(strategies, weights=weights)[0]
        
        try:
            return strategy()
        except Exception as e:
            print(f"Error generating agent: {e}")
            return None
    
    def _mutate_existing_agent(self) -> Agent:
        """Create new agent by mutating an existing high-performing agent"""
        parent_agents = self.archive.get_top_agents(3)
        if not parent_agents:
            return self._generate_random_agent()
        
        parent = random.choice(parent_agents)
        
        # Simple mutation: modify the agent's approach
        mutated_code = self._mutate_code(parent.code)
        
        agent_id = f"mutated_{self.generation_count}_{random.randint(1000, 9999)}"
        
        return Agent(
            id=agent_id,
            name=f"Mutated_{parent.name}_{self.generation_count}",
            code=mutated_code,
            performance_history=[],
            creation_time=time.time(),
            parent_ids=[parent.id],
            domain=parent.domain,
            description=f"Mutation of {parent.name}"
        )
    
    def _combine_agents(self) -> Agent:
        """Create new agent by combining features from multiple agents"""
        parent_agents = self.archive.get_top_agents(5)
        if len(parent_agents) < 2:
            return self._generate_random_agent()
        
        parents = random.sample(parent_agents, 2)
        
        # Simple combination: blend approaches
        combined_code = self._combine_code(parents[0].code, parents[1].code)
        
        agent_id = f"combined_{self.generation_count}_{random.randint(1000, 9999)}"
        
        return Agent(
            id=agent_id,
            name=f"Combined_{self.generation_count}",
            code=combined_code,
            performance_history=[],
            creation_time=time.time(),
            parent_ids=[p.id for p in parents],
            domain=parents[0].domain,  # Use first parent's domain
            description=f"Combination of {parents[0].name} and {parents[1].name}"
        )
    
    def _generate_random_agent(self) -> Agent:
        """Generate a completely new random agent"""
        agent_id = f"random_{self.generation_count}_{random.randint(1000, 9999)}"
        
        # Generate random agent code (this would be more sophisticated in reality)
        random_code = self._generate_random_code()
        
        return Agent(
            id=agent_id,
            name=f"Random_{self.generation_count}",
            code=random_code,
            performance_history=[],
            creation_time=time.time(),
            parent_ids=[],
            domain="general",
            description="Randomly generated agent"
        )
    
    def _mutate_code(self, original_code: str) -> str:
        """Apply mutations to agent code"""
        # This is a simplified mutation - in reality this would be much more sophisticated
        mutations = [
            "# Added exploration strategy\nif random.random() < 0.1: return random.choice(options)",
            "# Added memory mechanism\nif hasattr(self, 'memory'): self.memory.append(result)",
            "# Added verification step\nif result: double_check = verify_result(result)",
            "# Added adaptive behavior\nif performance < 0.5: strategy = 'explore' else: strategy = 'exploit'"
        ]
        
        mutation = random.choice(mutations)
        
        # Insert mutation at random position
        lines = original_code.split('\n')
        insert_pos = random.randint(0, len(lines))
        lines.insert(insert_pos, mutation)
        
        return '\n'.join(lines)
    
    def _combine_code(self, code1: str, code2: str) -> str:
        """Combine features from two agent codes"""
        # Simplified combination - take parts from each
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')
        
        combined_lines = []
        combined_lines.append("# Combined agent approach")
        
        # Take some lines from each agent
        if lines1:
            combined_lines.extend(lines1[:len(lines1)//2])
        combined_lines.append("# Incorporating second agent's approach")
        if lines2:
            combined_lines.extend(lines2[:len(lines2)//2])
        
        return '\n'.join(combined_lines)
    
    def _generate_random_code(self) -> str:
        """Generate random agent code"""
        templates = [
            '''def solve_task(problem):
    """Basic reasoning approach"""
    # Analyze the problem
    analysis = analyze_problem(problem)
    
    # Generate possible solutions
    solutions = []
    for approach in ['direct', 'step_by_step', 'reverse']:
        solution = apply_approach(problem, approach)
        solutions.append(solution)
    
    # Select best solution
    return select_best(solutions)''',
            
            '''def solve_task(problem):
    """Iterative refinement approach"""
    solution = initial_guess(problem)
    
    for iteration in range(3):
        feedback = evaluate_solution(solution, problem)
        solution = refine_solution(solution, feedback)
    
    return solution''',
            
            '''def solve_task(problem):
    """Multi-strategy approach"""
    strategies = ['analytical', 'heuristic', 'systematic']
    results = []
    
    for strategy in strategies:
        result = apply_strategy(problem, strategy)
        confidence = estimate_confidence(result)
        results.append((result, confidence))
    
    # Return result with highest confidence
    return max(results, key=lambda x: x[1])[0]'''
        ]
        
        return random.choice(templates)
    
    def _evaluate_agent(self, agent: Agent) -> float:
        """Evaluate agent performance using the task evaluator"""
        return self.evaluator.evaluate(agent.code)
    
    def _should_keep_agent(self, agent: Agent) -> bool:
        """Decide whether to keep agent in archive"""
        if not agent.performance_history:
            return False
        
        performance = agent.performance_history[-1]
        
        # Keep if performance is above threshold (made more lenient)
        threshold = 0.15  # Lowered from 0.3 to allow more agents to be kept
        
        # Or if it's better than worst agent in archive
        if self.archive.agents:
            worst_performance = min(a.average_performance for a in self.archive.agents.values())
            return performance > max(threshold, worst_performance * 0.9)  # Allow slightly worse agents
        
        return performance > threshold


def demonstrate_meta_agent_search():
    """Demonstration of the Meta Agent Search algorithm"""
    
    # This would normally be a real task evaluator
    class DummyEvaluator(TaskEvaluator):
        def evaluate(self, agent_code: str) -> float:
            # Simulate evaluation - in reality this would run the agent on actual tasks
            score = random.uniform(0.2, 0.8)  # Better base range
            
            # Slight bias toward longer, more complex code
            if len(agent_code) > 300:
                score += 0.15
            
            # Bias toward agents with certain keywords
            good_keywords = ['analyze', 'refine', 'strategy', 'confidence', 'verify', 'solution', 'approach']
            keyword_bonus = sum(0.08 for keyword in good_keywords if keyword in agent_code.lower())
            score = min(1.0, score + keyword_bonus)
            
            # Add some structure-based bonuses
            if 'def ' in agent_code:
                score += 0.05  # Has function definitions
            if 'for ' in agent_code or 'while ' in agent_code:
                score += 0.05  # Has loops
            if 'if ' in agent_code:
                score += 0.05  # Has conditionals
            
            return min(1.0, score)
        
        def get_task_description(self) -> str:
            return "General problem solving task"
    
    print("🤖 ADAS Meta Agent Search Demonstration")
    print("=" * 50)
    
    # Initialize components
    archive = AgentArchive(max_size=20)
    evaluator = DummyEvaluator()
    meta_search = MetaAgentSearch(archive, evaluator)
    
    # Run search
    discovered_agents = meta_search.search(iterations=8, verbose=True)
    
    # Show results
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 30)
    print(f"Total agents discovered: {len(discovered_agents)}")
    print(f"Archive size: {len(archive.agents)}")
    
    if archive.agents:
        top_agents = archive.get_top_agents(3)
        print(f"\n🏆 TOP PERFORMING AGENTS:")
        for i, agent in enumerate(top_agents, 1):
            print(f"{i}. {agent.name}: {agent.average_performance:.3f}")
            print(f"   Description: {agent.description}")
            print(f"   Parents: {agent.parent_ids if agent.parent_ids else 'None (original)'}")
    
    return archive, discovered_agents


if __name__ == "__main__":
    demonstrate_meta_agent_search() 