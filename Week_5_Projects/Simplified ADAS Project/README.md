# Simplified ADAS (Automated Design of Agentic Systems)

This project implements a simplified version of the **ADAS (Automated Design of Agentic Systems)** framework from the paper by Hu, Lu, and Clune (2024). ADAS represents a breakthrough approach where agents automatically discover and program new agents, moving beyond hand-designed solutions to learned, self-improving agentic systems.

## 🎯 Core Concept

**ADAS Key Innovation**: Instead of manually designing agents, a "meta agent" automatically programs new agents in code, leveraging the fact that programming languages are Turing Complete to theoretically enable learning any possible agentic system.

## 🏗️ Architecture

### Core Components

1. **`meta_agent_search.py`** - The heart of ADAS

   - `MetaAgentSearch`: Implements the core algorithm
   - `AgentArchive`: Manages discovered agents with performance-based pruning
   - `Agent`: Data structure representing discovered agents

2. **`domain_evaluators.py`** - Cross-domain evaluation system

   - `MathProblemEvaluator`: Tests mathematical reasoning
   - `LogicalReasoningEvaluator`: Tests logical/critical thinking
   - `CodingTaskEvaluator`: Tests programming/algorithmic skills
   - `MultiDomainEvaluator`: Composite evaluation across all domains

3. **`adas_experiments.py`** - Comprehensive experimental framework
   - Single-domain optimization experiments
   - Multi-domain generalization tests
   - Transfer learning analysis
   - Baseline comparisons

## 🔍 Key ADAS Principles Demonstrated

### 1. **Meta Agent Search Algorithm**

```python
# The meta agent uses three strategies to discover new agents:
strategies = [
    self._mutate_existing_agent,    # Evolve successful agents
    self._combine_agents,           # Combine features from multiple agents
    self._generate_random_agent     # Explore novel approaches
]
```

### 2. **Code-Based Agent Definition**

Agents are represented as executable code rather than just prompts:

```python
@dataclass
class Agent:
    id: str
    name: str
    code: str                    # The agent's actual code
    performance_history: List[float]
    parent_ids: List[str]        # Genealogy tracking
    domain: str
    description: str
```

### 3. **Performance-Driven Evolution**

```python
def _should_keep_agent(self, agent: Agent) -> bool:
    """Keep agents that outperform existing archive members"""
    performance = agent.performance_history[-1]
    threshold = max(0.3, min_archive_performance)
    return performance > threshold
```

### 4. **Cross-Domain Transfer**

Agents discovered in one domain are tested on others, demonstrating generalization:

```python
# Train on math → Test on reasoning, coding
# Train on coding → Test on math, reasoning
# etc.
```

## 🚀 Quick Start

### Basic Demo

```bash
python meta_agent_search.py
```

### Domain Evaluator Demo

```bash
python domain_evaluators.py
```

### Full Experimental Suite

```bash
python adas_experiments.py
```

## 📊 Example Output

### Meta Agent Search in Action

```
🤖 ADAS Meta Agent Search Demonstration
==================================================

=== Generation 1 ===
✓ Discovered agent 'Random_0' with performance 0.456

=== Generation 2 ===
✓ Discovered agent 'Mutated_Random_0_1' with performance 0.523

=== Generation 3 ===
✓ Discovered agent 'Combined_2' with performance 0.678
...

📊 RESULTS SUMMARY
==============================
Total agents discovered: 6
Archive size: 6

🏆 TOP PERFORMING AGENTS:
1. Combined_5: 0.734
   Description: Combination of Mutated_Random_0_1 and Random_3
   Parents: mutated_0_1234, random_0_5678
```

### Cross-Domain Transfer Results

```
🔄 Running Transfer Learning Experiment
==================================================

📚 Training agents on MATH
✓ Best math agent: 0.689

📚 Training agents on REASONING
✓ Best reasoning agent: 0.612

📚 Training agents on CODING
✓ Best coding agent: 0.578

🧪 Testing cross-domain transfer
math     → math    : 0.689 ✓
math     → reasoning: 0.423 ✓
math     → coding   : 0.234 ○
reasoning → math    : 0.345 ○
reasoning → reasoning: 0.612 ✓
reasoning → coding   : 0.289 ○
coding   → math     : 0.198 ✗
coding   → reasoning: 0.334 ○
coding   → coding   : 0.578 ✓
```

## 🧪 Experimental Framework

### 1. Single Domain Optimization

Tests focused improvement in specific domains:

- **Math**: Percentage, algebra, geometry problems
- **Reasoning**: Logic, contradiction detection, syllogisms
- **Coding**: Algorithms, data structures, optimization

### 2. Multi-Domain Generalization

Evaluates agents across all domains simultaneously, testing the hypothesis that general problem-solving capabilities can emerge.

### 3. Transfer Learning Analysis

Trains agents on one domain, then tests performance on others, measuring cross-domain knowledge transfer.

### 4. Baseline Comparisons

Compares ADAS-discovered agents against:

- **Random baseline**: Randomly generated agents
- **Expert baseline**: Hand-crafted domain-specific agents

## 📈 Key Results from Paper

The original ADAS paper demonstrated:

- **Significant improvements** over hand-designed agents across multiple domains
- **Cross-domain transfer**: Agents maintained performance when moved between domains
- **Progressive improvement**: Meta agent search consistently discovered better agents over time
- **Turing completeness advantage**: Code-based agents could implement any computable strategy

## 🔧 Extension Points

### Adding New Domains

```python
class YourDomainEvaluator(TaskEvaluator):
    def evaluate(self, agent_code: str) -> float:
        # Implement your domain-specific evaluation
        return performance_score

    def get_task_description(self) -> str:
        return "Description of your domain"
```

### Custom Agent Generation Strategies

```python
def _your_custom_strategy(self) -> Agent:
    # Implement novel agent generation approach
    return new_agent
```

### Advanced Archive Management

```python
class AdvancedArchive(AgentArchive):
    def _prune_archive(self):
        # Implement sophisticated pruning strategies
        # E.g., maintain diversity, consider agent age, etc.
```

## 🎓 Educational Value

This implementation demonstrates several key AI/ML concepts:

1. **Meta-learning**: Learning how to learn/design better agents
2. **Evolutionary algorithms**: Selection, mutation, crossover of solutions
3. **Transfer learning**: Knowledge application across domains
4. **Code generation**: Automated programming and self-modification
5. **Multi-objective optimization**: Balancing performance vs. generalization

## 📚 References

- **Original Paper**: "Automated Design of Agentic Systems" by Shengran Hu, Cong Lu, Jeff Clune (2024)
- **ArXiv**: https://arxiv.org/abs/2408.08435
- **Follow-up Work**: "SwarmAgentic" (2025) showing +261.8% improvement over ADAS

## 🚀 Future Directions

Potential extensions inspired by the paper:

1. **Real LLM Integration**: Use actual language models instead of simulated evaluation
2. **More Sophisticated Code Generation**: AST manipulation, advanced mutations
3. **Distributed Search**: Parallel agent discovery across multiple processes
4. **Safety Constraints**: Ensuring discovered agents remain aligned and safe
5. **Human-in-the-Loop**: Interactive refinement of discovered agents

---

**Note**: This is a simplified educational implementation. The full ADAS system would involve actual LLM-based code generation, more sophisticated evaluation environments, and advanced search strategies.
