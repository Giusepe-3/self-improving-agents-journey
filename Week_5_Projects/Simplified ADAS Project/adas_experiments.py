"""
ADAS Experiments - Demonstrating the Complete Workflow

This module runs comprehensive experiments showing how the ADAS system:
1. Discovers agents across different domains
2. Shows evolution and improvement over time
3. Demonstrates transfer across domains
4. Compares to baseline approaches

Key experiments:
- Single domain optimization
- Multi-domain generalization  
- Agent evolution visualization
- Performance transfer analysis
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import time
import os

from meta_agent_search import MetaAgentSearch, AgentArchive, Agent
from domain_evaluators import (
    MathProblemEvaluator, 
    LogicalReasoningEvaluator, 
    CodingTaskEvaluator,
    MultiDomainEvaluator
)


class ExperimentRunner:
    """Runs and tracks ADAS experiments across different scenarios"""
    
    def __init__(self, results_dir: str = "experiment_results"):
        self.results_dir = results_dir
        self.ensure_results_dir()
        
    def ensure_results_dir(self):
        """Create results directory if it doesn't exist"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def run_single_domain_experiment(self, domain: str, iterations: int = 15) -> Dict:
        """Run ADAS on a single domain to show focused optimization"""
        
        print(f"🔬 Running Single Domain Experiment: {domain.upper()}")
        print("=" * 50)
        
        # Select appropriate evaluator
        evaluators = {
            "math": MathProblemEvaluator(),
            "reasoning": LogicalReasoningEvaluator(),
            "coding": CodingTaskEvaluator()
        }
        
        evaluator = evaluators[domain]
        archive = AgentArchive(max_size=30)
        meta_search = MetaAgentSearch(archive, evaluator)
        
        # Track performance over time
        performance_history = []
        generation_data = []
        
        # Run search
        for i in range(iterations):
            print(f"\n--- Generation {i+1} ---")
            
            # Run one iteration
            discovered = meta_search.search(iterations=1, verbose=False)
            
            # Record performance
            if archive.agents:
                top_agents = archive.get_top_agents(1)
                best_performance = top_agents[0].average_performance if top_agents else 0
                avg_performance = np.mean([a.average_performance for a in archive.agents.values()])
            else:
                best_performance = 0
                avg_performance = 0
            
            performance_history.append({
                "generation": i + 1,
                "best_performance": best_performance,
                "average_performance": avg_performance,
                "archive_size": len(archive.agents)
            })
            
            print(f"Best: {best_performance:.3f}, Avg: {avg_performance:.3f}, Archive: {len(archive.agents)}")
        
        # Collect final results
        results = {
            "domain": domain,
            "iterations": iterations,
            "performance_history": performance_history,
            "final_archive_size": len(archive.agents),
            "top_agents": []
        }
        
        # Get details of top agents
        top_agents = archive.get_top_agents(3)
        for agent in top_agents:
            results["top_agents"].append({
                "name": agent.name,
                "performance": agent.average_performance,
                "description": agent.description,
                "generation": len([p for p in performance_history 
                                if p["generation"] <= meta_search.generation_count])
            })
        
        # Save results
        self.save_experiment_results(f"single_domain_{domain}", results)
        
        return results
    
    def run_multi_domain_experiment(self, iterations: int = 20) -> Dict:
        """Run ADAS on multi-domain tasks to show generalization"""
        
        print(f"🌐 Running Multi-Domain Experiment")
        print("=" * 50)
        
        # Use multi-domain evaluator
        evaluator = MultiDomainEvaluator()
        archive = AgentArchive(max_size=40)
        meta_search = MetaAgentSearch(archive, evaluator)
        
        # Track detailed performance across domains
        performance_history = []
        
        for i in range(iterations):
            print(f"\n--- Generation {i+1} ---")
            
            # Run one iteration
            discovered = meta_search.search(iterations=1, verbose=False)
            
            # Evaluate top agent on each domain separately
            if archive.agents:
                top_agent = archive.get_top_agents(1)[0]
                detailed_performance = evaluator.get_detailed_evaluation(top_agent.code)
            else:
                detailed_performance = {"math": 0, "reasoning": 0, "coding": 0, "overall": 0}
            
            performance_data = {
                "generation": i + 1,
                "archive_size": len(archive.agents),
                **detailed_performance
            }
            
            performance_history.append(performance_data)
            
            print(f"Overall: {detailed_performance['overall']:.3f} | " +
                  f"Math: {detailed_performance['math']:.3f} | " +
                  f"Reasoning: {detailed_performance['reasoning']:.3f} | " +
                  f"Coding: {detailed_performance['coding']:.3f}")
        
        results = {
            "experiment_type": "multi_domain",
            "iterations": iterations,
            "performance_history": performance_history,
            "final_archive_size": len(archive.agents)
        }
        
        # Save results
        self.save_experiment_results("multi_domain", results)
        
        return results
    
    def run_transfer_learning_experiment(self) -> Dict:
        """Test how agents trained on one domain perform on others"""
        
        print(f"🔄 Running Transfer Learning Experiment")
        print("=" * 50)
        
        domains = ["math", "reasoning", "coding"]
        evaluators = {
            "math": MathProblemEvaluator(),
            "reasoning": LogicalReasoningEvaluator(), 
            "coding": CodingTaskEvaluator()
        }
        
        transfer_matrix = {}
        
        # Train agents on each domain
        trained_agents = {}
        for source_domain in domains:
            print(f"\n📚 Training agents on {source_domain.upper()}")
            
            archive = AgentArchive(max_size=20)
            meta_search = MetaAgentSearch(archive, evaluators[source_domain])
            
            # Train for several iterations
            meta_search.search(iterations=12, verbose=False)
            
            # Get best trained agent
            if archive.agents:
                best_agent = archive.get_top_agents(1)[0]
                trained_agents[source_domain] = best_agent
                print(f"✓ Best {source_domain} agent: {best_agent.average_performance:.3f}")
        
        # Test each trained agent on all domains
        print(f"\n🧪 Testing cross-domain transfer")
        
        for source_domain, agent in trained_agents.items():
            transfer_matrix[source_domain] = {}
            
            for target_domain in domains:
                # Evaluate the agent on target domain
                target_evaluator = evaluators[target_domain] 
                performance = target_evaluator.evaluate(agent.code)
                transfer_matrix[source_domain][target_domain] = performance
                
                transfer_indicator = "✓" if performance > 0.4 else "○" if performance > 0.2 else "✗"
                print(f"{source_domain:8} → {target_domain:8}: {performance:.3f} {transfer_indicator}")
        
        results = {
            "experiment_type": "transfer_learning",
            "transfer_matrix": transfer_matrix,
            "trained_agents": {domain: {
                "name": agent.name,
                "performance": agent.average_performance,
                "description": agent.description
            } for domain, agent in trained_agents.items()}
        }
        
        # Save results
        self.save_experiment_results("transfer_learning", results)
        
        return results
    
    def run_baseline_comparison(self, domain: str = "math") -> Dict:
        """Compare ADAS to baseline approaches"""
        
        print(f"📊 Running Baseline Comparison: {domain.upper()}")
        print("=" * 50)
        
        evaluators = {
            "math": MathProblemEvaluator(),
            "reasoning": LogicalReasoningEvaluator(),
            "coding": CodingTaskEvaluator()
        }
        
        evaluator = evaluators[domain]
        
        # Baseline 1: Random agents
        print("🎲 Testing random baseline...")
        random_performances = []
        for _ in range(20):
            # Create random agent
            archive = AgentArchive()
            meta_search = MetaAgentSearch(archive, evaluator)
            random_agent = meta_search._generate_random_agent()
            performance = evaluator.evaluate(random_agent.code)
            random_performances.append(performance)
        
        random_baseline = np.mean(random_performances)
        
        # Baseline 2: Hand-designed expert
        print("👨‍💻 Testing hand-designed baseline...")
        
        expert_codes = {
            "math": '''def solve_task(problem):
    """Hand-designed math expert"""
    problem_lower = problem.lower()
    
    # Handle percentages
    if "%" in problem or "percent" in problem:
        numbers = extract_numbers(problem)
        if len(numbers) >= 2:
            return numbers[0] * numbers[1] / 100
    
    # Handle geometry
    if "area" in problem_lower and "rectangle" in problem_lower:
        numbers = extract_numbers(problem)
        if len(numbers) >= 2:
            return numbers[0] * numbers[1]
    
    # Handle algebra
    if "solve for" in problem_lower:
        # Simple linear equation solver
        return solve_linear_equation(problem)
    
    # Handle arithmetic sequences
    if "sum" in problem_lower and "natural" in problem_lower:
        n = extract_first_number(problem)
        return n * (n + 1) // 2
    
    return basic_calculation(problem)''',
            
            "reasoning": '''def solve_task(problem):
    """Hand-designed reasoning expert"""
    premises, question = parse_logical_problem(problem)
    
    # Check for contradictions
    if contains_contradiction(premises):
        return "contradiction_detected"
    
    # Apply basic syllogistic reasoning
    if can_apply_syllogism(premises, question):
        return apply_syllogism(premises, question)
    
    # Handle uncertainty
    if is_affirming_consequent(premises, question):
        return "possible_but_not_certain"
    
    return analyze_step_by_step(premises, question)''',
            
            "coding": '''def solve_task(problem):
    """Hand-designed coding expert"""
    problem_lower = problem.lower()
    
    # Pattern matching for common problems
    if "maximum" in problem_lower:
        return "max(input_list)"
    elif "reverse" in problem_lower and "string" in problem_lower:
        return "return string[::-1]"
    elif "prime" in problem_lower:
        return implement_optimized_prime_check()
    elif "sort" in problem_lower:
        return implement_bubble_sort()
    elif "binary search" in problem_lower:
        return implement_binary_search()
    
    return generic_algorithm_template()'''
        }
        
        expert_performance = evaluator.evaluate(expert_codes[domain])
        
        # ADAS performance
        print("🤖 Testing ADAS...")
        archive = AgentArchive(max_size=25)
        meta_search = MetaAgentSearch(archive, evaluator)
        
        # Run ADAS
        meta_search.search(iterations=15, verbose=False)
        
        if archive.agents:
            adas_performance = archive.get_top_agents(1)[0].average_performance
        else:
            adas_performance = 0
        
        results = {
            "domain": domain,
            "baselines": {
                "random": random_baseline,
                "expert": expert_performance,
                "adas": adas_performance
            },
            "improvements": {
                "vs_random": adas_performance / random_baseline if random_baseline > 0 else float('inf'),
                "vs_expert": adas_performance / expert_performance if expert_performance > 0 else float('inf')
            }
        }
        
        print(f"\n📈 COMPARISON RESULTS:")
        print(f"Random baseline:    {random_baseline:.3f}")
        print(f"Expert baseline:    {expert_performance:.3f}")
        print(f"ADAS best:          {adas_performance:.3f}")
        print(f"Improvement vs Random: {results['improvements']['vs_random']:.2f}x")
        print(f"Improvement vs Expert: {results['improvements']['vs_expert']:.2f}x")
        
        # Save results
        self.save_experiment_results(f"baseline_comparison_{domain}", results)
        
        return results
    
    def save_experiment_results(self, experiment_name: str, results: Dict):
        """Save experiment results to JSON file"""
        filepath = os.path.join(self.results_dir, f"{experiment_name}.json")
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filepath}")
    
    def visualize_performance_evolution(self, experiment_results: Dict, save_plot: bool = True):
        """Create visualization of performance evolution over time"""
        
        if "performance_history" not in experiment_results:
            print("No performance history to visualize")
            return
        
        history = experiment_results["performance_history"]
        generations = [h["generation"] for h in history]
        
        plt.figure(figsize=(12, 8))
        
        if "math" in history[0]:  # Multi-domain experiment
            # Plot each domain separately
            domains = ["math", "reasoning", "coding", "overall"]
            colors = ["blue", "green", "red", "black"]
            
            for domain, color in zip(domains, colors):
                performances = [h[domain] for h in history]
                plt.plot(generations, performances, 
                        label=domain.capitalize(), 
                        color=color, 
                        linewidth=2,
                        marker='o' if domain == "overall" else None)
        
        else:  # Single domain experiment
            best_performances = [h["best_performance"] for h in history]
            avg_performances = [h["average_performance"] for h in history]
            
            plt.plot(generations, best_performances, 
                    label="Best Agent", color="red", linewidth=2, marker='o')
            plt.plot(generations, avg_performances, 
                    label="Archive Average", color="blue", linewidth=2, marker='s')
        
        plt.xlabel("Generation")
        plt.ylabel("Performance Score")
        plt.title(f"ADAS Performance Evolution - {experiment_results.get('domain', 'Multi-Domain')}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        if save_plot:
            plot_name = f"performance_evolution_{experiment_results.get('domain', 'multi_domain')}.png"
            plt.savefig(os.path.join(self.results_dir, plot_name), dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {plot_name}")
        
        plt.show()
    
    def run_complete_adas_demonstration(self):
        """Run the complete suite of ADAS experiments"""
        
        print("🚀 COMPLETE ADAS DEMONSTRATION")
        print("=" * 60)
        print("This demonstration shows the key capabilities of ADAS:")
        print("1. Single-domain optimization")
        print("2. Multi-domain generalization") 
        print("3. Transfer learning across domains")
        print("4. Comparison with baseline approaches")
        print("=" * 60)
        
        all_results = {}
        
        # Run single domain experiments
        for domain in ["math", "reasoning", "coding"]:
            print(f"\n\n{'='*20} SINGLE DOMAIN: {domain.upper()} {'='*20}")
            results = self.run_single_domain_experiment(domain, iterations=12)
            all_results[f"single_{domain}"] = results
        
        # Run multi-domain experiment
        print(f"\n\n{'='*20} MULTI-DOMAIN EXPERIMENT {'='*20}")
        multi_results = self.run_multi_domain_experiment(iterations=15)
        all_results["multi_domain"] = multi_results
        
        # Run transfer learning experiment
        print(f"\n\n{'='*20} TRANSFER LEARNING {'='*20}")
        transfer_results = self.run_transfer_learning_experiment()
        all_results["transfer_learning"] = transfer_results
        
        # Run baseline comparisons
        for domain in ["math", "reasoning"]:
            print(f"\n\n{'='*20} BASELINE COMPARISON: {domain.upper()} {'='*20}")
            baseline_results = self.run_baseline_comparison(domain)
            all_results[f"baseline_{domain}"] = baseline_results
        
        # Save complete results
        self.save_experiment_results("complete_demonstration", all_results)
        
        print(f"\n\n🎉 DEMONSTRATION COMPLETE!")
        print(f"All results saved to {self.results_dir}/")
        
        return all_results


def main():
    """Main function to run ADAS experiments"""
    
    runner = ExperimentRunner()
    
    # Run complete demonstration
    results = runner.run_complete_adas_demonstration()
    
    # Create some visualizations
    if "single_math" in results:
        runner.visualize_performance_evolution(results["single_math"])
    
    if "multi_domain" in results:
        runner.visualize_performance_evolution(results["multi_domain"])


if __name__ == "__main__":
    main() 