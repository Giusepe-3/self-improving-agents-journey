"""
Quick ADAS Demo - 5-Minute Overview

This script provides a condensed demonstration of the core ADAS concepts:
1. Meta agent search discovering new agents
2. Cross-domain evaluation
3. Performance improvement over generations

Perfect for getting a quick taste of what ADAS can do!
"""

import random
from meta_agent_search import MetaAgentSearch, AgentArchive
from domain_evaluators import MathProblemEvaluator, MultiDomainEvaluator


def quick_adas_demo():
    """5-minute demo showing ADAS core functionality"""
    
    print("🚀 ADAS QUICK DEMO - Core Concepts in 5 Minutes")
    print("=" * 55)
    print("This demo shows:")
    print("✓ Meta agent automatically discovering new agents")
    print("✓ Agents improving over generations") 
    print("✓ Cross-domain evaluation")
    print("✓ Code-based agent evolution")
    print()
    
    # ============================================================================
    # DEMO 1: Single Domain Agent Discovery
    # ============================================================================
    
    print("🔬 DEMO 1: Single Domain Agent Discovery (Math)")
    print("-" * 50)
    
    # Setup
    evaluator = MathProblemEvaluator()
    archive = AgentArchive(max_size=15)
    meta_search = MetaAgentSearch(archive, evaluator)
    
    print("Starting agent discovery process...")
    print()
    
    # Run for 6 generations
    performance_progression = []
    
    for generation in range(1, 7):
        # Discover one agent per generation
        discovered = meta_search.search(iterations=1, verbose=False)
        
        # Track best performance
        if archive.agents:
            best_agent = archive.get_top_agents(1)[0]
            best_performance = best_agent.average_performance
            archive_size = len(archive.agents)
            
            performance_progression.append(best_performance)
            
            print(f"Generation {generation}: Best agent performance = {best_performance:.3f} "
                  f"(Archive: {archive_size} agents)")
            
            # Show agent details
            if generation in [1, 3, 6]:  # Show details for a few generations
                print(f"  └─ Agent '{best_agent.name}': {best_agent.description}")
                if best_agent.parent_ids:
                    print(f"     Parents: {best_agent.parent_ids}")
        else:
            print(f"Generation {generation}: No agents discovered yet")
    
    print()
    
    # Show improvement over time
    if len(performance_progression) > 1:
        improvement = performance_progression[-1] - performance_progression[0]
        improvement_pct = (improvement / performance_progression[0]) * 100 if performance_progression[0] > 0 else 0
        print(f"📈 IMPROVEMENT: {improvement:+.3f} ({improvement_pct:+.1f}%) from generation 1 to {len(performance_progression)}")
    
    print()
    
    # ============================================================================
    # DEMO 2: Agent Code Evolution
    # ============================================================================
    
    print("🧬 DEMO 2: Agent Code Evolution")
    print("-" * 50)
    
    if archive.agents:
        top_agents = archive.get_top_agents(3)
        
        print("Showing how agents evolved through code mutations and combinations:")
        print()
        
        for i, agent in enumerate(top_agents, 1):
            print(f"🤖 AGENT #{i}: {agent.name} (Performance: {agent.average_performance:.3f})")
            print(f"   Description: {agent.description}")
            
            # Show a snippet of the agent's code
            code_lines = agent.code.split('\n')
            relevant_lines = [line for line in code_lines if line.strip() and not line.strip().startswith('#')][:3]
            
            print("   Code snippet:")
            for line in relevant_lines:
                print(f"     {line.strip()}")
            
            if agent.parent_ids:
                print(f"   Evolution: Derived from {len(agent.parent_ids)} parent agent(s)")
            else:
                print("   Evolution: Original random agent")
            
            print()
    
    # ============================================================================
    # DEMO 3: Multi-Domain Performance
    # ============================================================================
    
    print("🌐 DEMO 3: Multi-Domain Evaluation")
    print("-" * 50)
    
    # Test best agent across multiple domains
    if archive.agents:
        best_agent = archive.get_top_agents(1)[0]
        multi_evaluator = MultiDomainEvaluator()
        
        print(f"Testing '{best_agent.name}' across all domains:")
        print()
        
        detailed_scores = multi_evaluator.get_detailed_evaluation(best_agent.code)
        
        domain_symbols = {
            "math": "🧮",
            "reasoning": "🧠", 
            "coding": "💻",
            "overall": "🎯"
        }
        
        for domain, score in detailed_scores.items():
            symbol = domain_symbols.get(domain, "📊")
            bar_length = int(score * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            print(f"{symbol} {domain.capitalize():9}: {score:.3f} |{bar}|")
        
        print()
        
        # Interpret results
        overall_score = detailed_scores["overall"]
        if overall_score > 0.6:
            print("🎉 Excellent! This agent shows strong general problem-solving ability.")
        elif overall_score > 0.4:
            print("👍 Good! This agent demonstrates decent cross-domain capabilities.")
        else:
            print("🔄 The agent is still learning. More generations could improve performance.")
    
    # ============================================================================
    # DEMO 4: Quick Baseline Comparison
    # ============================================================================
    
    print()
    print("📊 DEMO 4: Quick Baseline Comparison")  
    print("-" * 50)
    
    # Generate a few random agents for comparison
    random_performances = []
    for _ in range(5):
        random_agent = meta_search._generate_random_agent()
        random_perf = evaluator.evaluate(random_agent.code)
        random_performances.append(random_perf)
    
    avg_random = sum(random_performances) / len(random_performances)
    
    # Compare with our best discovered agent
    if archive.agents:
        best_discovered = archive.get_top_agents(1)[0].average_performance
        
        print(f"Random baseline (avg of 5):  {avg_random:.3f}")
        print(f"ADAS best discovered:        {best_discovered:.3f}")
        
        if avg_random > 0:  # Avoid division by zero
            if best_discovered > avg_random:
                improvement_factor = best_discovered / avg_random
                print(f"🚀 ADAS improved by {improvement_factor:.1f}x over random baseline!")
            else:
                print("🔄 More generations needed to surpass random baseline.")
        else:
            if best_discovered > 0:
                print("🚀 ADAS discovered working agents while random baseline failed completely!")
            else:
                print("🔄 Both ADAS and random baseline need more work.")
    
    # ============================================================================
    # DEMO CONCLUSION
    # ============================================================================
    
    print()
    print("🎓 DEMO COMPLETE - Key Takeaways")
    print("=" * 55)
    print("✓ Meta agents can automatically discover new agents")
    print("✓ Performance improves through evolutionary search")
    print("✓ Agents are represented as executable code") 
    print("✓ Cross-domain capabilities can emerge")
    print("✓ ADAS outperforms naive random baselines")
    print()
    print("🔬 For deeper exploration, run:")
    print("   • python meta_agent_search.py    (Core algorithm)")
    print("   • python domain_evaluators.py    (Cross-domain testing)")
    print("   • python adas_experiments.py     (Full experimental suite)")
    print()
    print("📚 This demonstrates the core ideas from the ADAS paper:")
    print("   'Automated Design of Agentic Systems' (Hu, Lu, Clune 2024)")


def interactive_agent_inspector():
    """Bonus: Interactive agent code inspector"""
    
    print("\n" + "="*55)
    print("🔍 BONUS: Interactive Agent Inspector")
    print("="*55)
    
    # Generate a few different agents to inspect
    evaluator = MathProblemEvaluator()
    archive = AgentArchive()
    meta_search = MetaAgentSearch(archive, evaluator)
    
    # Generate different types of agents
    agents_to_show = []
    
    # Random agent
    random_agent = meta_search._generate_random_agent()
    random_agent.performance_history = [evaluator.evaluate(random_agent.code)]
    agents_to_show.append(("Random Agent", random_agent))
    
    # Add to archive and create mutations
    archive.add_agent(random_agent)
    
    # Mutated agent
    mutated_agent = meta_search._mutate_existing_agent()
    mutated_agent.performance_history = [evaluator.evaluate(mutated_agent.code)]
    agents_to_show.append(("Mutated Agent", mutated_agent))
    
    print("Here are different agent types you can inspect:")
    print()
    
    for i, (agent_type, agent) in enumerate(agents_to_show, 1):
        print(f"{i}. {agent_type}: {agent.name}")
        print(f"   Performance: {agent.performance_history[-1]:.3f}")
        print(f"   Description: {agent.description}")
        print()
    
    print("🔍 Agent code represents different problem-solving approaches.")
    print("Each agent's code reflects its strategy for tackling problems.")
    print()
    
    # Show code snippet from best agent
    best_agent = max(agents_to_show, key=lambda x: x[1].performance_history[-1])[1]
    print(f"📋 Code snippet from best agent ({best_agent.name}):")
    print("-" * 40)
    
    code_lines = best_agent.code.split('\n')
    for i, line in enumerate(code_lines[:8], 1):  # Show first 8 lines
        print(f"{i:2}: {line}")
    
    if len(code_lines) > 8:
        print(f"    ... ({len(code_lines) - 8} more lines)")
    
    print("-" * 40)
    print("🎯 This code represents the agent's problem-solving logic!")


if __name__ == "__main__":
    # Set random seed for reproducible demo
    random.seed(42)
    
    # Run the main demo
    quick_adas_demo()
    
    # Optional interactive inspector
    try:
        user_input = input("\n🤔 Want to inspect agent code in detail? (y/n): ").strip().lower()
        if user_input in ['y', 'yes']:
            interactive_agent_inspector()
        else:
            print("\n👋 Thanks for trying the ADAS demo!")
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Thanks for trying the ADAS demo!") 