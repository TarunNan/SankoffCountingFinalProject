from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from PhyloTree import PhylogeneticTree, TreeNode


class SankoffAlgorithm:
    """Sankoff's algorithm with solution counting (ovarian cancer version)"""
    
    def __init__(self, tree: PhylogeneticTree):
        """
        tree: should have leaf cost matrices already set
        """
        self.tree = tree
        self.alphabet = tree.alphabet
        self.substitution_cost = tree.substitution_cost
        
        if self.substitution_cost is None:
            raise ValueError("Tree must have substitution cost matrix set!")
        
        for leaf in tree.get_leaves():
            if leaf.cost_matrix is None:
                raise ValueError(f"Leaf {leaf} doesn't have a cost_matrix!")
    
    def run(self) -> Dict:
        """
        Run the full algorithm and return results
        """
        print("=" * 60)
        print("STARTING SANKOFF'S ALGORITHM")
        print("=" * 60)
        
        print("\n[STEP 1] Bottom-Up Pass (Computing Cost Matrices)")
        print("-" * 60)
        self._postorder_pass()
        
        print("\n[STEP 2] Finding Optimal Solution at Root")
        print("-" * 60)
        optimal_cost, optimal_root_states = self._find_optimal_at_root()
        
        print("\n[STEP 3] Top-Down Pass (Traceback - Assigning States)")
        print("-" * 60)
        optimal_assignment = self._traceback(optimal_root_states[0])
        
        print("\n[STEP 4] Counting All Optimal Solutions")
        print("-" * 60)
        total_solutions, solutions_per_root = self._count_all_solutions(optimal_root_states)
        
        print("\n" + "=" * 60)
        print("SANKOFF'S ALGORITHM COMPLETE")
        print("=" * 60)
        print(f"\n*** TOTAL OPTIMAL SOLUTIONS: {total_solutions} ***\n")
        
        return {
            'optimal_cost': optimal_cost,
            'optimal_root_states': optimal_root_states,
            'assignment': optimal_assignment,
            'total_solutions': total_solutions,
            'solutions_per_root': solutions_per_root
        }
    
    def _postorder_pass(self):
        """
        Bottom-up: process children before parents
        Leaves already have costs set, we compute internal nodes here
        """
        nodes_to_process = self.tree.postorder_traversal()
        
        print(f"Processing {len(nodes_to_process)} nodes in post-order:")
        
        for i, node in enumerate(nodes_to_process):
            if node.is_leaf():
                print(f"  [{i+1}] Node {node.id} (LEAF) - cost matrix already set")
                print(f"       Cost matrix: {node.cost_matrix}")
            else:
                print(f"  [{i+1}] Node {node.id} (INTERNAL) - computing cost matrix...")
                self._compute_node_costs(node)
                print(f"       Cost matrix: {node.cost_matrix}")
    
    def _compute_node_costs(self, node: TreeNode):
        """
        Compute cost matrix for an internal node
        
        For each state we could assign here, sum up the minimum cost
        to connect to each child (considering substitution costs)
        """
        node.cost_matrix = {}
        
        for parent_state in self.alphabet:
            total_cost = 0
            
            for child in node.children:
                min_child_cost = float('inf')
                
                for child_state in self.alphabet:
                    parent_idx = self.alphabet.index(parent_state)
                    child_idx = self.alphabet.index(child_state)
                    
                    cost = (self.substitution_cost[parent_idx][child_idx] + 
                           child.cost_matrix[child_state])
                    
                    if cost < min_child_cost:
                        min_child_cost = cost
                
                total_cost += min_child_cost
            
            node.cost_matrix[parent_state] = total_cost
    
    def _find_optimal_at_root(self) -> Tuple[float, List[str]]:
        """
        Find optimal cost at root, but constrained to ovarian sites only
        (LOv, ROv, or RUt since cancer must originate there)
        """
        root = self.tree.root
        
        # biologically, the tumor has to start at one of these sites
        allowed_root_states = {'LOv', 'ROv', 'RUt'}
        
        # only consider allowed states
        constrained_costs = {
            state: cost 
            for state, cost in root.cost_matrix.items() 
            if state in allowed_root_states
        }
        
        if not constrained_costs:
            raise ValueError(
                f"Constraint Error: None of {allowed_root_states} found in root costs. "
                f"Available states: {list(root.cost_matrix.keys())}"
            )

        optimal_cost = min(constrained_costs.values())
        
        optimal_states = [
            state for state, cost in constrained_costs.items() 
            if cost == optimal_cost
        ]
        
        print(f"Root (Node {root.id}) full cost matrix: {root.cost_matrix}")
        print(f"*** CONSTRAINT APPLIED: Root must be {allowed_root_states} ***")
        print(f"Optimal cost (constrained): {optimal_cost}")
        print(f"Optimal state(s) at root: {optimal_states}")
        
        return optimal_cost, optimal_states
    
    def _traceback(self, root_state: str) -> Dict[int, str]:
        """
        Top-down: assign states starting from root
        """
        print(f"Starting traceback from root with state '{root_state}'")
        assignment = {}
        self._traceback_recursive(self.tree.root, root_state, assignment, depth=0)
        return assignment
    
    def _traceback_recursive(self, node: TreeNode, assigned_state: str, 
                            assignment: Dict[int, str], depth: int):
        """
        Recursively assign states going down the tree
        Pick the child state that gives minimum cost at each step
        """
        indent = "  " * depth
        
        assignment[node.id] = assigned_state
        print(f"{indent}Node {node.id}: assigned state '{assigned_state}'")
        
        if node.is_leaf():
            return
        
        for child in node.children:
            best_child_state = None
            min_cost = float('inf')
            
            for child_state in self.alphabet:
                parent_idx = self.alphabet.index(assigned_state)
                child_idx = self.alphabet.index(child_state)
                
                cost = (self.substitution_cost[parent_idx][child_idx] + 
                       child.cost_matrix[child_state])
                
                if cost < min_cost:
                    min_cost = cost
                    best_child_state = child_state
            
            self._traceback_recursive(child, best_child_state, assignment, depth + 1)
    
    def _count_all_solutions(self, optimal_root_states: List[str]) -> Tuple[int, Dict[str, int]]:
        """
        Count all optimal solutions using the multiplicative principle
        
        If a node has k children and child i has n_i optimal choices,
        total solutions = n_1 * n_2 * ... * n_k
        """
        solutions_per_root = {}
        total_solutions = 0
        
        print(f"Counting solutions for {len(optimal_root_states)} optimal root state(s):")
        
        for root_state in optimal_root_states:
            count = self._count_solutions_recursive(self.tree.root, root_state)
            solutions_per_root[root_state] = count
            total_solutions += count
            print(f"  From root state '{root_state}': {count} solution(s)")
        
        return total_solutions, solutions_per_root
    
    def _count_solutions_recursive(self, node: TreeNode, assigned_state: str) -> int:
        """
        Recursively count solutions from this subtree
        
        Leaf = 1 solution
        Internal = multiply solution counts from each child
        """
        if node.is_leaf():
            return 1
        
        total_solutions = 1
        
        for child in node.children:
            # find ALL optimal child states (there may be ties)
            optimal_child_states = []
            min_cost = float('inf')
            
            for child_state in self.alphabet:
                parent_idx = self.alphabet.index(assigned_state)
                child_idx = self.alphabet.index(child_state)
                
                cost = (self.substitution_cost[parent_idx][child_idx] + 
                       child.cost_matrix[child_state])
                
                if cost < min_cost:
                    min_cost = cost
                    optimal_child_states = [child_state]
                elif cost == min_cost:
                    optimal_child_states.append(child_state)
            
            # sum solutions across all tied optimal states
            child_solutions = 0
            for child_state in optimal_child_states:
                child_solutions += self._count_solutions_recursive(child, child_state)
            
            total_solutions *= child_solutions
        
        return total_solutions