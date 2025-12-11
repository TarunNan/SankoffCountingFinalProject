from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from PhyloTree import PhylogeneticTree, TreeNode


class SankoffAlgorithm:
    """Runs Sankoff's algorithm and counts how many optimal solutions there are"""
    
    def __init__(self, tree: PhylogeneticTree):
        """
        tree: should already have leaf cost matrices set up
        """
        self.tree = tree
        self.alphabet = tree.alphabet
        self.substitution_cost = tree.substitution_cost
        
        if self.substitution_cost is None:
            raise ValueError("Tree must have substitution cost matrix set!")
        
        # make sure all leaves have their costs
        for leaf in tree.get_leaves():
            if leaf.cost_matrix is None:
                raise ValueError(f"Leaf {leaf} doesn't have a cost_matrix!")
    
    def run(self) -> Dict:
        """
        Run the whole algorithm and return all the results
        """
        print("=" * 60)
        print("STARTING SANKOFF'S ALGORITHM")
        print("=" * 60)
        
        # go bottom-up first
        print("\n[STEP 1] Bottom-Up Pass (Computing Cost Matrices)")
        print("-" * 60)
        self._postorder_pass()
        
        # see what's optimal at the root
        print("\n[STEP 2] Finding Optimal Solution at Root")
        print("-" * 60)
        optimal_cost, optimal_root_states = self._find_optimal_at_root()
        
        # trace back down to assign states
        print("\n[STEP 3] Top-Down Pass (Traceback - Assigning States)")
        print("-" * 60)
        optimal_assignment = self._traceback(optimal_root_states[0])
        
        # count all the ways we can get an optimal solution
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
        Leaves already have their costs, we just need to compute internal nodes
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
        Figure out the cost matrix for an internal node
        
        For each state we could assign to this node, we look at all our children
        and find the cheapest way to connect to them. Then sum those up.
        """
        node.cost_matrix = {}
        
        for parent_state in self.alphabet:
            total = 0
            
            for child in node.children:
                # find the cheapest child state for this parent state
                best_cost = float('inf')
                
                for child_state in self.alphabet:
                    p_idx = self.alphabet.index(parent_state)
                    c_idx = self.alphabet.index(child_state)
                    
                    cost = self.substitution_cost[p_idx][c_idx] + child.cost_matrix[child_state]
                    
                    if cost < best_cost:
                        best_cost = cost
                
                total += best_cost
            
            node.cost_matrix[parent_state] = total
    
    def _find_optimal_at_root(self) -> Tuple[float, List[str]]:
        """
        Look at the root's cost matrix and find the minimum cost
        Also track which state(s) achieve that minimum (there might be ties)
        """
        root = self.tree.root
        
        optimal_cost = min(root.cost_matrix.values())
        
        # collect all states that hit the minimum
        optimal_states = [state for state, cost in root.cost_matrix.items() 
                         if cost == optimal_cost]
        
        print(f"Root (Node {root.id}) cost matrix: {root.cost_matrix}")
        print(f"Optimal cost: {optimal_cost}")
        print(f"Optimal state(s) at root: {optimal_states}")
        
        return optimal_cost, optimal_states
    
    def _traceback(self, root_state: str) -> Dict[int, str]:
        """
        Top-down: start at root with chosen state, then pick optimal states for children
        """
        print(f"Starting traceback from root with state '{root_state}'")
        assignment = {}
        self._traceback_recursive(self.tree.root, root_state, assignment, depth=0)
        return assignment
    
    def _traceback_recursive(self, node: TreeNode, assigned_state: str, 
                            assignment: Dict[int, str], depth: int):
        """
        Recursively assign states going down the tree
        """
        indent = "  " * depth
        
        assignment[node.id] = assigned_state
        print(f"{indent}Node {node.id}: assigned state '{assigned_state}'")
        
        if node.is_leaf():
            return
        
        # for each child, pick the state that minimizes cost
        for child in node.children:
            best_state = None
            best_cost = float('inf')
            
            for child_state in self.alphabet:
                p_idx = self.alphabet.index(assigned_state)
                c_idx = self.alphabet.index(child_state)
                
                cost = self.substitution_cost[p_idx][c_idx] + child.cost_matrix[child_state]
                
                if cost < best_cost:
                    best_cost = cost
                    best_state = child_state
            
            self._traceback_recursive(child, best_state, assignment, depth + 1)
    
    def _count_all_solutions(self, optimal_root_states: List[str]) -> Tuple[int, Dict[str, int]]:
        """
        Count every possible optimal assignment
        
        Key idea: if a node has k children and each child has some number of optimal choices,
        we multiply them together (that's the multiplicative principle)
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
        Count solutions from this subtree given the state we assigned to this node
        
        Leaf = 1 solution (it is what it is)
        Internal = multiply together the solution counts from each child
        """
        if node.is_leaf():
            return 1
        
        total = 1
        
        for child in node.children:
            # find ALL optimal child states (not just one - there could be ties!)
            optimal_child_states = []
            min_cost = float('inf')
            
            for child_state in self.alphabet:
                p_idx = self.alphabet.index(assigned_state)
                c_idx = self.alphabet.index(child_state)
                
                cost = self.substitution_cost[p_idx][c_idx] + child.cost_matrix[child_state]
                
                if cost < min_cost:
                    min_cost = cost
                    optimal_child_states = [child_state]
                elif cost == min_cost:
                    optimal_child_states.append(child_state)
            
            # sum up solutions from each optimal child state
            child_solutions = 0
            for child_state in optimal_child_states:
                child_solutions += self._count_solutions_recursive(child, child_state)
            
            # multiply (that's where the counting magic happens)
            total *= child_solutions
        
        return total