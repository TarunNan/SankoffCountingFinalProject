from __future__ import annotations  # <-- ADD THIS AS LINE 1

import numpy as np
from typing import Dict, List, Optional, Tuple
from PhyloTree import PhylogeneticTree, TreeNode


class SankoffAlgorithm:
    """Implements Sankoff's Algorithm for ancestral state reconstruction with solution counting"""
    
    def __init__(self, tree: PhylogeneticTree):
        """
        Args:
            tree: PhylogeneticTree with leaf cost matrices already set
        """
        self.tree = tree
        self.alphabet = tree.alphabet
        self.substitution_cost = tree.substitution_cost
        
        if self.substitution_cost is None:
            raise ValueError("Tree must have substitution cost matrix set!")
        
        # Verify all leaves have cost matrices
        for leaf in tree.get_leaves():
            if leaf.cost_matrix is None:
                raise ValueError(f"Leaf {leaf} does not have cost_matrix initialized!")
    
    def run(self) -> Dict:
        """
        Run complete Sankoff's algorithm with solution counting
        
        Returns:
            Dictionary with results including solution count
        """
        print("=" * 60)
        print("STARTING SANKOFF'S ALGORITHM")
        print("=" * 60)
        
        # Step 1: Bottom-up pass - compute cost matrices from leaves to root
        print("\n[STEP 1] Bottom-Up Pass (Computing Cost Matrices)")
        print("-" * 60)
        self._postorder_pass()
        
        # Step 2: Find optimal cost and state(s) at root
        print("\n[STEP 2] Finding Optimal Solution at Root")
        print("-" * 60)
        optimal_cost, optimal_root_states = self._find_optimal_at_root()
        
        # Step 3: Top-down pass - assign states (traceback)
        print("\n[STEP 3] Top-Down Pass (Traceback - Assigning States)")
        print("-" * 60)
        optimal_assignment = self._traceback(optimal_root_states[0])
        
        # Step 4: Count all optimal solutions
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
        Bottom-up pass: Process nodes from leaves to root
        
        For each node in post-order (children before parents):
        - Leaves: Already have cost matrices (from input)
        - Internal nodes: Compute cost matrix based on children
        """
        # Get nodes in post-order traversal
        nodes_to_process = self.tree.postorder_traversal()
        
        print(f"Processing {len(nodes_to_process)} nodes in post-order:")
        
        for i, node in enumerate(nodes_to_process):
            if node.is_leaf():
                # Leaf node - cost matrix already set from input
                print(f"  [{i+1}] Node {node.id} (LEAF) - cost matrix already set")
                print(f"       Cost matrix: {node.cost_matrix}")
            else:
                # Internal node - compute cost matrix
                print(f"  [{i+1}] Node {node.id} (INTERNAL) - computing cost matrix...")
                self._compute_node_costs(node)
                print(f"       Cost matrix: {node.cost_matrix}")
    
    def _compute_node_costs(self, node: TreeNode):
        """
        Compute cost matrix for an internal node
        
        Algorithm:
        For each possible state 'i' at this node:
            total_cost = 0
            For each child:
                Find minimum over all child states 'j':
                    cost = substitution_cost[i][j] + child.cost_matrix[j]
                total_cost += minimum_cost_from_this_child
            node.cost_matrix[i] = total_cost
        
        Args:
            node: Internal node to compute costs for
        """
        node.cost_matrix = {}
        
        # For each possible state at THIS node
        for parent_state in self.alphabet:
            total_cost = 0  # Sum of costs from all children
            
            # For each child of this node
            for child in node.children:
                min_child_cost = float('inf')
                best_child_state = None
                
                # Find minimum cost over all possible child states
                for child_state in self.alphabet:
                    # Get index for substitution cost lookup
                    parent_idx = self.alphabet.index(parent_state)
                    child_idx = self.alphabet.index(child_state)
                    
                    # Cost = substitution cost + child's cost for that state
                    cost = (self.substitution_cost[parent_idx][child_idx] + 
                           child.cost_matrix[child_state])
                    
                    # Track minimum
                    if cost < min_child_cost:
                        min_child_cost = cost
                        best_child_state = child_state
                
                # Add this child's minimum cost to total
                total_cost += min_child_cost
            
            # Store total cost for this parent state
            node.cost_matrix[parent_state] = total_cost
    
    def _find_optimal_at_root(self) -> Tuple[float, List[str]]:
        """
        Find the optimal cost and which state(s) achieve it at the root
        
        Returns:
            (optimal_cost, list_of_optimal_states)
        """
        root = self.tree.root
        
        # Find minimum cost
        optimal_cost = min(root.cost_matrix.values())
        
        # Find all states that achieve this minimum cost
        optimal_states = [state for state, cost in root.cost_matrix.items() 
                         if cost == optimal_cost]
        
        print(f"Root (Node {root.id}) cost matrix: {root.cost_matrix}")
        print(f"Optimal cost: {optimal_cost}")
        print(f"Optimal state(s) at root: {optimal_states}")
        
        return optimal_cost, optimal_states
    
    def _traceback(self, root_state: str) -> Dict[int, str]:
        """
        Top-down pass: Assign states to all nodes
        
        Start at root with chosen optimal state, then recursively
        assign states to children by choosing child states that
        minimize the cost
        
        Args:
            root_state: The state to assign to the root
            
        Returns:
            Dictionary mapping node_id -> assigned_state
        """
        print(f"Starting traceback from root with state '{root_state}'")
        assignment = {}
        self._traceback_recursive(self.tree.root, root_state, assignment, depth=0)
        return assignment
    
    def _traceback_recursive(self, node: TreeNode, assigned_state: str, 
                            assignment: Dict[int, str], depth: int):
        """
        Recursive traceback to assign states
        
        Args:
            node: Current node
            assigned_state: State we're assigning to this node
            assignment: Dictionary to fill with assignments
            depth: Depth for printing (visualization)
        """
        indent = "  " * depth
        
        # Assign state to this node
        assignment[node.id] = assigned_state
        print(f"{indent}Node {node.id}: assigned state '{assigned_state}'")
        
        if node.is_leaf():
            # Base case - reached a leaf
            return
        
        # For each child, find which child state was optimal
        for child in node.children:
            best_child_state = None
            min_cost = float('inf')
            
            # Try all possible child states
            for child_state in self.alphabet:
                # Get indices for cost lookup
                parent_idx = self.alphabet.index(assigned_state)
                child_idx = self.alphabet.index(child_state)
                
                # Calculate total cost for this child state
                cost = (self.substitution_cost[parent_idx][child_idx] + 
                       child.cost_matrix[child_state])
                
                # Track minimum
                if cost < min_cost:
                    min_cost = cost
                    best_child_state = child_state
            
            # Recursively assign to child
            self._traceback_recursive(child, best_child_state, assignment, depth + 1)
    
    def _count_all_solutions(self, optimal_root_states: List[str]) -> Tuple[int, Dict[str, int]]:
        """
        Count ALL optimal solutions by exploring all optimal paths
        
        Key insight: Use the multiplicative principle
        - For a node with k children
        - If child i has n_i optimal state choices
        - Total solutions from this node = n_1 × n_2 × ... × n_k
        
        Args:
            optimal_root_states: List of states that are optimal at root
            
        Returns:
            (total_solutions, solutions_per_root_state)
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
        Recursively count solutions from a node with assigned state
        
        This implements the multiplicative principle:
        - Base case: Leaf node contributes 1 solution
        - Recursive case: For each child, find all optimal states
        - Multiply the number of solutions from each child
        
        Args:
            node: Current node
            assigned_state: State assigned to this node
            
        Returns:
            Number of optimal solutions from this subtree
        """
        # Base case: leaf node
        if node.is_leaf():
            return 1
        
        # For internal nodes, multiply solutions from all children
        total_solutions = 1
        
        for child in node.children:
            # Find ALL optimal child states (not just one!)
            optimal_child_states = []
            min_cost = float('inf')
            
            # Try all possible child states
            for child_state in self.alphabet:
                parent_idx = self.alphabet.index(assigned_state)
                child_idx = self.alphabet.index(child_state)
                
                cost = (self.substitution_cost[parent_idx][child_idx] + 
                       child.cost_matrix[child_state])
                
                # Track all states that achieve minimum
                if cost < min_cost:
                    min_cost = cost
                    optimal_child_states = [child_state]
                elif cost == min_cost:
                    optimal_child_states.append(child_state)
            
            # Count solutions for each optimal child state
            child_solutions = 0
            for child_state in optimal_child_states:
                child_solutions += self._count_solutions_recursive(child, child_state)
            
            # Multiply by this child's contribution (multiplicative principle)
            total_solutions *= child_solutions
        
        return total_solutions