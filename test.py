#!/usr/bin/env python3
"""
Sankoff's Algorithm - Comprehensive Test Suite
Tests all 4 phylogenetic trees from the problem set
"""

import numpy as np
from PhyloTree import create_tree_from_input
from SankoffCounting import SankoffAlgorithm


def test_tree_1():
    """
    Tree 1: Original example from lecture slides
    
    Tree Structure:
              root
             /    \
           v1      v2
          /  \    /  \
         A    C  T    G
    
    Cost matrix from image 1
    """
    print("\n" + "=" * 80)
    print("TREE 1: ORIGINAL LECTURE EXAMPLE")
    print("=" * 80)
    print("\nTree Structure:")
    print("              root")
    print("             /    \\")
    print("           v1      v2")
    print("          /  \\    /  \\")
    print("         A    C  T    G")
    print()
    
    alphabet = ['A', 'T', 'G', 'C']
    
    # Cost matrix from the image
    substitution_cost = np.array([
        [0, 3, 4, 9],  # A
        [3, 0, 2, 4],  # T
        [4, 2, 0, 4],  # G
        [9, 4, 4, 0]   # C
    ])
    
    print("Substitution Cost Matrix:")
    print("     A  T  G  C")
    for i, row_label in enumerate(alphabet):
        print(f"{row_label} {substitution_cost[i]}")
    print()
    
    # Leaf cost matrices
    leaf_cost_matrices = {
        0: {'A': 0, 'T': float('inf'), 'G': float('inf'), 'C': float('inf')},  # A
        1: {'A': float('inf'), 'T': float('inf'), 'G': float('inf'), 'C': 0},  # C
        2: {'A': float('inf'), 'T': 0, 'G': float('inf'), 'C': float('inf')},  # T
        3: {'A': float('inf'), 'T': float('inf'), 'G': 0, 'C': float('inf')}   # G
    }
    
    print("Leaf Observations:")
    print("  Node 0: A")
    print("  Node 1: C")
    print("  Node 2: T")
    print("  Node 3: G")
    print()
    
    # Tree edges (parent, child)
    edges = [
        (6, 4),  # root -> v1
        (6, 5),  # root -> v2
        (4, 0),  # v1 -> A
        (4, 1),  # v1 -> C
        (5, 2),  # v2 -> T
        (5, 3)   # v2 -> G
    ]
    
    # Create tree
    tree = create_tree_from_input(
        leaf_cost_matrices=leaf_cost_matrices,
        edges=edges,
        root_id=6,
        substitution_cost=substitution_cost,
        alphabet=alphabet
    )
    
    # Run Sankoff's algorithm
    sankoff = SankoffAlgorithm(tree)
    result = sankoff.run()
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Optimal Parsimony Score: {result['optimal_cost']}")
    print(f"Optimal Root State(s): {result['optimal_root_states']}")
    print(f"Total Optimal Solutions: {result['total_solutions']}")
    print(f"Solutions per root state: {result['solutions_per_root']}")
    print(f"Expected cost: 9 (from image)")
    print()
    
    return result


def test_tree_2():
    """
    Tree 2: Example from image 2
    
    Tree Structure:
           root
          /    \
        v1      leaf_t
       /  \
    leaf_c  leaf_g
    """
    print("\n" + "=" * 80)
    print("TREE 2: SIMPLE 3-LEAF EXAMPLE")
    print("=" * 80)
    print("\nTree Structure:")
    print("           root")
    print("          /    \\")
    print("        v1      t")
    print("       /  \\")
    print("      c    g")
    print()
    
    alphabet = ['a', 'g', 'c', 't']
    
    # Cost matrix from image 2
    substitution_cost = np.array([
        [0, 1, 3, 3],  # a
        [1, 0, 3, 3],  # g
        [3, 3, 0, 1],  # c
        [3, 3, 1, 0]   # t
    ])
    
    print("Substitution Cost Matrix:")
    print("     a  g  c  t")
    for i, row_label in enumerate(alphabet):
        print(f"{row_label} {substitution_cost[i]}")
    print()
    
    # Leaf cost matrices
    leaf_cost_matrices = {
        0: {'a': float('inf'), 'g': float('inf'), 'c': 0, 't': float('inf')},  # c
        1: {'a': float('inf'), 'g': 0, 'c': float('inf'), 't': float('inf')},  # g
        2: {'a': float('inf'), 'g': float('inf'), 'c': float('inf'), 't': 0}   # t
    }
    
    print("Leaf Observations:")
    print("  Node 0: c")
    print("  Node 1: g")
    print("  Node 2: t")
    print()
    
    # Tree edges
    edges = [
        (4, 3),  # root -> v1
        (4, 2),  # root -> t
        (3, 0),  # v1 -> c
        (3, 1)   # v1 -> g
    ]
    
    # Create tree
    tree = create_tree_from_input(
        leaf_cost_matrices=leaf_cost_matrices,
        edges=edges,
        root_id=4,
        substitution_cost=substitution_cost,
        alphabet=alphabet
    )
    
    # Run Sankoff's algorithm
    sankoff = SankoffAlgorithm(tree)
    result = sankoff.run()
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Optimal Parsimony Score: {result['optimal_cost']}")
    print(f"Optimal Root State(s): {result['optimal_root_states']}")
    print(f"Total Optimal Solutions: {result['total_solutions']}")
    print(f"Solutions per root state: {result['solutions_per_root']}")
    print(f"Expected cost: 4 or 5 (from image 2)")
    print()
    
    return result


def test_tree_3():
    """
    Tree 3: Unbalanced Tree
    
    Tree Structure:
              root
             /    \
           v1      C
          /  \
        A    v2
             / \
            G   T
    """
    print("\n" + "=" * 80)
    print("TREE 3: UNBALANCED TREE")
    print("=" * 80)
    print("\nTree Structure:")
    print("              root")
    print("             /    \\")
    print("           v1      C")
    print("          /  \\")
    print("        A    v2")
    print("             / \\")
    print("            G   T")
    print()
    
    alphabet = ['a', 'g', 'c', 't']
    
    substitution_cost = np.array([
        [0, 1, 2, 3],  # a
        [1, 0, 2, 3],  # g
        [2, 2, 0, 3],  # c
        [3, 3, 3, 0]   # t
    ])
    
    print("Substitution Cost Matrix:")
    print("     a  g  c  t")
    for i, row_label in enumerate(alphabet):
        print(f"{row_label} {substitution_cost[i]}")
    print()
    
    # Leaf cost matrices
    leaf_cost_matrices = {
        0: {'a': 0, 'g': float('inf'), 'c': float('inf'), 't': float('inf')},  # A
        1: {'a': float('inf'), 'g': 0, 'c': float('inf'), 't': float('inf')},  # G
        2: {'a': float('inf'), 'g': float('inf'), 'c': float('inf'), 't': 0},  # T
        3: {'a': float('inf'), 'g': float('inf'), 'c': 0, 't': float('inf')}   # C
    }
    
    print("Leaf Observations:")
    print("  Node 0 (A): a")
    print("  Node 1 (G): g")
    print("  Node 2 (T): t")
    print("  Node 3 (C): c")
    print()
    
    # Tree edges
    edges = [
        (6, 5),  # root -> v1
        (6, 3),  # root -> C
        (5, 0),  # v1 -> A
        (5, 4),  # v1 -> v2
        (4, 1),  # v2 -> G
        (4, 2)   # v2 -> T
    ]
    
    # Create tree
    tree = create_tree_from_input(
        leaf_cost_matrices=leaf_cost_matrices,
        edges=edges,
        root_id=6,
        substitution_cost=substitution_cost,
        alphabet=alphabet
    )
    
    # Run Sankoff's algorithm
    sankoff = SankoffAlgorithm(tree)
    result = sankoff.run()
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Optimal Parsimony Score: {result['optimal_cost']}")
    print(f"Optimal Root State(s): {result['optimal_root_states']}")
    print(f"Total Optimal Solutions: {result['total_solutions']}")
    print(f"Solutions per root state: {result['solutions_per_root']}")
    print(f"Expected cost: 6, Expected solutions: 6")
    print()
    
    return result


def test_tree_4():
    """
    Tree 4: Symmetric Balanced Tree
    
    Tree Structure:
             root
            /    \
          v1      v2
         /  \    /  \
        A    T  G    C
    """
    print("\n" + "=" * 80)
    print("TREE 4: SYMMETRIC BALANCED TREE")
    print("=" * 80)
    print("\nTree Structure:")
    print("             root")
    print("            /    \\")
    print("          v1      v2")
    print("         /  \\    /  \\")
    print("        A    T  G    C")
    print()
    
    alphabet = ['a', 'g', 'c', 't']
    
    # Symmetric cost matrix
    substitution_cost = np.array([
        [0, 2, 2, 2],  # a
        [2, 0, 2, 2],  # g
        [2, 2, 0, 2],  # c
        [2, 2, 2, 0]   # t
    ])
    
    print("Substitution Cost Matrix (Symmetric):")
    print("     a  g  c  t")
    for i, row_label in enumerate(alphabet):
        print(f"{row_label} {substitution_cost[i]}")
    print("(All non-identity transitions cost 2)")
    print()
    
    # Leaf cost matrices
    leaf_cost_matrices = {
        0: {'a': 0, 'g': float('inf'), 'c': float('inf'), 't': float('inf')},  # A
        1: {'a': float('inf'), 'g': float('inf'), 'c': float('inf'), 't': 0},  # T
        2: {'a': float('inf'), 'g': 0, 'c': float('inf'), 't': float('inf')},  # G
        3: {'a': float('inf'), 'g': float('inf'), 'c': 0, 't': float('inf')}   # C
    }
    
    print("Leaf Observations:")
    print("  Node 0 (A): a")
    print("  Node 1 (T): t")
    print("  Node 2 (G): g")
    print("  Node 3 (C): c")
    print()
    
    # Tree edges
    edges = [
        (6, 4),  # root -> v1
        (6, 5),  # root -> v2
        (4, 0),  # v1 -> A
        (4, 1),  # v1 -> T
        (5, 2),  # v2 -> G
        (5, 3)   # v2 -> C
    ]
    
    # Create tree
    tree = create_tree_from_input(
        leaf_cost_matrices=leaf_cost_matrices,
        edges=edges,
        root_id=6,
        substitution_cost=substitution_cost,
        alphabet=alphabet
    )
    
    # Run Sankoff's algorithm
    sankoff = SankoffAlgorithm(tree)
    result = sankoff.run()
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Optimal Parsimony Score: {result['optimal_cost']}")
    print(f"Optimal Root State(s): {result['optimal_root_states']}")
    print(f"Total Optimal Solutions: {result['total_solutions']}")
    print(f"Solutions per root state: {result['solutions_per_root']}")
    print(f"Expected cost: 6, Expected solutions: 12")
    print(f"(All 4 states should be optimal due to symmetry)")
    print()
    
    return result


def main():
    """Run all test trees"""
    print("\n" + "#" * 80)
    print("# SANKOFF'S ALGORITHM - COMPREHENSIVE TEST SUITE")
    print("# Testing 4 Different Phylogenetic Trees")
    print("#" * 80)
    
    results = []
    
    # Run all tests
    results.append(("Tree 1 (Lecture Example)", test_tree_1()))
    results.append(("Tree 2 (3-Leaf Example)", test_tree_2()))
    results.append(("Tree 3 (Unbalanced)", test_tree_3()))
    results.append(("Tree 4 (Symmetric)", test_tree_4()))
    
    # Summary
    print("\n" + "#" * 80)
    print("# SUMMARY OF ALL TESTS")
    print("#" * 80)
    print()
    
    for i, (name, result) in enumerate(results, 1):
        print(f"Test {i}: {name}")
        print(f"  Optimal Cost: {result['optimal_cost']}")
        print(f"  Optimal Root States: {result['optimal_root_states']}")
        print(f"  Total Solutions: {result['total_solutions']}")
        print(f"  Distribution: {result['solutions_per_root']}")
        print()
    
    print("=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()