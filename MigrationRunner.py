"""
MigrationRunner.py

Main script to run Sankoff's algorithm on McPherson et al. migration data.
Compares results with the reported solution.
"""

from MigrationParser import MigrationParser, load_reported_solution
from SankoffCountingOvarian import SankoffAlgorithm


def compare_solutions(computed: dict, reported: dict) -> bool:
    """
    Compare computed solution with reported solution
    
    Args:
        computed: Dictionary from Sankoff's algorithm (node -> location)
        reported: Dictionary from paper (node -> location)
        
    Returns:
        True if solutions match, False otherwise
    """
    print("\n" + "=" * 60)
    print("COMPARING WITH REPORTED SOLUTION")
    print("=" * 60)
    
    # Get all nodes
    all_nodes = sorted(set(computed.keys()) | set(reported.keys()))
    
    matches = 0
    mismatches = 0
    
    print(f"\n{'Node':<10} {'Computed':<10} {'Reported':<10} {'Match'}")
    print("-" * 50)
    
    for node in all_nodes:
        comp_loc = computed.get(node, "MISSING")
        rep_loc = reported.get(node, "MISSING")
        match = "✓" if comp_loc == rep_loc else "✗"
        
        if comp_loc == rep_loc:
            matches += 1
        else:
            mismatches += 1
        
        print(f"{node:<10} {comp_loc:<10} {rep_loc:<10} {match}")
    
    print("-" * 50)
    print(f"Matches: {matches}/{len(all_nodes)}")
    print(f"Mismatches: {mismatches}/{len(all_nodes)}")
    
    if mismatches == 0:
        print("\n✓✓✓ PERFECT MATCH! ✓✓✓")
    else:
        print(f"\n✗ {mismatches} mismatches found")
    
    print("=" * 60)
    
    return mismatches == 0


def count_migrations(assignment: dict, tree_edges: list) -> int:
    """
    Count the total number of migrations (location changes) along edges
    
    Args:
        assignment: Dictionary mapping node -> location
        tree_edges: List of (parent, child) tuples
        
    Returns:
        Total number of migrations
    """
    migrations = 0
    migration_edges = []
    
    for parent, child in tree_edges:
        parent_loc = assignment.get(parent)
        child_loc = assignment.get(child)
        
        if parent_loc != child_loc:
            migrations += 1
            migration_edges.append((parent, child, parent_loc, child_loc))
    
    return migrations, migration_edges


def print_migration_summary(assignment: dict, tree_edges: list, optimal_cost: float):
    """Print summary of migrations"""
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    
    migrations, migration_edges = count_migrations(assignment, tree_edges)
    
    print(f"\nOptimal cost (from Sankoff): {optimal_cost}")
    print(f"Total migrations (edge count): {migrations}")
    print(f"\nMigration events:")
    
    for parent, child, parent_loc, child_loc in migration_edges:
        print(f"  {parent}({parent_loc}) → {child}({child_loc})")
    
    print("=" * 60)


def run_migration_analysis(patient_prefix: str):
    """
    Complete analysis pipeline for a patient
    
    Args:
        patient_prefix: Prefix for patient files (e.g., "patient1")
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print(f"║ MIGRATION ANALYSIS: {patient_prefix:<40} ║")
    print("╚" + "=" * 58 + "╝")
    
    # File paths
    tree_file = f"{patient_prefix}.tree"
    labeling_file = f"{patient_prefix}.labeling"
    reported_file = f"{patient_prefix}.reported.labeling"
    
    # Step 1: Parse data and create tree
    parser = MigrationParser(tree_file, labeling_file)
    tree = parser.create_phylogenetic_tree()
    
    # Step 2: Run Sankoff's algorithm
    sankoff = SankoffAlgorithm(tree)
    results = sankoff.run()
    
    # Step 3: Extract results
    optimal_cost = results['optimal_cost']
    assignment = results['assignment']
    total_solutions = results['total_solutions']
    
    # Step 4: Print migration summary
    print_migration_summary(assignment, parser.edges, optimal_cost)
    
    # Step 5: Compare with reported solution
    reported = load_reported_solution(reported_file)
    matches = compare_solutions(assignment, reported)
    
    # Step 6: Final summary
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║ FINAL RESULTS" + " " * 44 + "║")
    print("╚" + "=" * 58 + "╝")
    print(f"  Optimal migration cost: {optimal_cost}")
    print(f"  Total optimal solutions: {total_solutions}")
    print(f"  Match with reported: {'YES ✓' if matches else 'NO ✗'}")
    print("\n")
    
    return results, matches


if __name__ == "__main__":
    # Run analysis for patient 1
    results, matches = run_migration_analysis("patient10")
    
    # Additional analysis if there are multiple solutions
    if results['total_solutions'] > 1:
        print("\n" + "!" * 60)
        print(f"NOTE: Found {results['total_solutions']} optimal solutions!")
        print("The traceback returned one arbitrary solution.")
        print("Different optimal solutions may have the same cost")
        print("but different internal node labelings.")
        print("!" * 60)
