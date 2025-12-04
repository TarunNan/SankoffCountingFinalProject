"""
MigrationParser.py

Parses McPherson et al. patient data files and creates PhylogeneticTree
for migration analysis using Sankoff's algorithm.

Input files format:
- patient.tree: parent child (one edge per line)
- patient.labeling: node location (leaf node locations)
- patient.reported.labeling: node location (published solution for verification)
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from PhyloTree import PhylogeneticTree, TreeNode


class MigrationParser:
    """Parse migration data files and create tree for Sankoff's algorithm"""
    
    def __init__(self, tree_file: str, labeling_file: str):
        """
        Args:
            tree_file: Path to .tree file (parent-child edges)
            labeling_file: Path to .labeling file (leaf locations)
        """
        self.tree_file = tree_file
        self.labeling_file = labeling_file
        
        # Data structures to fill
        self.edges = []  # List of (parent, child) tuples
        self.leaf_locations = {}  # node_name -> location
        self.all_nodes = set()  # All node names
        self.alphabet = set()  # All unique locations
        
    def parse(self) -> Tuple[List[Tuple[str, str]], Dict[str, str], List[str]]:
        """
        Parse the tree and labeling files
        
        Returns:
            (edges, leaf_locations, alphabet)
            - edges: List of (parent, child) tuples
            - leaf_locations: Dict mapping leaf node -> location
            - alphabet: List of all unique anatomical locations
        """
        print("=" * 60)
        print("PARSING MIGRATION DATA FILES")
        print("=" * 60)
        
        # Parse tree structure
        self._parse_tree_file()
        
        # Parse leaf locations
        self._parse_labeling_file()
        
        # Create alphabet from unique locations
        alphabet = sorted(list(self.alphabet))
        
        print(f"\n[SUMMARY]")
        print(f"  Total nodes: {len(self.all_nodes)}")
        print(f"  Edges: {len(self.edges)}")
        print(f"  Leaf nodes: {len(self.leaf_locations)}")
        print(f"  Internal nodes: {len(self.all_nodes) - len(self.leaf_locations)}")
        print(f"  Anatomical locations (alphabet): {alphabet}")
        print("=" * 60)
        
        return self.edges, self.leaf_locations, alphabet
    
    def _parse_tree_file(self):
        """Parse the .tree file to get parent-child edges"""
        print(f"\n[1] Parsing tree structure from: {self.tree_file}")
        
        with open(self.tree_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid edge format: {line}")
                
                parent, child = parts[0], parts[1]
                self.edges.append((parent, child))
                self.all_nodes.add(parent)
                self.all_nodes.add(child)
        
        print(f"  Found {len(self.edges)} edges")
        print(f"  Found {len(self.all_nodes)} unique nodes")
    
    def _parse_labeling_file(self):
        """Parse the .labeling file to get leaf locations"""
        print(f"\n[2] Parsing leaf locations from: {self.labeling_file}")
        
        with open(self.labeling_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid labeling format: {line}")
                
                node, location = parts[0], parts[1]
                self.leaf_locations[node] = location
                self.alphabet.add(location)
        
        print(f"  Found {len(self.leaf_locations)} leaf locations")
        print(f"  Found {len(self.alphabet)} unique anatomical locations")
    
    def create_phylogenetic_tree(self) -> PhylogeneticTree:
        """
        Create a PhylogeneticTree object ready for Sankoff's algorithm
        
        Returns:
            PhylogeneticTree with:
            - All nodes created
            - Edges connected
            - Leaf cost matrices initialized
            - Substitution cost matrix set (uniform: 0 diagonal, 1 off-diagonal)
        """
        edges, leaf_locations, alphabet = self.parse()
        
        print("\n" + "=" * 60)
        print("CREATING PHYLOGENETIC TREE")
        print("=" * 60)
        
        # Create tree
        tree = PhylogeneticTree(alphabet=alphabet)
        
        # Identify internal nodes (appear as parents but not in leaf_locations)
        leaf_nodes = set(leaf_locations.keys())
        internal_nodes = self.all_nodes - leaf_nodes
        
        print(f"\n[1] Creating nodes...")
        print(f"  Internal nodes: {sorted(internal_nodes)}")
        print(f"  Leaf nodes: {sorted(leaf_nodes)}")
        
        # Create all nodes
        for node_name in self.all_nodes:
            if node_name in leaf_locations:
                # Leaf node - create cost matrix that constrains to observed location
                location = leaf_locations[node_name]
                cost_matrix = {
                    loc: 0.0 if loc == location else float('inf')
                    for loc in alphabet
                }
                tree.add_node(node_name, cost_matrix=cost_matrix)
                print(f"    Leaf {node_name}: location={location}")
            else:
                # Internal node - cost matrix will be computed by Sankoff
                tree.add_node(node_name)
                print(f"    Internal {node_name}")
        
        # Add edges
        print(f"\n[2] Adding edges...")
        for parent, child in edges:
            tree.add_edge(parent, child)
        print(f"  Added {len(edges)} edges")
        
        # Find root (node that is never a child)
        print(f"\n[3] Identifying root...")
        children = set(child for _, child in edges)
        parents = set(parent for parent, _ in edges)
        roots = parents - children
        
        if len(roots) != 1:
            raise ValueError(f"Expected exactly 1 root, found {len(roots)}: {roots}")
        
        root = list(roots)[0]
        tree.set_root(root)
        print(f"  Root node: {root}")
        
        # Create substitution cost matrix
        # Migration cost: 0 for staying at same location, 1 for any migration
        print(f"\n[4] Creating substitution cost matrix...")
        n = len(alphabet)
        substitution_cost = np.ones((n, n)) - np.eye(n)
        tree.set_cost_matrix(substitution_cost)
        print(f"  Matrix shape: {n}x{n}")
        print(f"  Migration cost: 0 (same location), 1 (any migration)")
        
        print("=" * 60)
        print("TREE CREATION COMPLETE")
        print("=" * 60)
        
        return tree


def load_reported_solution(reported_file: str) -> Dict[str, str]:
    """
    Load the reported solution from McPherson et al. for comparison
    
    Args:
        reported_file: Path to .reported.labeling file
        
    Returns:
        Dictionary mapping node -> location
    """
    reported = {}
    with open(reported_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                node, location = parts[0], parts[1]
                reported[node] = location
    return reported
