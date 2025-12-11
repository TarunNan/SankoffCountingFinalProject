"""
MigrationParser.py

Parses McPherson et al. patient data files and builds a PhylogeneticTree
for migration analysis using Sankoff's algorithm.

Input file formats:
- patient.tree: parent child (one edge per line)
- patient.labeling: node location (leaf node locations)
- patient.reported.labeling: node location (published solution for verification)
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from PhyloTree import PhylogeneticTree, TreeNode


class MigrationParser:
    """Reads migration data files and builds the tree for Sankoff's algorithm"""
    
    def __init__(self, tree_file: str, labeling_file: str):
        """
        tree_file: path to .tree file (parent-child edges)
        labeling_file: path to .labeling file (leaf locations)
        """
        self.tree_file = tree_file
        self.labeling_file = labeling_file
        
        self.edges = []
        self.leaf_locations = {}
        self.all_nodes = set()
        self.alphabet = set()
        
    def parse(self) -> Tuple[List[Tuple[str, str]], Dict[str, str], List[str]]:
        """
        Parse both files and return the data we need
        
        Returns:
            (edges, leaf_locations, alphabet)
        """
        print("=" * 60)
        print("PARSING MIGRATION DATA FILES")
        print("=" * 60)
        
        self._parse_tree_file()
        self._parse_labeling_file()
        
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
        """Read the .tree file to get parent-child edges"""
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
        """Read the .labeling file to get leaf locations"""
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
        Build a PhylogeneticTree ready for Sankoff's algorithm
        
        Sets up nodes, edges, leaf cost matrices, and the substitution
        cost matrix (uniform: 0 on diagonal, 1 elsewhere)
        """
        edges, leaf_locations, alphabet = self.parse()
        
        print("\n" + "=" * 60)
        print("CREATING PHYLOGENETIC TREE")
        print("=" * 60)
        
        tree = PhylogeneticTree(alphabet=alphabet)
        
        leaf_nodes = set(leaf_locations.keys())
        internal_nodes = self.all_nodes - leaf_nodes
        
        print(f"\n[1] Creating nodes...")
        print(f"  Internal nodes: {sorted(internal_nodes)}")
        print(f"  Leaf nodes: {sorted(leaf_nodes)}")
        
        for node_name in self.all_nodes:
            if node_name in leaf_locations:
                # leaf: cost is 0 for observed location, inf for everything else
                location = leaf_locations[node_name]
                cost_matrix = {
                    loc: 0.0 if loc == location else float('inf')
                    for loc in alphabet
                }
                tree.add_node(node_name, cost_matrix=cost_matrix)
                print(f"    Leaf {node_name}: location={location}")
            else:
                # internal: Sankoff will compute the costs
                tree.add_node(node_name)
                print(f"    Internal {node_name}")
        
        print(f"\n[2] Adding edges...")
        for parent, child in edges:
            tree.add_edge(parent, child)
        print(f"  Added {len(edges)} edges")
        
        # find root (the node that's never a child)
        print(f"\n[3] Identifying root...")
        children = set(child for _, child in edges)
        parents = set(parent for parent, _ in edges)
        roots = parents - children
        
        if len(roots) != 1:
            raise ValueError(f"Expected exactly 1 root, found {len(roots)}: {roots}")
        
        root = list(roots)[0]
        tree.set_root(root)
        print(f"  Root node: {root}")
        
        # migration costs: 0 to stay, 1 to move
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
    Load the published solution from McPherson et al. so we can compare
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