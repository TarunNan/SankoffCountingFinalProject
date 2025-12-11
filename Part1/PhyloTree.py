import numpy as np
from typing import Dict, List, Optional, Tuple

class TreeNode:
    """A single node in our phylogenetic tree"""
    
    def __init__(self, node_id: int, observed_state: Optional[str] = None, cost_matrix: Optional[Dict[str, float]] = None):
        """
        node_id: unique ID for this node
        observed_state: the character we see here (only matters for leaves)
        cost_matrix: if we already know the costs, we can pass them in directly
        """
        self.id = node_id
        self.observed_state = observed_state
        self.children = []
        self.parent = None
        
        # stuff for Sankoff's algorithm
        self.cost_matrix = cost_matrix
        self.optimal_count = None
        self.backpointers = None
        
    def add_child(self, child_node):
        """Add a child to this node"""
        self.children.append(child_node)
        child_node.parent = self
        
    def is_leaf(self):
        """Leaves have no children"""
        return len(self.children) == 0
    
    def __repr__(self):
        state = self.observed_state if self.observed_state else "?"
        return f"Node({self.id}, {state})"


class PhylogeneticTree:
    """The whole tree structure - keeps track of all nodes and how they connect"""
    
    def __init__(self, alphabet: List[str] = ['A', 'C', 'G', 'T']):
        """
        alphabet: what characters are we working with (default is DNA bases)
        """
        self.alphabet = alphabet
        self.nodes = {}
        self.root = None
        self.substitution_cost = None  # 4x4 matrix we'll use for internal nodes
        
    def add_node(self, node_id: int, observed_state: Optional[str] = None, 
                 cost_matrix: Optional[Dict[str, float]] = None) -> TreeNode:
        """
        Make a new node and add it to our tree
        
        node_id: give it a unique number
        observed_state: what character is here (leave None for internal nodes)
        cost_matrix: pre-computed costs if we have them (usually for leaves)
        """
        node = TreeNode(node_id, observed_state, cost_matrix)
        self.nodes[node_id] = node
        return node
    
    def add_edge(self, parent_id: int, child_id: int):
        """Connect parent to child"""
        parent = self.nodes[parent_id]
        child = self.nodes[child_id]
        parent.add_child(child)
        
    def set_root(self, node_id: int):
        """Tell the tree which node is the root"""
        self.root = self.nodes[node_id]
        
    def set_cost_matrix(self, cost_matrix: np.ndarray):
        """Set up the substitution cost matrix (should be 4x4 for DNA)"""
        if cost_matrix.shape != (len(self.alphabet), len(self.alphabet)):
            raise ValueError(f"Cost matrix must be {len(self.alphabet)}x{len(self.alphabet)}")
        
        self.substitution_cost = cost_matrix
        
    def get_leaves(self) -> List[TreeNode]:
        """Get all leaf nodes"""
        return [node for node in self.nodes.values() if node.is_leaf()]
    
    def get_internal_nodes(self) -> List[TreeNode]:
        """Get all internal nodes"""
        return [node for node in self.nodes.values() if not node.is_leaf()]
    
    def postorder_traversal(self, node: Optional[TreeNode] = None) -> List[TreeNode]:
        """Traverse bottom-up (children before parents)"""
        if node is None:
            node = self.root
            
        result = []
        
        for child in node.children:
            result.extend(self.postorder_traversal(child))
        
        result.append(node)
        
        return result
    
    def __repr__(self):
        return f"PhylogeneticTree(nodes={len(self.nodes)}, leaves={len(self.get_leaves())})"


def create_tree_from_input(
    leaf_cost_matrices: Dict[int, Dict[str, float]],
    edges: List[Tuple[int, int]], 
    root_id: int,
    substitution_cost: np.ndarray,
    alphabet: List[str] = ['A', 'C', 'G', 'T']
) -> PhylogeneticTree:
    """
    Quick way to build a tree from the pieces we need
    
    leaf_cost_matrices: for each leaf, what are the costs? e.g. {0: {'A': 0, 'C': inf, ...}}
    edges: list of (parent, child) pairs
    root_id: which node is the root
    substitution_cost: the 4x4 cost matrix
    alphabet: what characters we're using
    """
    tree = PhylogeneticTree(alphabet)
    
    # figure out all the node IDs we need
    all_node_ids = set(leaf_cost_matrices.keys())
    for parent, child in edges:
        all_node_ids.add(parent)
        all_node_ids.add(child)
    
    # create all the nodes
    for node_id in all_node_ids:
        if node_id in leaf_cost_matrices:
            # it's a leaf - use the cost matrix we were given
            tree.add_node(node_id, cost_matrix=leaf_cost_matrices[node_id])
        else:
            # internal node - Sankoff will figure out the costs
            tree.add_node(node_id)
    
    # wire up the edges
    for parent_id, child_id in edges:
        tree.add_edge(parent_id, child_id)
    
    # finish setup
    tree.set_root(root_id)
    tree.set_cost_matrix(substitution_cost)
    
    return tree