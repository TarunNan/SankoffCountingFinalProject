import numpy as np
from typing import Dict, List, Optional, Tuple

class TreeNode:
    """Represents a node in the phylogenetic tree"""
    
    def __init__(self, node_id: int, observed_state: Optional[str] = None, cost_matrix: Optional[Dict[str, float]] = None):
        """
        Args:
            node_id: Unique identifier for this node
            observed_state: Character at this node (only for leaves, None for internal nodes)
            cost_matrix: Pre-initialized cost matrix (typically for leaf nodes)
        """
        self.id = node_id
        self.observed_state = observed_state
        self.children = []
        self.parent = None
        
        # Sankoff algorithm data
        self.cost_matrix = cost_matrix  # Can be set directly now!
        self.optimal_count = None
        self.backpointers = None
        
    def add_child(self, child_node):
        """Add a child to this node"""
        self.children.append(child_node)
        child_node.parent = self
        
    def is_leaf(self):
        """Check if this is a leaf node"""
        return len(self.children) == 0
    
    def __repr__(self):
        state = self.observed_state if self.observed_state else "?"
        return f"Node({self.id}, {state})"


class PhylogeneticTree:
    """Manages the phylogenetic tree structure"""
    
    def __init__(self, alphabet: List[str] = ['A', 'C', 'G', 'T']):
        """
        Args:
            alphabet: List of possible states (characters)
        """
        self.alphabet = alphabet
        self.nodes = {}
        self.root = None
        self.substitution_cost = None  # 4x4 matrix for internal nodes
        
    def add_node(self, node_id: int, observed_state: Optional[str] = None, 
                 cost_matrix: Optional[Dict[str, float]] = None) -> TreeNode:
        """
        Create and add a node to the tree
        
        Args:
            node_id: Unique ID for this node
            observed_state: Character state (only for leaf nodes, can be None)
            cost_matrix: Cost matrix for this node (typically provided for leaves)
                        Format: {'A': cost_A, 'C': cost_C, 'G': cost_G, 'T': cost_T}
        
        Returns:
            The created TreeNode
        """
        node = TreeNode(node_id, observed_state, cost_matrix)
        self.nodes[node_id] = node
        return node
    
    def add_edge(self, parent_id: int, child_id: int):
        """Connect two nodes (parent -> child)"""
        parent = self.nodes[parent_id]
        child = self.nodes[child_id]
        parent.add_child(child)
        
    def set_root(self, node_id: int):
        """Set the root of the tree"""
        self.root = self.nodes[node_id]
        
    def set_cost_matrix(self, cost_matrix: np.ndarray):
        """Set the cost matrix (4x4 for internal nodes)"""
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
        """Return nodes in post-order (children before parents)"""
        if node is None:
            node = self.root
            
        result = []
        
        for child in node.children:
            result.extend(self.postorder_traversal(child))
        
        result.append(node)
        
        return result
    
    def __repr__(self):
        return f"PhylogeneticTree(nodes={len(self.nodes)}, leaves={len(self.get_leaves())})"

# Helper function for easy tree creation
def create_tree_from_input(
    leaf_cost_matrices: Dict[int, Dict[str, float]],  # node_id -> cost dict
    edges: List[Tuple[int, int]], 
    root_id: int,
    substitution_cost: np.ndarray,
    alphabet: List[str] = ['A', 'C', 'G', 'T']
) -> PhylogeneticTree:
    """
    Create phylogenetic tree with user-provided leaf cost matrices
    
    Args:
        leaf_cost_matrices: Dictionary mapping node_id -> cost_matrix for each leaf
                           Example: {0: {'A': 0, 'C': inf, 'G': inf, 'T': inf}}
        edges: List of (parent_id, child_id) tuples
        root_id: ID of root node
        substitution_cost: 4x4 numpy array for internal node calculations
        alphabet: List of possible states
    
    Returns:
        Initialized PhylogeneticTree ready for Sankoff's algorithm
    """
    tree = PhylogeneticTree(alphabet)
    
    # Find all node IDs
    all_node_ids = set(leaf_cost_matrices.keys())
    for parent, child in edges:
        all_node_ids.add(parent)
        all_node_ids.add(child)
    
    # Create nodes
    for node_id in all_node_ids:
        if node_id in leaf_cost_matrices:
            # Leaf node - use provided cost matrix
            tree.add_node(node_id, cost_matrix=leaf_cost_matrices[node_id])
        else:
            # Internal node - cost matrix will be computed by Sankoff
            tree.add_node(node_id)
    
    # Add edges
    for parent_id, child_id in edges:
        tree.add_edge(parent_id, child_id)
    
    # Set root and substitution cost matrix
    tree.set_root(root_id)
    tree.set_cost_matrix(substitution_cost)
    
    return tree