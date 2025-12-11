# Sankoff's Algorithm for Phylogenetic Analysis

Implementation of Sankoff's algorithm for ancestral state reconstruction with optimal solution counting. Includes application to cancer migration analysis using data from McPherson et al.

## Requirements

```bash
pip install numpy
```

---

## Part 1: Sankoff's Algorithm (Basic Trees)

### Running the Code

Run test.py. This runs Sankoff's algorithm on 4 different test trees and prints results for each.

### Code Structure

| File | Description |
|------|-------------|
| `PhyloTree.py` | Defines `TreeNode` and `PhylogeneticTree` classes. Handles tree construction, node management, and traversal. |
| `SankoffCounting.py` | Implements Sankoff's algorithm with solution counting. Contains the bottom-up cost computation, traceback, and recursive solution counter. |
| `test.py` | Test suite with 4 phylogenetic trees of varying structure and cost matrices. |

### How It Works

1. **Bottom-Up Pass**: Compute the cost matrices for each internal node using the Sankoff method (as described in the background section)

2. **Find Optimal Root**: Identify the minimum parsimony score and all possible root states that achieve this optimal cost

3. **Traceback**: Assign states to all internal nodes starting from the root, producing a single possible optimal tree labeling

4. **Count All Optimal Solutions**: Count all of the possible optimal solutions using the multiplicative principle, by identifying all optimal state choices at each node

### Interpreting Results

```
Optimal Parsimony Score: 9
Optimal Root State(s): ['T']
Total Optimal Solutions: 1
Solutions per root state: {'T': 1}
```

- **Optimal Parsimony Score**: The minimum total substitution cost across the entire tree
- **Optimal Root State(s)**: Which character(s) at the root achieve this minimum cost
- **Total Optimal Solutions**: How many different valid assignments exist with this cost
- **Solutions per root state**: Breakdown of solution count by root state (useful when multiple root states are optimal)

---

## Part 2: Cancer Migration Analysis

Applies Sankoff's algorithm to analyze tumor migration patterns in ovarian cancer patients using data from McPherson et al.

### Running the Code

Run MigrationRunner.py for any patient you want to process accordingly.

**Important**: The code runs one patient at a time. To analyze a different patient, open `MigrationRunner.py` and change the patient number at the bottom of the file:

```python
if __name__ == "__main__":
    # Change "patient10" to the patient you want to analyze
    results, matches = run_migration_analysis("patient10")
```

For example, to run patient 3:
```python
    results, matches = run_migration_analysis("patient3")
```

Make sure the corresponding data files exist:
- `patient3.tree`
- `patient3.labeling`
- `patient3.reported.labeling`

### Code Structure

| File | Description |
|------|-------------|
| `PhyloTree.py` | Same tree classes as Part 1 |
| `MigrationParser.py` | Parses McPherson et al. data files and builds the phylogenetic tree with anatomical locations as states |
| `SankoffCountingOvarian.py` | Modified Sankoff's algorithm with root constraint (tumor must originate at LOv, ROv, or RUt) |
| `MigrationRunner.py` | Main script that runs the analysis and compares with reported solutions |

### Input File Formats

- `.tree`: Parent-child edges (phylogenetic tree structure)
- `.labeling`: Leaf node locations (node name + anatomical site)
- `.reported.labeling`: Published solution from McPherson et al.

### Interpreting Results

The output shows:

1. **Migration Summary**: Lists each edge where the tumor migrated from one site to another
2. **Comparison Table**: Shows computed vs. reported assignments for each node
3. **Final Results**:
   - Optimal migration cost (minimum number of migration events)
   - Total optimal solutions (how many equally-good assignments exist)
   - Whether our solution matches the published one

A "PERFECT MATCH" means our algorithm found the same assignment as reported in the paper. Multiple optimal solutions indicate there are other equally valid migration histories.
