# **LENC Algorithm Implementation**

### **Description**
This repository implements the Local Entropy Node Centrality (LENC) algorithm for identifying influential nodes in complex networks. The algorithm leverages information entropy and local structural properties to address the limitations of traditional methods. A few custom modifications were introduced in this implementation to explore alternative approaches and enhance the original LENC algorithm.

---

### **Table of Contents**
1. [Introduction](#introduction)
2. [Major Contributions of the Paper](#major-contributions-of-the-paper)
3. [Novel Contribution & Proposed Experiments](#novel-contribution--proposed-experiments)
4. [Methodology](#methodology)
5. [Results & Analysis](#results--analysis)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Limitations and Future Enhancements](#limitations-and-future-enhancements)
8. [Citation](#citation)
9. [How to Use](#how-to-use)

---

### **Introduction**
The Local Entropy Node Centrality (LENC) algorithm, developed by Bin Wang, Junkai Zhang, Jinying Dai, and Jinfang Sheng, identifies influential nodes by utilizing information entropy and local structural properties. It introduces a virtual node to capture edge weights and neighbor contributions, which helps in measuring influence more accurately and efficiently, especially in large-scale networks. This repository replicates and explores LENC's potential for various network types.

### **Major Contributions of the Paper**
1. **LENC Algorithm**: A novel approach for identifying influential nodes using local network properties and entropy.
2. **Virtual Node Introduction**: Enhances differentiation by reconstructing network structure with a virtual node, allowing edge significance to be accounted for in influence ranking.
3. **Computational Efficiency**: By focusing on local neighbors, the algorithm reduces complexity, making it feasible for large networks.
4. **Evaluation and Validation**: Performance is tested on eight real-world networks, demonstrating LENC’s effectiveness with high Kendall τ correlations in the SIR model and consistency across influence metrics.

### **Novel Contribution & Proposed Experiments**
In this work, we extend the LENC algorithm with a modified SIR model, introducing weighted centrality measures and randomness to more realistically simulate infection spread dynamics. The experiments utilize a discrete-time approach, which models infection spread by simulating node interactions over time and incorporates centrality-based influence measures.

### **Methodology**
1. **Network Representation**: Nodes and edges are represented graphically, with triangles indicating clustering.
2. **Edge Weight Calculation**: Calculations incorporate node degree and triangle counts.
3. **Virtual Node and Edge Weight**: A virtual node is introduced to differentiate nodes and set a common influence baseline.
4. **Entropy and Influence Calculation**: Influence is measured by the entropy of edges, capturing the diversity and distribution of node connections.

### **Results & Analysis**
- **Tiny Network Results**: Influence rankings for nodes are generated and validated against infection spread using scatter plots and Kendall τ correlation. For example, Node 4 achieved the highest influence score, aligning with expectations.
- **Opsahl-PowerGrid Network**: LENC identified the top influential nodes effectively, with high degrees correlating with high influence scores. The results are visualized to show infection spread patterns, demonstrating LENC’s utility in capturing critical network influencers.

### **Evaluation Metrics**
1. **SIR Model**: Susceptible-Infected-Recovered model simulates infection spread with specified infection and recovery probabilities.
2. **Kendall τ Coefficient**: Measures ranking correlation, demonstrating LENC’s consistency with the SIR model.
3. **Comparison with Other Algorithms**: LENC is compared against algorithms like Closeness Centrality, Eigenvector Centrality, HITS, H-index, and Degree Influence Line (DIL), showing high alignment with the SIR model.

### **Limitations and Future Enhancements**
- **Directed Graphs**: LENC currently assumes undirected graphs. Extending it to directed graphs could improve influence accuracy.
- **Dynamic Network Adaptation**: LENC assumes a static network. Adapting for dynamic networks would increase its utility in real-time applications.
- **Computational Complexity of SIR Evaluation**: Exploring more efficient techniques for large networks would improve scalability.
  
### **Citation**
If you use this repository or build upon this implementation, please cite the following paper:

**Bin Wang, Junkai Zhang, Jinying Dai, Jinfang Sheng**. *Influential nodes identification using network local structural properties*. Scientific Reports, vol. 12, no. 1, p. 1842, 2022. DOI: [10.1038/s41598-022-05564-6](https://doi.org/10.1038/s41598-022-05564-6).

### **How to Use**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/LENC-Algorithm-Implementation.git
   cd LENC-Algorithm-Implementation
   ```
2. **Install necessary libraries**:
   ```bash
   pip install numpy pandas networkx matplotlib scipy
   ```
3. **Run the notebooks**: Open `.ipynb` files in Jupyter Notebook to run the LENC algorithm and experiment with network data.
