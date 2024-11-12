# Optimized Identification of Influential Nodes in Networks Through Entropy and SIR Model Validation

**Kartabya Krishna**  
***B.Tech in Data Science & Engineering, MIT Manipal***

### Base Paper
***B. Wang, J. Zhang, J. Dai, and J. Sheng, "Influential nodes identification using network local structural properties"***

---

## Abstract
This paper presents an algorithm, **LENC (Local Entropy Node Centrality)**, for identifying influential nodes within complex networks by leveraging information entropy and local structural properties. Traditional methods, often focusing solely on node degree or requiring global network structures, have high computational complexity or overlook edge significance in information dissemination. 

LENC addresses these issues by introducing a virtual node to account for edge weights and neighbor contributions, calculating node influence through a combination of first-order and second-order edge entropies. Tested on eight real-world networks, LENC is shown to more accurately rank influential nodes compared to existing algorithms, verified by its performance in the Susceptible-Infected-Recovered (SIR) model and high consistency in Kendall τ correlation tests. 

The LENC algorithm achieves a balance between efficiency and accuracy, making it suitable for large-scale networks, though further refinement for third-order influences could improve precision.

---
## Introduction

In recent years, understanding and managing the dynamics of complex networks has become a critical focus in research, as networks play a foundational role in diverse fields like epidemiology, social media, biology, and communication. Complex networks consist of nodes and edges that represent entities and interactions, and their structure can range from highly organized to remarkably decentralized. Identifying influential nodes—those with a disproportionately high impact on network behavior—is essential, as these nodes often control the spread of information, diseases, and influence within a network. Applications include identifying super-spreaders in epidemiology, key influencers in social networks, and central hubs in transportation and power grids.

Traditional methods to identify influential nodes primarily leverage centrality measures, each capturing a specific aspect of influence. **Degree centrality**, for instance, highlights nodes with the most direct connections, while **betweenness centrality** focuses on nodes that act as bridges in the shortest paths across the network. **Closeness centrality** examines nodes with the shortest paths to others, suggesting they can influence the network quickly. Despite their utility, these approaches often rely on global network structure, which can be computationally prohibitive and less effective in capturing local nuances within complex, heterogeneous networks.

Recent research has sought to overcome these limitations by integrating entropy-based methods that consider both node and edge properties. **Information entropy** offers a promising way to measure uncertainty and potential influence within networks, as it accounts for variability in connections and information flow. This approach provides a deeper understanding of node influence by considering not just connections but also the strength and distribution of these connections. Entropy-based methods can yield more nuanced influence metrics, especially in networks where nodes have varied roles and importance.

Given the rapid growth of real-world networks and the importance of real-time analysis, developing methods that accurately and efficiently identify influential nodes remains a significant research challenge. By focusing on local network properties and minimizing the need for global network data, entropy-based algorithms hold the potential to address these demands, creating new avenues for influence-based modeling in complex network research.

---
## Major Contributions

The major contributions of the paper, *Influential Nodes Identification Using Network Local Structural Properties*, are as follows:

1. **Development of the LENC Algorithm**: The paper introduces the Local Entropy Node Centrality (LENC) algorithm, a novel approach to identify influential nodes based on local network properties and entropy. Unlike traditional methods, LENC integrates edge weight distribution and local structural information to measure node influence, improving the accuracy of influence ranking.

2. **Use of Virtual Nodes for Enhanced Differentiation**: By introducing a virtual node, the LENC algorithm reconstructs the network structure to better differentiate nodes with similar structural features. This addition allows the algorithm to account for edge contribution in influence ranking, providing a more nuanced analysis of node importance.

3. **Reduction in Computational Complexity**: The LENC algorithm focuses on first- and second-order neighbors, reducing the need for global network data and lowering time complexity. This improvement makes LENC more computationally efficient, making it suitable for large-scale complex networks.

4. **Comprehensive Evaluation and Validation**: The paper validates the LENC algorithm using eight real-world networks and compares its effectiveness against several existing algorithms. Through infection size analysis, Kendall τ correlation, and the Susceptible-Infected-Recovered (SIR) model, the study demonstrates LENC's superior performance in accurately ranking influential nodes across diverse network types.
---

## Novel Contribution & Proposed Experiments

In this work, we experimented with a modified **SIR (Susceptible-Infected-Recovered)** model that incorporated a weighted approach, integrating centrality measures and randomness to handle the inherent variability in node interactions. The centrality measures, such as **Closeness Centrality** and **Eigenvector Centrality**, were used to weigh node connections, reflecting their influence in the network while introducing randomness to simulate more realistic infection spread dynamics.

Additionally, instead of relying on traditional ODE-based differential equations, we implemented a **discrete-time approach** to model the spread of infection. The simulation was performed by iterating over each node in the graph, where nodes transitioned from susceptible to infected or recovered states based on random probabilities, incorporating both the node's connections and a stochastic element. The infection dynamics were tracked over multiple time steps, with infected node counts recorded at each step, providing a detailed view of the epidemic's progression across the network.
---
## Methodology

The proposed methodology focuses on identifying influential nodes within complex networks by leveraging the **Local Entropy Node Centrality (LENC)** algorithm. The LENC algorithm emphasizes local structural properties, using information entropy to measure influence based on the weight distribution of edges connecting nodes. To differentiate nodes with similar structures, the method introduces a virtual node to reconstruct the network, calculating influence based on first-order and second-order neighbouring edges. This local approach provides efficient and accurate influence rankings while avoiding the computational burden of global network analysis.

The process begins with the introduction of a virtual node and virtual edges connected to all nodes, enabling enhanced differentiation between nodes with similar local structures. LENC then calculates entropy for each edge, accounting for the weight and contribution of both real and virtual edges to the node’s influence. This step-by-step approach builds up the influence score for each node based on both immediate and neighbouring connections. This influence score is then used to rank nodes within the network, with experimental validation demonstrating improved accuracy and efficiency.

### Preliminaries and Key Concepts

#### Network Representation  \( G=(V,E) \)

The network is represented by \( G=(V,E) \), where \( V \) is the set of nodes, and \( E \) is the set of edges connecting these nodes. This representation allows us to model the network as a graph, facilitating the analysis of relationships and connections among nodes. Each edge between nodes \( V_m \) and \( V_n \) is denoted as \( E_{mn} \). This graph structure is foundational for assessing complex networks, enabling a structured approach to analysing connections and dependencies between nodes.

#### Triangle Counting

The number of triangles formed between two nodes \( V_m \) and \( V_n \) and edge \( E_{mn} \) with common neighbour’s node sets being \( \Gamma(V_m) \) and \( \Gamma(V_n) \). The number of triangles that can form between the two nodes can be defined as:

$$
\text{Triangle}(E_{mn}) = \| \Gamma(V_m) \cap \Gamma(V_n) \|
$$

This triangle count helps in evaluating the clustering coefficient of a node, a crucial measure in network analysis, as it reveals how interconnected a node's neighbourhood is. Triangles are often indicators of tightly knit groups in the network, making them vital in analysing local node influence.

#### Edge Weight Calculation

The weight of an edge \( E_{mn} \) between nodes \( V_m \) and \( V_n \) is calculated as:

$$
\text{Weight}(E_{mn}) = \frac{(k(v_m) - T_{mn})(k(v_n) - T_{mn}) R_{mn} w_{mn}}{\left(\frac{T_{mn}}{2} + 1\right)}
$$

where \( k(v_m) \) and \( k(v_n) \) are the degrees of nodes \( v_m \) and \( v_n \), and \( T_{mn} \) is the triangle count involving edge \( E_{mn} \). This weighting scheme considers the information load and potential alternative paths available to a node, which influences the importance of edges. Higher weights indicate edges that are more crucial for information flow, helping identify key connections in the network.

$$
R_{mn} = \frac{k(v_m)}{k(v_m) + k(v_n)}
$$

#### Virtual Node and Edge Weight

To assess node influence comprehensively, a virtual node \( V' \) is introduced, connected to all other nodes via virtual edges. The edge weight for this virtual edge is defined as:

$$
W_{mv'} = \frac{k(v_m) k(v')}{k(v_m) + k(v')}
$$

where \( k(V') \) is the degree (information load) of the virtual node, set to the total number of nodes \( N \) in the network. This virtual node concept allows for the comparison of influence across all nodes by providing a common baseline, enhancing the accuracy of influence measurements.

The influence of all edges around the node is added, and the sum of the weights of the first-order edges of the nodes can be expressed as:

$$
W_m = W_{mv'} + \sum_{V_n \in \Gamma(V_m)} W_{mn}
$$

#### Information Entropy

Information entropy is used to measure the uncertainty associated with the weight distribution of edges connected to a node. For an edge \( E_{mn} \), the entropy is:

$$
\text{Entropy}(E_{mn}) = - \frac{W_{mn}}{W_m} \log_2 \left(\frac{W_{mn}}{W_m}\right)
$$

For the virtual edge \( E_{mv'} \), the entropy is:

$$
\text{Entropy}(E_{mv'}) = - \frac{W_{mv'}}{W_m} \log_2 \left(\frac{W_{mv'}}{W_m}\right)
$$

where \( W_m \) is the sum of weights of all edges connected to node \( V_m \). Entropy helps evaluate the information distribution and the stability of a node’s connections. Higher entropy values suggest a more diversified information flow, which is critical for assessing node influence in complex networks.

$$
\text{Entropy}(V_m) = \text{Entropy}(E_{mv'}) + \sum_{V_n \in \Gamma(V_m)} \text{Entropy}(E_{mn})
$$

#### Node Influence Calculation

Node influence is determined by the combined entropy of its edges and its position within the network, represented by the **k-core coefficient**. The influence of node \( V_m \) is calculated as:

$$
\text{Influence}(v_m) = \text{Entropy}(v_m) \times \text{k-core}(v_m)
$$

This influence measure integrates local connectivity and positional importance, allowing us to distinguish between central and peripheral nodes. Nodes with higher influence are more integral to network connectivity, which is crucial for identifying influential nodes in network analysis.

The total influence of a node considers not only its first-order influence (based on its direct connections) but also the influence contributed by its neighbours. It is calculated as:

$$
\text{Influence}(v_m) = \text{Influence}(v_m) + \sum_{V_n \in \Gamma(V_m)} \text{Influence}(v_n)
$$

Second-order influence expands the scope of influence measurement by including the contribution from neighbouring nodes, making the analysis more robust. This layered approach captures indirect influence, which is essential in understanding the broader impact of nodes within the network.
---
![The flow chart of LENC Algorithm](https://github.com/user-attachments/assets/ab51ecf8-bfba-4a22-8c24-71649e1ca517)
---

## Workflow of LENC Algorithm

1. **Construct the Network**
    - **Introduce a Virtual Node \( V' \)**: A virtual node is added to the network, connecting to all nodes. This aids in distinguishing nodes with similar structures by creating additional reference edges.
    - **Generate New Network Graph \( G(N,E) \)**: The addition of \( V' \) modifies the network structure, forming a new graph representation.

2. **Calculate the Weights of the Edges**
    - **Calculate the Number of Triangles \( \text{Triangle}(E_{mn}) \)**: For each edge \( E_{mn} \) between nodes \( V_m \) and \( V_n \), calculate the number of triangles that the edge forms with other nodes, reflecting the local clustering.
    - **Calculate the Edge Weights \( \text{Weight}(E_{mn}) \)**: Determine the weight of each edge, which depends on the degrees of connected nodes and the clustering effect within their neighborhoods.
    - **Calculate the Entropy of Edge \( \text{Entropy}(E_{mn}) \)**: Entropy is calculated to capture the uncertainty or diversity in the distribution of edge weights around each node.

3. **Calculate the First-Order Edge Effect**
    - **Sum of Line Entropy \( \text{Entropy}(v_m) \)**: Calculate the total entropy of edges connected to a node, which provides an indication of its local influence based on immediate connections.
    - **Introduce Influence Parameter \( k\text{-core}(v_m) \)**: Incorporate the node’s \( k \)-core level to weigh its importance within the network’s hierarchical structure.
    - **First-Order Link-Effect Contribution \( \text{Influence}(E_{mn}) \)**: Combine entropy with the \( k \)-core parameter to obtain a refined influence score.

4. **Calculate the Total Influence**
    - **Expanding the Scope of Influence**: Expand influence measurements to include indirect effects, considering second-order edges and beyond.
    - **Evaluation Index of Node Network Influence \( \text{Influence}(v_m) \)**: Calculate a comprehensive influence score for each node, ranking nodes based on their local and extended influence in the network.
---
## Time Complexity of the LENC Algorithm

The LENC algorithm's time complexity has three main parts:

1. **Edge Weight Calculation**:  
   Calculating edge weights involves determining the number of common neighbours (triangles) between nodes. This has a time complexity of \( O(N \langle k \rangle) \), where \( N \) is the number of nodes and \( \langle k \rangle \) is the network's average degree.

2. **Local Influence Calculation**:  
   To compute the local influence, the algorithm uses the k-core attribute, requiring a traversal of all network edges. This step has a time complexity of \( O(|E|) \), where \( E \) is the number of edges.

3. **Total Influence Calculation**:  
   For total influence, the algorithm accumulates the weighted entropy of first and second-order edges by traversing two-layer neighbour nodes. This has a time complexity of \( O(N \langle k \rangle^2) \).

Thus, the overall time complexity of the LENC algorithm is:

$$
O(N \langle k \rangle^2 + |E|)
$$

As shown in the table below, LENC has a relatively low time complexity compared to other entropy-based centrality measures.

## Algorithm Complexity Comparison

| Algorithm | Complexity |
|-----------|------------|
| CC        | \( O(n^2 \log n + nm) \) |
| EC        | \( O(n^2) \) |
| HITS      | \( O(n) \) |
| Hindex    | \( O(n \log n) \) |
| DIL       | \( O(n \langle k \rangle^2) \) |
| LENC      | \( O(n \langle k \rangle^2) \) |
| IIE       | \( O(n) \) |
| AIE       | \( O(n^2) \) |
| IE        | \( O(m + n + r \log n + rm^2) \) |


## Experimental Setup Details

In this study, the Local Entropy Node Centrality (LENC) algorithm was implemented for identifying influential nodes in complex networks, rather than using machine learning or deep learning models. The algorithm's parameters and computations focused on network metrics, including triangle counting, edge weight calculation, information entropy, and influence measures based on node degrees and k-core decomposition. Each parameter and calculation were tailored to accurately reflect the local and global influence of nodes within the network, emphasizing computational efficiency and relevance to real-world networks.

### Experimental Environment

The experiments were conducted on a system with the following specifications:

- **Processor**: Ryzen 7 6800h
- **RAM**: 32 GB
- **Operating System**: Windows 11
- **Python Version**: 3.11
- **Libraries**: NetworkX, Pandas, Matplotlib, Numpy, Collections, Math, Random, SciPy

### Tools Used for Implementation

- **Pandas**: Used for efficient data manipulation, storage, and retrieval, allowing seamless integration of node and edge information for influence calculations and analysis of the results.
- **NetworkX** (imported as `nx`): Used to create and analyze graph structures, calculate centrality measures, and implement the LENC algorithm. NetworkX facilitated operations like node degree calculations, k-core decomposition, and edge handling, all essential for assessing influence across nodes.
- **Matplotlib** (imported as `plt` and `cm`): Utilized to visualize network structures and influence metrics, generating scatter plots and heatmaps to show correlations between centrality metrics, entropy, and infection spread, as modeled by the SIR algorithm.
- **Numpy** (imported as `np`): Employed for array operations, mathematical computations, and statistical calculations, supporting the efficient handling of data structures required for entropy and influence calculations.
- **Defaultdict** (from `collections`): Used to manage dictionary-based data structures, particularly useful for grouping nodes and edges based on criteria like degree or influence level in the network.
- **Math**: Utilized for mathematical operations essential in calculating influence metrics, entropies, and probability functions, supporting the core computations in the algorithm.
- **Random**: Employed to introduce randomness in simulations, such as randomized infection spread in the SIR model, providing a stochastic element to the model’s output and influence calculation.
- **SciPy’s Kendall Tau** (from `scipy.stats`): Used to compute the Kendall Tau correlation coefficient, helping measure the similarity between influence rankings produced by different centrality metrics, providing insight into ranking consistency across metrics.

---
## Result & Analysis

### Opsahl-PowerGrid Network

This undirected network represents the power grid of the Western States of the United States, where each node is a generator, transformer, or substation, and each edge represents a power supply line. The dataset can be downloaded from the website Network Repository/Opsahl-PowerGrid.

Here are the characteristics of the Opsahl-PowerGrid Network:

- **Nodes**: 4,941
- **Edges**: 6,594
- **Density**: 0.000540303

**Degree Statistics**:
- **Maximum Degree**: 19
- **Minimum Degree**: 1
- **Average Degree**: 2

**Assortativity**: 0.00345699

**Triangle Statistics**:
- **Number of Triangles**: 2,000
- **Average Number of Triangles**: 0
- **Maximum Number of Triangles**: 21

**Clustering**:
- **Average Clustering Coefficient**: 0.0801036
- **Fraction of Closed Triangles**: 0.103153

**k-core**: Maximum k-core of 6

**Clique Bound**: Lower bound of the maximum clique is 6

---

### Tiny Network

#### Table 2: Tiny Network Used for Manual Tracing of LENC Algorithm (saved as `tiny.csv`)

| Edge No. | Vertex 1 | Vertex 2 |
|----------|----------|----------|
| 1        | 1        | 2        |
| 2        | 2        | 4        |
| 3        | 2        | 3        |
| 4        | 4        | 5        |
| 5        | 4        | 6        |
| 6        | 4        | 3        |
| 7        | 6        | 3        |

Here are the characteristics of the Tiny Network:

- **Nodes**: 6
- **Edges**: 7
- **Density**: 0.466667

**Degree Statistics**:
- **Maximum Degree**: 4
- **Minimum Degree**: 1
- **Average Degree**: 2.33

**Assortativity**: -0.458333

**Triangle Statistics**:
- **Number of Triangles**: 2
- **Average Number of Triangles**: 0.33
- **Maximum Number of Triangles**: 2

**Clustering**:
- **Average Clustering Coefficient**: 0.388889

**k-core**:
- **Maximum k-core**: 2

**Clique Bound**:
- **Lower bound of the maximum clique**: 3
---

## SIR Model

Kermack and McKendrick proposed the SIR model in 1927. The model includes **S**, **I**, and **R** states. **S** indicates susceptible, **I** indicates infected, and it can infect other healthy nodes with a certain probability. **R** indicates recovered and has immunity. The SIR model is defined as follows:

$$
\frac{dS(t)}{dt} = -\beta S(t) I(t) \\
\frac{dI(t)}{dt} = \beta S(t) I(t) - \gamma I(t) \\
\frac{dR(t)}{dt} = \gamma I(t)
$$

Where:
- $S(t)$ represents the number of susceptible nodes,
- $I(t)$ represents the number of infected nodes,
- $R(t)$ represents the number of recovered nodes at time $t$,
- $\beta$ represents the probability of infection,
- $\gamma$ represents the probability of recovery.

### Pseudocode for SIR Model Simulation:

1. **Initialize the sets**:
   - Susceptible: all nodes in the network
   - Infected: set of initially infected nodes
   - Recovered: empty set
   
2. **For each step in the simulation (total number of steps specified)**:
   - a. Initialize an empty set for `new_infected` nodes.
   - b. For each node in the Infected set:
     - i. For each neighbour of the node:
       - If the neighbour is in the Susceptible set and a random chance < infection probability $\beta$:
         - Add the neighbour to `new_infected` set.
     - ii. With a probability equal to the recovery rate $\gamma$, add the node to the Recovered set.
     
   - c. **Update the sets**:
     - Remove `new_infected` nodes from Susceptible.
     - Update Infected to include `new_infected` nodes and remove any that have recovered.
     
   - Return the final Infected and Recovered sets.

### Pseudocode for Ranking Nodes by Influence:

1. **Initialize a dictionary** (`influence_scores`) to store influence scores for each node, initially set to 0.
   
2. **For each simulation (total number specified)**:
   - a. Choose a random node as the initial infected node.
   - b. Run the SIR model with this initial infected node.
   - c. For each node in the resulting Infected or Recovered set:
     - Increment the influence score for that node by 1.
   - d. Sort nodes by their influence scores in descending order to determine the most influential nodes.

---

## SIR Model Parameters

The Susceptible-Infected-Recovered (SIR) model parameters are crucial for simulating and analysing the spread of influence in a network. By controlling infection and recovery rates, as well as the choice of initial infected nodes, we can observe the relative effectiveness of different ranking algorithms in spreading influence.

### Parameters:

#### **Infection Probability $\beta$**:
The infection probability $\beta$ represents the likelihood that a susceptible node will become infected when it interacts with an infected neighbour. Here, $\beta$ is calculated as:

$$
\beta = \frac{2 \langle k \rangle}{\langle k^2 \rangle}
$$

Where:
- $\langle k \rangle$ is the average degree of nodes in the network,
- $\langle k^2 \rangle$ is the average square of the degree, accounting for second-order neighbour effects.

This calculation helps adapt the infection spread rate based on network connectivity, allowing more realistic infection modelling in various network structures.

#### **Recovery Probability $\gamma$**:
The recovery probability $\gamma$ defines the rate at which infected nodes recover and become immune. For this experiment, $\gamma = 1$, meaning each infected node transitions to the recovered state with certainty in each time step, providing a stable benchmark across all simulations.

#### **Steps $T$**:
The simulation is run for a total of $T$, representing the number of cycles in which infection may spread. Each step models a progression in time, where nodes may infect their susceptible neighbors and potentially recover. This parameter controls the duration over which influence spread is observed.

#### **Initial Infected Nodes**:
In this setup, the top-10 nodes identified by various ranking algorithms are chosen as the initial infected nodes. These nodes serve as the starting points for infection, allowing us to compare the influence-spreading capabilities of nodes ranked by different methods. By measuring the number of infections originating from each top-10 set, we can assess each algorithm's effectiveness.

#### **Simulations**:

To account for the stochastic nature of infection and recovery, the model runs 1,000 independent simulations. This ensures that the results are statistically robust and provides a comprehensive assessment of each ranking algorithm's initial infection potential.

### Kendall Coefficient

The **Kendall τ coefficient** is used to explain the correlation of two sequences. The correlation coefficient reflects the proximity of two sequences. Suppose two sequences are related and have the same number of elements, expressed as:

- $X = x_1, x_2, \dots, x_n$
- $Y = y_1, y_2, \dots, y_n$

For the elements in both sequences, if $x_i > x_j, y_i > y_j$ or $x_i < x_j, y_i < y_j$, then any pair of sequence tuples $(x_i, y_i)$ and $(x_j, y_j), (i \neq j)$ are considered to be **concordant**. If $x_i < x_j, y_i > y_j$ or $x_i > x_j, y_i < y_j$, they are considered **discordant**. If $x_i = x_j$ or $y_i = y_j$, they are considered neither consistent nor inconsistent.

The **Kendall τ coefficient** is defined as:

$$
\tau(X, Y) = \frac{n_c - n_d}{0.5n(n-1)}
$$

Where:
- $n$ is the total number of combinations in these sequences.
- $n_c$ and $n_d$ indicate the number of concordant and discordant pairs, respectively.

This coefficient reflects the correlation and matching between two sequences. In general, $\tau \in [-1, 1]$, where $\tau > 0$ indicates a positive correlation and $\tau < 0$ indicates a negative correlation. That is, the higher the $\tau$ value, the more accurate the ranking.

---

### Using Kendall's Tau in Network Influence Rankings

Using Kendall's Tau in the context of network influence rankings can help measure how similarly two ranking algorithms (or methods) rate nodes in a network in terms of their influence. Here’s an example with a highly positive and a highly negative Kendall’s Tau to illustrate:

---

### Example 1: Highly Positive Kendall's Tau (close to +1)

Suppose we have two different influence-ranking algorithms, **Algorithm A** and **Algorithm B**, applied to rank five users (nodes) in a social network based on their influence. The rankings are as follows:

| User | Algorithm A Rank | Algorithm B Rank |
|------|------------------|------------------|
| U1   | 1                | 1                |
| U2   | 2                | 2                |
| U3   | 3                | 3                |
| U4   | 4                | 4                |
| U5   | 5                | 5                |

Here:
- Both algorithms agree perfectly on the order of influential users.
- **Kendall's Tau** for this data would be $+1$, representing a perfect positive correlation, as all pairs of users are ranked in the same relative order.

**Interpretation:** This means that both algorithms view the influence of each user very similarly, showing a strong agreement in their assessments of who is most influential in the network. This is often ideal when validating a new algorithm against a trusted method.

---

### Example 2: Highly Negative Kendall's Tau (close to -1)

In this case, consider the rankings from two algorithms, **Algorithm C** and **Algorithm D**, where the results are almost entirely reversed:

| User | Algorithm C Rank | Algorithm D Rank |
|------|------------------|------------------|
| U1   | 1                | 5                |
| U2   | 2                | 4                |
| U3   | 3                | 3                |
| U4   | 4                | 2                |
| U5   | 5                | 1                |

Here:
- Algorithm C ranks U1 as the most influential, while Algorithm D ranks U1 as the least influential, and so forth, with each rank completely opposite between the two algorithms.
- **Kendall's Tau** here would be $-1$, representing a perfect negative correlation.

**Interpretation:** A **Kendall's Tau** of $-1$ shows a complete disagreement in rankings. Algorithm C and Algorithm D have an entirely opposite view on the influence hierarchy within the network, indicating a fundamental difference in how each algorithm interprets network influence.


### Comparison Algorithms

Five comparison algorithms are selected in our experiment. They are described as follows:

---

#### 1. **CC (Closeness Centrality)**

Closeness centrality is based on the global information of the network to determine the network influence of nodes. The smaller the relative distance between all the node pairs, the stronger the accessibility of node information, and the more important the nodes are. It has been widely used in research, but its time complexity is high.

---

#### 2. **EC (Eigenvector Centrality)**

This method considers that the influence of nodes in the network depends on both the number of neighbouring nodes and the influence of the neighbouring nodes themselves. Its essence is to increase the influence of the node itself by connecting to other nodes of relative influence. However, when there are many nodes with a large degree in the network, the phenomenon of fractional convergence may occur.

---

#### 3. **HITS (Hyperlink-Induced Topic Search)**

The HITS algorithm uses different metrics to assess the influence of nodes in the network. Each node is assigned a hub value and an authority value to evaluate its influence. The authority value measures the originality of nodes in providing information, while hub values reflect the role of nodes in information transmission. These values interact and converge iteratively.

---

#### 4. **H-index**

This algorithm is mainly used to evaluate a scholar’s academic achievements. A higher H-index value indicates a greater influence of the node.

---

#### 5. **DIL (Degree and Influence-based Locality)**

DIL is a new algorithm that considers both the degree attribute of the node and the edge attribute of the node.


## Results

### Tiny Network

According to the LENC algorithm, the ranking based on decreasing order of Influence is as follows:

- **Node 4**: 13.2647 (Rank 1)
- **Node 3**: 12.5208 (Rank 2)
- **Node 2**: 10.6186 (Rank 3)
- **Node 6**: 8.9922 (Rank 4)
- **Node 5**: 4.2660 (Rank 5)
- **Node 1**: 4.3784 (Rank 6)

The results of the influence ranking for this set of nodes also demonstrate a clear correlation between degree and influence, though the relationship is less pronounced compared to the higher-degree nodes in the previous set. Node 4, with the highest influence score of 13.26 and a degree of 4, supports the idea that nodes with more connections tend to have higher influence. Similarly, Node 3 and Node 2, both with degrees of 3, rank second and third in influence, showing a moderate correlation between degree and influence in this subset. However, as we move down the list, nodes with lower degrees, such as Node 1 and Node 5 (with degrees of 1), exhibit significantly lower influence scores, suggesting that their limited connectivity diminishes their overall impact. 

This result can be used to evaluate the performance of the LENC algorithm, as the algorithm's effectiveness in ranking influence can be assessed by observing whether higher-degree nodes are appropriately assigned greater influence, as seen in this case.

Another way to interpret these results is by considering the impact of removing highly influential nodes. For instance, Node 4, which has the highest influence score of 13.26 and a degree of 4, would affect a larger portion of the network if removed, as it connects to more nodes than those with lower degrees. Eliminating Node 4 would disrupt connections to four other nodes, potentially reducing the overall influence flow and altering the structure of the network. This highlights the critical role that high-influence, high-degree nodes play in maintaining connectivity and influence distribution. This analysis further underscores the effectiveness of the LENC algorithm, as it correctly identifies nodes whose removal would have the most significant impact on the network.

---

### Scatter Plots Analysis

Figure 3 shows scatter plots depicting the relationship between various centrality measures (Closeness Centrality, Eigenvector Centrality, H-index, HITS Authority, Degree Influence Line, and Local Entropy Node Centrality) and infection spread in the network on the tiny network. The scatter plots presented here show the relationship between centrality scores (for each centrality measure) and infection spread across nodes in the network. Each subplot represents a different centrality measure, comparing the infection spread to the centrality score for that particular metric. Let’s analyse each:

1. **Closeness Centrality (CC)**: 
   This plot shows a somewhat positive trend where higher centrality scores are loosely associated with higher infection spread. This suggests that nodes with higher closeness centrality—those closer to other nodes in the network—may play a significant role in spreading infections.
   
2. **Eigenvector Centrality (EC)**: 
   There’s a mild association between higher eigenvector centrality and infection spread, but it's not very strong. Eigenvector centrality prioritizes nodes connected to other influential nodes, so this weaker association might indicate that such connections alone aren't enough to drive infection spread significantly.

3. **H-index**: 
   The plot for H-index centrality appears to have some clustering, with higher H-index nodes sometimes associated with greater infection spread. However, there’s variability in the spread even among nodes with similar H-index values, which might suggest that while H-index can identify influential nodes, other factors also play a role.

4. **HITS (Authority Scores)**: 
   The authority score from the HITS algorithm shows a mix of results, with higher scores occasionally aligning with higher infection spread. However, the scatter is wide, indicating no strong trend. HITS may capture influence differently, focusing more on directed networks, which might limit its predictive power in this case.

5. **Degree Influence Line (DIL)**: 
   DIL shows a somewhat clearer trend, where higher centrality values seem to correlate with increased infection spread. This measure emphasizes nodes with higher degrees, suggesting that more connected nodes are more influential in spreading infections.

6. **Local Entropy Node Centrality (LENC)**: 
   The LENC plot indicates a strong alignment between high centrality scores and higher infection spread. This result aligns well with previous findings (as noted by the Kendall's Tau of 0.9999), showcasing LENC's reliability in identifying influential nodes for spreading infection.

In summary, **LENC** and **DIL** show the strongest relationships with infection spread, implying that both metrics effectively capture node influence in this context. The Kendall's Tau correlation of 0.9999 for LENC further supports its accuracy as a predictor of infection spread, making it particularly valuable for this network analysis.

---

### Comparative Analysis of Node Rankings

Here is the table with ranks and scores for each algorithm, including the SIR model rankings:

| Rank | SIR Model      | LENC       | EC        | HITS      | CC        | H-index  | DIL      |
|------|----------------|------------|-----------|-----------|-----------|----------|----------|
| 1    | Node 4 (847)    | Node 4 (13.26) | Node 4 (0.5641) | Node 4 (0.2474) | Node 4 (0.8333) | Node 2 (2) | Node 4 (9) |
| 2    | Node 3 (842)    | Node 3 (12.52) | Node 3 (0.5095) | Node 3 (0.2234) | Node 2 (0.7143) | Node 3 (2) | Node 3 (9) |
| 3    | Node 2 (841)    | Node 2 (10.62) | Node 2 (0.4491) | Node 2 (0.1969) | Node 3 (0.7143) | Node 6 (2) | Node 2 (8) |
| 4    | Node 6 (815)    | Node 6 (8.99)  | Node 6 (0.3899) | Node 6 (0.1710) | Node 6 (0.5556) | Node 1 (1) | Node 6 (7) |
| 5    | Node 1 (721)    | Node 1 (4.38)  | Node 5 (0.2048) | Node 5 (0.0898) | Node 5 (0.5)   | Node 5 (1) | Node 5 (4) |
| 6    | Node 5 (696)    | Node 5 (4.27)  | Node 1 (0.1631) | Node 1 (0.0715) | Node 1 (0.4545) | Node 4 (0)  | Node 1 (3) |

**Table 3: Comparative Analysis of Node Rankings Across Influence Calculation Algorithms vs. SIR Model in Social Network Analysis.**

The **Kendall's Tau** correlation coefficient for the LENC ranking compared to the SIR model ranking is exceptionally high at **0.9999**, with a **p-value of 0.0028**. This result indicates a nearly perfect positive correlation between the rankings produced by the LENC algorithm and the SIR model. 

In practical terms, this suggests that LENC is highly consistent with the SIR model in identifying influential nodes within the network. The very low p-value also supports the statistical significance of this correlation, meaning the probability of observing such a strong correlation by chance is minimal. Therefore, **LENC** serves as a reliable predictor for influence spread, as validated by its strong alignment with the SIR model rankings.

# **Opsahl-PowerGrid Network**

Based on our implementation of **LENC algorithm**, the top 10 influential nodes identified are as follows:

- **Node 4436**: 125.9438
- **Node 4422**: 118.2816
- **Node 4417**: 109.1994
- **Node 4452**: 107.4150
- **Node 4434**: 106.5460
- **Node 4419**: 105.1733
- **Node 4453**: 103.6524
- **Node 4438**: 100.1719
- **Node 4407**: 92.4536
- **Node 4427**: 90.2799

The results of the influence ranking align well with expectations, as nodes with higher degrees tend to rank higher in influence. For example, **Node 4436**, which holds the highest influence score of **125.94**, also has a relatively high degree of **14**, suggesting that nodes with more connections in the network are more influential. Similarly, other nodes in the top 10, such as **Node 4422** and **Node 4438**, have degrees of **11**, and they exhibit high influence scores as well. This pattern indicates that the **LENC algorithm** may be working as intended, as higher-degree nodes are capturing significant influence. Additionally, if a higher-degree node like **Node 4436** were deleted, it would likely affect the maximum number of nodes in the network, as it connects to more nodes than those with lower degrees. This further supports the idea that high-degree, high-influence nodes play a critical role in the network’s structure and influence distribution. This is one way to evaluate whether the **LENC algorithm** is effectively ranking nodes based on their influence in the network.

### **Figure 3: Scatter plots depicting the relationship between various centrality measures (Closeness Centrality, Eigenvector Centrality, H-index, HITS Authority, Degree Influence Line, and Local Entropy Node Centrality) and infection spread in the network**

In **Figure 3**, the scatter plots present the relationship between centrality scores and infection spread for six centrality measures, with a primary focus on **Local Enhanced Neighborhood Centrality (LENC)**, our main algorithm. This analysis provides insight into how different centrality measures relate to infection spread, allowing us to assess **LENC’s** effectiveness in comparison to other metrics.

- **Local Enhanced Neighborhood Centrality (LENC):**
  - **LENC** scores show a wide distribution, extending up to **125**, with most nodes concentrated in the lower and middle ranges. This distribution suggests that **LENC** captures a broad spectrum of nodes’ local influence within the network.
  - The infection spread for nodes with higher **LENC** scores tends to be more variable but leans toward higher values. This indicates that **LENC** effectively identifies influential nodes that can significantly impact infection spread within the network. The wide distribution and varied infection spread values among nodes with high **LENC** scores suggest that **LENC** successfully captures critical influencers in the network, making it an ideal choice for applications involving infection or influence spread.
  - Unlike some of the other centrality measures, **LENC** does not suffer from over-clustering at low centrality scores, which enhances its utility in distinguishing between nodes with varying levels of influence.

- **Closeness Centrality (CC):**
  - Closeness centrality scores are narrowly spread between **0.04 and 0.08**, with infection spread covering the full range from **0 to 100**. The lack of variation in **CC** scores suggests it might not fully capture influential nodes as well as **LENC**, given that high infection spread is observed even for nodes with similar **CC** scores.

- **Eigenvector Centrality (EC):**
  - The majority of **EC** scores are concentrated around zero, with few nodes reaching up to **0.30**. This narrow range indicates that **EC** may not be as effective in identifying nodes with substantial influence, as the infection spread is inconsistent across centrality scores. Compared to **LENC**, **EC’s** low score concentration highlights its limitations in capturing influential nodes across the network.

- **H-index Centrality (Hindex):**
  - **H-index** values are distributed in segments from **1.0 to 4.5**, indicating a degree-based structure in the network. However, the segmented infection spread for each **H-index** score shows that it lacks the nuance **LENC** provides in capturing varying levels of influence, as infection spread does not consistently increase with **H-index** scores.

- **HITS Centrality (HITS):**
  - **HITS authority** scores are concentrated near zero, indicating that only a few nodes hold significant authority. The infection spread varies broadly among nodes with low **HITS** scores, suggesting that it might not be an effective predictor of spread. Compared to **LENC**, **HITS** centrality lacks the ability to distinguish highly influential nodes in terms of infection spread.

- **Degree Influence Line (DIL):**
  - **DIL** scores are segmented across a wide range (up to **80**), but infection spread is not strongly correlated with these scores. This pattern indicates that while **DIL** captures some level of influence, it is not as reliable as **LENC** in predicting which nodes will drive infection spread.

**LENC** emerges as a robust measure for identifying influential nodes in infection spread, outperforming other measures like **CC**, **EC**, and **HITS**. Its broad range and differentiated infection spread patterns demonstrate its ability to capture varying levels of influence within the network, making it highly suitable for applications involving influence or infection dynamics.

The **Kendall's Tau correlation coefficient** for the **LENC ranking** compared to the **SIR model ranking** on the Opsahl-PowerGrid Network is high, approximately **0.81**, with a low **p-value**. This result indicates a strong positive correlation between the rankings produced by the **LENC algorithm** and the **SIR model**, although it is not as close to perfect as in other networks. In practical terms, this suggests that **LENC** is largely consistent with the **SIR model** in identifying influential nodes within the PowerGrid network.

The strong correlation demonstrates that **LENC** is an effective method for ranking nodes by their infection-spreading potential, closely aligning with the **SIR model’s** influence criteria. The low **p-value** further supports the statistical significance of this correlation, indicating that the probability of observing this correlation by chance is minimal. Therefore, **LENC** serves as a reliable predictor for influence spread in the **Opsahl-PowerGrid Network**, validated by its alignment with the **SIR model rankings**.

---

### **Transmission Capacity**

**Figure 4: Transmission Initial Infection Capacity of Top-10 Nodes**

This graph, titled **"Transmission Initial Infection Capacity of Top-10 Nodes"**, visualizes the infection spread capacity over time for the top-10 influential nodes identified by various algorithms, including our proposed **LENC algorithm**. The y-axis (**F(t)**) represents the infection-spreading potential, while the x-axis (**Time Step, t**) tracks the progression over time.

In comparison to other centrality measures such as **Closeness Centrality (CC)**, **Eigenvector Centrality (EC)**, **H-index**, **HITS**, and **Degree Influence Line (DIL)**, **LENC** demonstrates a rapid decrease in **F(t)**, indicating faster convergence and effective infection spread. This trend suggests that **LENC** effectively identifies highly influential nodes, as its selected nodes reach infection stability more swiftly than most competing algorithms.

---

### **S I R Model Convergence Comparison between Various Algorithms**

**Figure 5: SIR Model Convergence Comparison Across Centrality Measures**

Based on the convergence times obtained from the simulations, **LENC** provides a highly effective selection of starting nodes for the infection model. With a convergence time of **344 steps**, it outperforms most other algorithms, including **CC (1561 steps)**, **HITS (916 steps)**, and **DIL (713 steps)**. This demonstrates that **LENC** identifies influential nodes more efficiently, allowing the infection to spread and reach stability significantly faster. Compared to **Random** starting nodes (**565 steps**), **LENC** also achieves quicker convergence, proving its robustness as a reliable choice for influence maximization in network diffusion scenarios.

---
### Conclusion

This study proposed the LENC (Local Entropy-based Node Centrality) algorithm to identify influential nodes in networks, emphasizing local structural properties and efficient computation. By integrating the **three degrees of influence theory**, which states that influence diminishes beyond two degrees, the LENC approach effectively captures both direct (first-order) and indirect (second-order) influence while reducing third-degree influence.

Experimental results validated LENC's accuracy, showing a strong correlation with SIR model spread simulations and alignment with actual network influence patterns. LENC's low computational complexity makes it suitable for large networks, offering a practical tool for applications in targeted information dissemination and epidemic control. Future work may focus on adapting this approach to dynamic network changes, further enhancing real-time influence analysis.

### Limitations and Future Enhancements

1. **Undirected Graph Assumption**: LENC currently operates on undirected graphs, which may not capture the asymmetrical influence paths present in directed networks like social or information networks. Future work could extend LENC to handle directed graphs, allowing for a more realistic assessment of influence in these contexts.

2. **Static Network Structure**: The algorithm assumes a static network, limiting its applicability in dynamic networks where connections change over time, such as social or communication networks. Adapting LENC for dynamic networks would enhance its utility in real-time influence analysis.

3. **Fixed Influence Decay**: LENC uses fixed weights to reduce third-degree influence, which may oversimplify influence decay across complex networks. Introducing variable decay factors based on network density or interaction patterns could yield more precise influence estimates, especially in networks with non-linear influence propagation.

4. **Limited Testing Across Network Types**: The current testing of LENC was focused on specific network types. Applying the algorithm to a broader range of networks (e.g., biological or transportation networks) may reveal unique challenges, suggesting potential adjustments for enhanced accuracy in diverse applications.

5. **Computational Complexity of SIR Model Evaluation**: The evaluation of the SIR model, especially when implementing differential methods, is computationally expensive. As the network size grows, the number of calculations required increases significantly, making it challenging to scale for large networks. Future work should explore more efficient techniques for simulating the SIR model, such as approximation methods or parallel computing. Additionally, it is important to investigate and incorporate a wider range of evaluation metrics to better assess the model's performance, especially in the context of dynamic or large-scale networks.

By addressing these limitations, LENC can be improved to more effectively model influence in a variety of network environments, making it suitable for real-world, dynamic applications.

### References

1. B. Wang, J. Zhang, J. Dai, and J. Sheng, "[Influential nodes identification using network local structural properties](https://doi.org/10.1038/s41598-022-05564-6)," *Scientific Reports*, vol. 12, no. 1, p. 1842, 2022. doi: 10.1038/s41598-022-05564-6.
2. T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed. New York, NY, USA: Wiley, 2006, ch. 7. doi: [10.1007/978-3-0348-8645-1_7](https://doi.org/10.1007/978-3-0348-8645-1_7).
3. W. O. Kermack and A. G. McKendrick, "[Contributions to the mathematical theory of epidemics. II.—the problem of endemicity](https://doi.org/10.1016/S0092-8240(05)80041-2)," *Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences*, vol. 138, no. 834, pp. 55–83, 1932. doi: 10.1016/S0092-8240(05)80041-2.
4. M. G. Kendall, "[A new measure of rank correlation](https://doi.org/10.2307/2332226)," *Biometrika*, vol. 30, no. 1, pp. 81–93, 1938. doi: 10.2307/2332226.
5. P. Marjai and A. Kiss, "[Influential performance of nodes identified by relative entropy in dynamic networks](https://doi.org/10.1142/S2196888821500032)," *Vietnam Journal of Computer Science*, vol. 8, no. 1, pp. 93–112, 2021. doi: 10.1142/S2196888821500032.
