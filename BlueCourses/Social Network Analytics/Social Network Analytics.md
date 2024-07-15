## Social Network Applications
- Social networks: Facebook, Twitter, LinkedIn, etc.
- Relation by: friendship, follow, professional, etc.
- Applications: recommendation, fraud detection, etc.

Example of social networks analytics

| Social Network App         | Nodes         | Edges          |
|----------------------------|---------------|----------------|
| email community mining     | people        | email sent     |
| churn prediction in Telco  | people        | call made      |
| fraud detection            | companies     | transaction    |
| web community mining       | web pages     | hyperlink      | 
| research community mining  | researchers   | citation       |
| terrorist community mining | people        | communication  |
| anti-money laundering      | bank accounts | cash transfers | 

- Traditional churn prediction models treat customers as isolated entitites.
- Customers are strongly influenced by their social env in the following ways:
- - recommendations from friends
- - mouth-to-mouth publicity
- - social leader influence
- - promotional offers from operators to acquire group of friends
- - reduced tariffs for intra-operator traffic

- Call Detail Record (CDR) data:
- - enables extracting the social network between the customers of an operator
- - GBs or Tbs per day
- Edges: 
- - Uni or Bi-directional
- - Weighted?
- - SMS, voice, MMS or email?

## Social Networks for Fraud Detection
- Fraudsters: 
- - tend to cluster together
- - exchange knowledge how to commit fraud
- - use the same resources
- - often related to the same person 
- - different profiles belong to the same person (identify theft)
- Fraud examples:
- - credit card transaction fraud
- - identify theft

## Social Networks
- Nodes: 
- - Vertical, points
- - Customers (private/professional), households/families, credit cards, merchants, patients, 
    doctors, papers, authors, terrorists, web-pages.
- Edges:
- - Different kinds of relationships: e.g. friendships, call, transmission of disease, 
    followings, reference, etc.
- - Weighted based on e.g. interaction frequency importance of information exchange, intimacy, 
    emotional intensity, etc.
- - can also be directed.
- Links aggregate intensity of relationship between 2 nodes
- Examples: 
- - Person A and B call each other 3 times per day and person B and C each other a month
- - Computer A and B communicate each ms and computer B and C each hour.
- - Website publishes 5 links to web pages of website B and website B publishes only 1 link to C


## Representing Social Networks
- Sociogram - best 
- Adjacency matrix - less
- Adjacency list - worst

## Network Centrality Measures
- Network centrality measures identify the most important vertices within a network
- Examples: 
- - Degree - number of edges
- - Closeness - distance to a node to all other nodes in the network
- - Betweenness: counts number of times that node or edge appears in geodesics
- - Graph Theoretic Center: node with the smallest sum of distances to all other nodes
- Geodesic: shortest path between two nodes

## Community Mining
- Community - substructure of graph with dense linkage between members of community and sparse 
  density between members of different communities
- Communities often occur in WWW, telecommunication networks, academic networks, friendship networks, etc.
- Peer pressure can strengthen tendency to commit fraud
- Are people more likely to commit fraud if they are influenced by a whole community than if 
  they were influenced by only one fraudulent individual? - Yes.
- Basic methods:
- - Graph partitioning
- - Girvan-Newman algorithm
- Advanced methods:
- - spectral clustering
- - directly optimizing modularity
- - finding communities with overlap

## Graph Partitioning Approaches
- Split whole graph into predetermined number of clusters
- Optimize ratio between within-community and between-community edges
- Different techniques to achieve optimal cut
- Iterative Bisection:
- - splits given graph into 2 groups using minimum cut size
- - cut size quantifies between-community edges
- - metrics: min cut, ratio cut and min-max cut

## Girvan Newman algorithm
- Similar to divisive hierarchical clustering
- Steps:
- - Calculate betweenness of all edges in graph
- - Edge with the highest betweenness is removed
- Result of the algorithm is essentially a dendrogram
- Other approaches: Q-modularity, FastModularity, etc.

## Bottom Up Community Mining
- Starts with one node and add more nodes to community based on links
- Extracted communities can be:
- - Complete: each node connected to each other node
- - Partial: each node connected to at least x% of other nodes
- Overlapping communities: communities where some nodes belong to more than one community
- Graph partitioning algorithms do no generate overlapping communities
- Bottom-up approaches able to create overlapping communities

## Modularity Q
-

## Homophily
-

## Social Network Based Predictive Analytics
-

## Relational Neighbor Classifier
-

## Probabilistic Relational Neighbor Classifier
-

## Relational Logistic Regression
-

## Social Network Featurization
-

## Collective Inference
-

## Gibbs Sampling
-

## Iterative classification
-

## PageRank
-

## From Unipartite towards Bipartite Networks
-

## Featurizing a Bigraph
-

## Propagation in bipartite graphs
-

## Multipartite graphs
-

## Gotcha!
-

## BiRank
-

## Representation Learning
-
