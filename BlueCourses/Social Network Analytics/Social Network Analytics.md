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
- Measure used to determine number of communities
- Suppose we have k communities
- Define a k * k symmetric matrix B with entries specifying fraction of edges that connect nodes 
  in community i with nodes in community j
- Trace gives fraction of edges that connect nodes in the same community
- Define row or column sum
- If communities are randomly connected, we have Q = 0
- Q-modularity is a measure of how well the network is partitioned into communities

## Homophily
- Homophily - is social networks:
- - People have tendency to associate with others whom they perceive as being similar
- - "Birds of a feather flock together"
- Homophily in fraud networks:
- - Fraudsters are likely to be connected to other fraudsters, and legitimate people are more likely to be connected to other legitimate people.
- Depends on: 
- - Connectedness between nodes with same label
- - Connectedness between nodes with opposite label
- Duadicity - measures connectedness between nodes with same label, also measures number of same label edges compared to what is expected in random configuration of network
- Expected number of same label edges in random network is calculated as product of degrees of nodes with same label divided by 2 * number of edges in network
- Example: 8 blue nodes, 4 green nodes, connectance = 0,2 = 4*3/2*0,2 = 1,2
- Dyadicity = number of same label edges / expected number of same label edges
- Example: 3 same label edges, dyadicity = 3/1,2 = 2,5
- Dyadicity: > 1: dyadic, ~ 1 random, < 1: anti-dyadic
- Heterophilicity - measures connectedness between nodes with opposite labels compared to what is expected  in random configuration of network
- Expected number of cross-label edges = product of degrees of nodes with different labels divided by 2 * number of edges in network
- Example: 8 * 4 * 0,2 = 6,4 
- Heterophilicity = actual number of cross label edges / expected number of cross label edges
- Example: 2 cross label edges, heterophilicity = 2/6,4 = 0,3
- Heterophilicity: > 1: heterophilic, ~ 1 random, < 1: heterophobic
- Homophilic network: Dyanicity > 1, Heterophilicity < 1
- Dyadicity and heterophlicity not be interpreted in an absolute way 
- Network can be either dyadic or heterophilic, but not both
- Not random distribution of labels can be meaningful in social networks
- Featurization!

## Social Network Based Predictive Analytics
- Estimate behaviour for nodes based on behaviour of their neighbours
- Challenges:
- - Data are now independent and identically distributed (IID) - behaviour of one node is dependent on behaviour of its neighbours, correlational behaviour between nodes
- - Collective inference - Nodes mutually influence one another, inference of one node depends on inference of other nodes
- Not easy separation in training and test data - out-of-time validation needed.
- Behaviours that cascade from node to node like an epidemic: news, opinions, etc.
- Markow assumption: future state of node depends only on current state of node and its neighbours
- Components: 
- - Non-relational classifiers: Only uses local info, can be esimated using traditional machine learning techniques, used to generate priors for relational learning and 
    collective inference
- Relational model - Makes use of relations/links in network, can be used to generate features for non-relational classifiers
- Collective inference - Inference of node depends on inference of other nodes, can be done using Gibbs sampling, iterative classification, etc.

## Relational Neighbor Classifier
- Assumptions: 
- - Homophily " guild by association "
- - Posterior class probability of a node is proportional to the number of neighbours of the node that belong to the class
- Gave good performance in fraud detection, churn prediction, etc.

## Relational Logistic Regression
- Local attributes - Describing a customer's behaviour (age, income)
- Network/link attributes:
- - Most frequently occuring class in neighbourhood
- - Frequency of classes of neighbours
- - Binary indicators indicates class presence
- Combine local and network attributes in a single logistic regression model
- Gave good performance in fraud detection, churn prediction, etc.
- Most frequently adopted in industry

## Social Network Featurization
- Map neighbor and network characteristics into features
- Combine with local variables for predictive modeling
- Network features can measure behavior in terms of:
- - target of neighbors
- - local characteristics of neighbors
- - network characteristics
- Carefully think about which features to include
- Variable selection will be very important
- Include as many features as possible
- Use variable selection
- Construct the predictive model using logistic regression, neural networks, SVMs, random forest.

## Collective Inference
- Given a network initialized by a local model and relational model, a collective inference method infers a set of class labels/probabilities for the unknown nodes.
- Gibbs sampling German and German (1984)
- Iterative classification - Lu and Getoor (2003)
- Relaxation labeling - Chakrabarti et al. (1998)
- Loopy belief propagation - Pearl (1988)
- PageRank - Brin and Page (1998)

## Gibbs Sampling
1. Given a network with known and unknown nodes
2. Initialize every unknown node using local classifier to obtain posterior
3. Sample class value of each node according to probabilities
4. Generate random ordering for unknown nodes
5. For each node i in ordering:
6. apply relational learner to node i to obtain posterior probabilities
7. sample class value for node i according to probabilities
8. Repeat step 5 during 200-iterations without keeping any statistic
9. Repeat step during 2000-iterations counting number of times each class is assigned to a node, normalizing these counts gives us the final class probability estimates.

## Iterative classification
Step 1: Bootstrap
- Use only local attributes to assign initial class to each node
Step 2: Iteration
- Start from given ordering of nodes
- Iteratively apply full model to classify each node until termination criterion is met (compute link stats based on current assigments, compute posterior probability for each 
  class, choose class based on largest posterior probability)
- Also need to decide on ordering strategy (which nodes to consider first step 2, based on number of links, link destiny, ...)

## PageRank
- Basis of Google's search engine
- Represents probability of visiting web page
- Depends on PageRank of linking pages and their number of outgoing links
- Takes into account random surfing behaviour
- Can be used to propagate behaviour in any type of social network:
- - adjacency matrix represents people-2-people network
- - known fraudulent nodes included in restart vector
- - vector contains fraud ranking

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
