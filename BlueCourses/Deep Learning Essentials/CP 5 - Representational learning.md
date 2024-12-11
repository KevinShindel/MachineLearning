### Embeddings in text
- Let's start by focusing our attention on text
- Text is everywhere, and it is a rich source of information
- - Medical records
- - Product reviews
- - Repair notes
- - Facebook posts
- - Book recommendations
- - Tweets
- - Declarations
- - Legislation
- - Email
- - Websites
- Just like images, text is rather unstructured
- - No clean feature vector representation
- - Linguistic structure (languages, words relations, importance, etc.)
- - Text can be dirty (grammatical, spelling, etc.)
- Text is intended for human consumption: context is important
- Some approaches already well known to convert text to a feature vector
- E.g Bag of Words, TF-IDF, etc.

### Concept
- 

### Word embeddings
- The goal is to construct a dense vector of real values per word
- - The vector dimension is typically much smaller than the number of items
- - You can imagine the vectors as coordinates for items in the embedding space
- In other words: for each item we obtain a representation (a vector of real values)
- Distance metrics can be used to define a notion of relatedness between items in this space
- Matrix factorization based: Non-negative matrix factorization, Latent Semantic Analysis
- Neural network based: Word2Vec, GloVe, FastText

### word2vec
- Converts each term to a vector representation
- - Works at term level, not at document level
- - Such a vector comes to represent in some abstract way the 'meaning' of a word
- - Possible to learn word vectors that can capture the relationships between words in a surprisingly expressive way
- The general idea is that a word is correlated with its context: a form of self-supervised learning
- Two methods of learning: Continuous Bag of Words (CBOW) and Continuous Skip-gram (CSG)
- - CBOW: predict the current word given the context
- Layout of the network:
- - Input (CxV)
- - W (VxN) - the same weights are used for all words
- - CxV x VxN = CxN output tensor in the hidden layer
- - To collapse this to N, we simply average over each column
- - Output layer V, softmax activation
- Context words form the input layer: train word against context
- - Each words is encoded in one-hot form
- - If the vocabulary size is V these will be V-dimensional vectors with just one of the elements set to one, and rest all to zeros
- There is a single hidden layer and an output layer
- The training objective is to predict the output word given the input context words
- - The activation function of the hidden layer units is a weighted average
- - The final layer is a softmax layer
- Let's make an example with just one context word
- We want to derive a word vector of size N=4
- Layout:
- - Input (V)
- - W (VxN), no activation
- - Hidden layer (N)
- - Output layer (V), softmax activation
- Using the weight of hidden layer is sufficient to obtain word vectors after trainign
- Can apply division by context size to obtain averages, though it is not necessary
- Whole context and desired focus is presented as one training example
- This together with averaging activation function allows for some 'smoothing'
- Continuous Skip-gram is the opposite of CBOW
- - The focus word forms the input layer: train context agains word
- Also encoded in one-hot form
- Activation function is a simple pass-through
- The training objective is to predict the context
- The hidden layer here is simple pass-through
- We can hence also use the weights W after training
- Somewhat easier to implement
- Better quality vectors, but more training data required
- Here too, after training network, we can basically throw it away and only retain th hidden weights to get our vectors
- Size of hidden layer determines the word vector dimensionality
- - 32, 64, 128, 256, 512 are common choices
- - Approach somewhat comparable to auto-encoders
- - So: a kind of dimensionality reduction with context
- Advanced implementations also utilize:
- - Subsampling to skip retraining frequent words
- - Negative sampling: don't always modify all the weights
- - Hierarchical softmax: reduce the number of output nodes
- 
### ☞ Building a word2vec model
- See notebook "dle_emb_cbow.ipynb"

### Use cases
- "Though vector" - is a term popularized by Geoffrey Hinton, which is using vectors based on natural language to improve search results.
- - A though vector is like a word vector; vector represents one thought's relationships to others
- - A though vector is trained to generate a thought's context. Just as a words are linked by grammar.
- - Given one thought, a neural network can predict the next thought
- Mapping different language to the same vector space: explore if certain concepts, entities still position in a similar way in other languages.
- Airbnb: evaluate what characteristics of listings can be captured by embeddings
- - K-means clustering on the learned embeddings
- - Figure shows the resulting 100 clusters in California, and confirms that listings from similar locations are clustered together
- - Evaluated average cosine similarity between listings in the same cluster
- Twitter recommendation to users:
- - Being able to find similar items is an essential task for many candidate generation schemes
- - Find an embedding of items and a measure of distance between them such that similar items have embeddings that are close together

### Generalizing embeddings
- An embedding is simply a layer where a categorical input is mapped to vector of weights.
- More generally, many dimensionality reduction techniques (t-SNE, UMAP, PCA, auto-encoders) can be seen as embeddings, could also be considered as techniques which can be used 
  to create embeddings or representations of data.

### Variants
- Pennington et al.argue that the approach used by word2vec is suboptimal since it doesn't fully exploit statistical information regarding word co-occurrences.
- They propose GloVe (Global Vectors for Word Representation) which is based on matrix factorization techniques.
- Similar output as word2vec, but with a different training objective
- par2vec: extends word2vec to paragraphs
- doc2vec: extends word2vec to documents (some of these are relatively trivial and simply add or average the component vectors)
- node2vec: extends word2vec to graphs (random walk in graph the construct a sentence, then apply word2vec)
- word2vec is somewhat obsolete now
- fastText (Facebook) is a more recent variant of word2vec, performs better on small datasets
- The NLP and embeddings fields have progressed rapidly
- - ULMFiT (Universal Language Model Fine-tuning) by Howard and Ruder
- - ELMo (Embeddings from Language Models) by Peters et al.
- - BERT (Bidirectional Encoder Representations from Transformers) by Devlin et al.
- - COmbines recurrent approaches to predict the next word/character, bidirectional and transfer learning approaches, more nuanced notion of context.

### ☞ Graph embeddings example
- See notebook "dle_emb_graph.ipynb"
- Also see: graph neural networks and graph convolutions networks
- - Network which works over NxF feature matrix X and NxN adjacency matrix A

### Software
- It's pretty rare to construct word2vec models by hand
- - NLT for python (https://www.nltk.org/)
- - Gensim (https://radimrehurek.com/gensim/)
- - FastText (https://fasttext.cc/)
- - spaCy (https://spacy.io/)
- - AllenNLP (https://allennlp.org/)
- In many cases, pre-trained word vectors or models will be used

### Categorical embeddings
- Embeddings have also been applied for high-level sparse categorical data
- Embedding size somewhat arbitrary

### ☞ Featurization with categorical embeddings
- See notebook "dle_emb_categorical.ipynb"

### Auto-encoders
- Arhitecture which squeezes an input through a low-dimensionality bottleneck
- Desired output can be the same as the input, or a denoised version of the input
- Force the network to lean a sparse representation
- - Use full network after training
- - Or only use compressed representation as embeddings

### ☞ Anomaly detection with auto-encoders
- See notebook "dle_ae_anomalydetection.ipynb"

### ☞ Image denoising with auto-encoders
- See notebook "dle_ae_imagedenoise.ipynb"
