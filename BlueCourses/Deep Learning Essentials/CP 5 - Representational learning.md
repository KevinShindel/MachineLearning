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
-

### Generalizing embeddings
-

### Further aspects
-

### Variants
-

### ☞ Graph embeddings example
-

### Software
-

### Categorical embeddings
-

### ☞ Featurization with categorical embeddings
-

### Auto-encoders
-

### ☞ Anomaly detection with auto-encoders
-

### ☞ Image denoising with auto-encoders
-
