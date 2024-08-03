## Neural Networks
- Predictive analytics technique: Classification, Regression.
- Biological view: Inspired by functioning of human brain.
- Statistical view: Generalization of stat models.
- Only 1 single hidden layer: universal approximation, shallow neural networks.
- Weights determined using optimization procedure:
- - Minimize error function (MSE)
- Number of hidden neurons aka hidden units: depends on complexity of problem.
- Hidden neurons tuned using iterative methods:
- - split data into training and validation sets.
- - vary hidden neurons from 1 to 10.
- - train neural networks on training set and measure performance on validation set.
- - choose number of hidden neurons with maximum validation set performance.
- - measure performance on test set.
- Hidden layer extracts features from data
- Features automatically learned by neural network
- But: computations required!

## Deep Learning Neural Networks
- Deep learning triggers:
- - GPUs and TPUs
- - Parallel computing
- - Cloud computing
- - NoSQL databases
- - Software: TensorFlow, Keras, Theano, Caffe, Torch, etc.
- Deep learning neural networks:
- - Image coloring
- - Image recognition
- - Image captioning
- - Translating text from image

## Deep Neural Networks for Recommendation
- Able to overcome drawbacks of collaborative/content filtering.
- Easier to combine rating, content and context data.
- Use of unstructured data: text, image, audio, video.
- Bias towards popular items can be overcome.
- Multiple deep neural network architecture possible:
- - popular approach considers recommendation as multiclass classification problem.
- Deep learning neural network also learn embeddings (matrix factorization).
- Squeezing input features through lower dimensional layers.

## AutoRec
- Item based or user based autoencoder:
- - Input: partially observed item or user rating vectors
- - Project onto low-dimensional hidden space
- - Output: fully reconstructed item or user rating vectors for recommendation.

## AutoRec in Python

```python
# Imports and Data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Install tensorflow with
# conda install -c conda-forge tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers
import tensorflow.keras.backend as K
# We're going to use the Movielens100k data set, which can be obtained in an easy CSV format at 
# https://www.kaggle.com/rajmehra03/movielens100k#ratings.csv

df = pd.read_csv('ratings.csv')
df.head()

# We preferable work with identifiers ranging continuously from 0..N, so let's change that first

# Replace user ids and movie ids with range(0, ...):

user_mask  = { v: n for n, v in enumerate(np.unique(df['userId'])) }
movie_mask = { v: n for n, v in enumerate(np.unique(df['movieId'])) }

df['userId']  = df['userId'].map(user_mask)
df['movieId'] = df['movieId'].map(movie_mask)

df.head()

num_users, num_items = len(np.unique(df['userId'])), len(np.unique(df['movieId'])); num_users, num_items

train_df, validate_df = train_test_split(df, stratify=df['userId'], test_size=0.1)

# AutoRec
# Create a helper function to convert our data frame to a Numpy matrix with shape [n_users, 
# n_items], with possibility to initialize with a constant or average value

def numpyfy(rating_df, num_users, num_items, init_value=0.0, init_with_average=True):
    if init_with_average: init_value = 0.0
    matrix = np.full((num_users, num_items), init_value)
    for (_, userID, itemID, rating, timestamp) in rating_df.itertuples():
        matrix[userID, itemID] = rating
    if init_with_average:
        average = np.true_divide(matrix.sum(1), np.maximum((matrix!=0).sum(1), 1))
        inds = np.where(matrix == init_value)
        matrix[inds] = np.take(average, inds[0])

    return matrix

# Let's now create our auto-encoder. We're using a very simple setup here with only one 
# bottleneck layer. Note that you can play around with adding more layers, finetuning 
# regularization, adding dropout, or even adding gaussian noise

def AutoRec(io_size, bottleneck_size=512, l2=0.0005, bottle_act='linear', output_act='linear'):
    model = tf.keras.Sequential()
    layer_input     = layers.Input(shape=(io_size,), name='InputLayer')
    layer_bottlenec = layers.Dense(bottleneck_size, activation=bottle_act, name='BottleNeck', 
                                   kernel_regularizer=regularizers.l2(l2))
    layer_output    = layers.Dense(io_size, activation=output_act, name='OutputLayer', 
                                   kernel_regularizer=regularizers.l2(l2))
    
    model.add(layer_input)
    model.add(layer_bottlenec)
    model.add(layer_output)

    return model

model = AutoRec(num_items)
model.summary()

# Next, we need a custom loss function, which only calculates the loss for items that were rated. 
# Here, we will assume that a rating of 0 means unrated, as the ratings are between 1 and 5

def loss_masked(y_true, y_pred):
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    mask_se   = K.square(mask_true * (y_true - y_pred))
    mask_sse  = K.sum(mask_se, axis=-1)
    return mask_sse

# Similarly, as an evaluation metric (RMSE), we also want to ignore unrated items. We also clip 
# the model's outputs to be between 1 and 5

def rmse_masked_clipped(y_true, y_pred):
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    y_pred    = K.clip(y_pred, 1, 5)
    mask_se   = K.square(mask_true * (y_true - y_pred))
    mask_rmse = K.sqrt(K.sum(mask_se, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
    return mask_rmse

train_m_X = numpyfy(train_df, num_users, num_items, init_with_average=True)
train_m_Y = numpyfy(train_df, num_users, num_items, init_with_average=False)
val_m_X   = numpyfy(validate_df, num_users, num_items, init_with_average=False)
model.compile(optimizer = optimizers.Adam(lr=0.001), loss=loss_masked, metrics=[rmse_masked_clipped])

# Take care of what's happening here:

# The autoencoder is trained using averaged rates on the input side and zero-d rates on the outut 
# side. The zeroes will be ignored by the loss function
# We validate using the average training rates on the input side (as that's what we have for the 
# users, but zero-d rates as seen in thhe validation set the calculate the metric (which will 
# also ignores zeroes)

fit_hist = model.fit(x=train_m_X, y=train_m_Y, epochs=200, batch_size=256, verbose=0,
                     validation_data=[train_m_X, val_m_X])
fit_hist.history.keys()

plt.plot(np.arange(len(fit_hist.history['rmse_masked_clipped'])), fit_hist.history['rmse_masked_clipped'])
plt.plot(np.arange(len(fit_hist.history['val_rmse_masked_clipped'])), fit_hist.history['val_rmse_masked_clipped'])
plt.title('Train vs. Validation RMSE')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()



plt.plot(np.arange(len(fit_hist.history['loss'])), fit_hist.history['loss'])
plt.plot(np.arange(len(fit_hist.history['val_loss'])), fit_hist.history['val_loss'])
plt.title('Train vs. Validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()

# Let's test the network for a particular user:

user_idx = 0

predictions = model.predict(np.array([train_m_X[user_idx,:]]))[0]
print(predictions)

for i, prediction in enumerate(predictions):
    if val_m_X[user_idx, i] == 0: continue # only show if in validation set
    print(i, prediction, 'versus true validation value', val_m_X[user_idx, i])

# Cleanup

tf.keras.backend.clear_session()
```

## Item2Vec
- Item based collaborative filtering method.
- Embedding items in low dimensional space.
- Inspired by Skipgram Word2Vec method.
- Sequence of words = set of items.
- Item embeddings can be used for:
- - dimensional reduction
- - clustering
- - prediction
- Note: performance similar to SVD in Barkan and Koenigstein (2016).

## Item2Vec in Python

```python
# Imports and Data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform

# Install tensorflow with
# conda install -c conda-forge tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers
import tensorflow.keras.backend as K
# We're going to use the Movielens100k data set, which can be obtained in an easy CSV format at 
# https://www.kaggle.com/rajmehra03/movielens100k#ratings.csv

df = pd.read_csv('ratings.csv')
df.head()

# We preferable work with identifiers ranging continuously from 0..N, so let's change that first

# We also keep an inverse mask here to map back to movie titles later on

# Replace user ids and movie ids with range(0, ...):
user_mask  = { v: n for n, v in enumerate(np.unique(df['userId'])) }
movie_mask = { v: n for n, v in enumerate(np.unique(df['movieId'])) }
movie_mask_inverse = { n: v for n, v in enumerate(np.unique(df['movieId'])) }

df['userId']  = df['userId'].map(user_mask)
df['movieId'] = df['movieId'].map(movie_mask)

df.head()

num_users, num_items = len(np.unique(df['userId'])), len(np.unique(df['movieId'])); num_users, num_items

# Save the movie names for later reporting

# Get the movie names
movies = pd.read_csv('movies.csv')
movies.set_index('movieId', inplace=True)

# Take care: we need to inverse the mask we have applied here
movie_titles = [movies.loc[movie_mask_inverse[i], 'title'] for i in range(num_items)]

# Item2Vec
# Create a helper function to convert our data frame to a Numpy binary matrix with shape [n_users,
# n_items]. Items rated above a cutoff are indicated as positives, otherwise negative

def numpyfy(rating_df, num_users, num_items, cutoff=3.5):
    matrix = np.full((num_users, num_items), 0)
    for (_, userID, itemID, rating, timestamp) in rating_df.itertuples():
        matrix[userID, itemID] = 1 if rating >= cutoff else 0
    return matrix
Xm = numpyfy(df, num_users, num_items); Xm.shape

# For item2vec, we will set up our network to train on pairs indicating whether two items appear together as positives in the same user-given ratings

pos_couples, neg_couples = np.argwhere(Xm == 1), np.argwhere(Xm == 0); len(pos_couples), len(neg_couples)

# Since we have way more negative pairs, we apply some negative sampling here

neg_couples = neg_couples[np.random.choice(neg_couples.shape[0], size=pos_couples.shape[0], replace=False), :]
# Final inputs X and y:

X = np.vstack([pos_couples, neg_couples])
y = np.hstack([np.full((len(pos_couples)), 1), np.full((len(neg_couples)), 0)])
# We're now ready to construct our model. Item2Vec uses a shared embedding layer for both the 
# target and context item, then simply applies a dot product. The output is equal to whether the 
# items show up together for the same user or not

def Item2Vec(num_items, vector_dim):
    input_target  = layers.Input((1,), name='inp_target')
    input_context = layers.Input((1,), name='inp_context')
    
    # Shared embedding as both our inputs are item integers
    embedding     = layers.Embedding(num_items, vector_dim, input_length=1, name='embedding')

    # Look up target and context vectors
    target  = embedding(input_target)
    target  = layers.Reshape((vector_dim, 1))(target)
    context = embedding(input_context)
    context = layers.Reshape((vector_dim, 1))(context)
    
    # Calculate dot product
    dot_product = layers.dot([target, context], axes=1)
    dot_product = layers.Reshape((1,))(dot_product)

    output = layers.Dense(1, activation='sigmoid', name='output')(dot_product)
    
    model = keras.Model(inputs=[input_target, input_context], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    return model

model = Item2Vec(num_items, 64)
model.summary()

# Now fit the model. Note that in a more fine-tuned approach, we'd call fit multiple times over the epochs whilst resampling from the negatives every epoch. Other opportunities for improvement include: adding regularization to the embeddings, dropout, etc.

fit_hist = model.fit(x=[X[:,0], X[:,1]], y=y, epochs=3, batch_size=32, verbose=1)

# We now have a trained set of embeddings for the items. Let's visualize them using t-SNE

vectors = model.get_layer('embedding').get_weights()[0]

tsne = TSNE(perplexity=40)
transformed = tsne.fit_transform(vectors)
sample_idx = np.random.choice(transformed.shape[0], size=50)

fig = plt.figure(figsize=(20,20))
plt.scatter(transformed[sample_idx, 0], transformed[sample_idx, 1])
for i in sample_idx:
    plt.text(transformed[i, 0], transformed[i, 1], movie_titles[i], color=np.random.rand(3)*0.7, fontsize=16)

plt.show()

# However, note that normally you'd calculate cosine similarity between the original higher-dimensional representation, in other words: t-SNE is mainly for visualisation purposes as we have higher-dimensional representations to work with.

# Let's try it this approach out

dists = squareform(pdist(vectors, 'cosine'))
selected_idx = movie_titles.index('Interstellar (2014)')
closest_idx  = np.argsort(dists[selected_idx, :])[1:1+10]

for i in closest_idx:
    print(movie_titles[i])

# Cleanup

tf.keras.backend.clear_session()
```

## Other Deep Learning Neural Networks for Recommendation
- Convolutional NN:
- - Feedforward NN with convolutional layers and pooling operations.
- - Capture global and local patterns in data.
- - Extract features from unstructured multimedia data.
- Recurrent NN:
- - Model sequential data
- - Remember previous computations when making predictions.
- Restricted Boltzmann Machines:
- - two-layer NN consisting of visible layer and a hidden layer.
- Deep reinforcement learning:
- - Dynamic systems
- - Capture user's temporal intentions and respond accordingly.
- Deep Hybrid Learning.

## Deep Learning: Evaluation
- Advantages:
- - Complex (unstructured) data
- - Non-linear behavior
- Disadvantages:
- - Interpretability
- - Extensive hyperparameter tuning
- - Scalability
- From the Dacrema et al. study:
- - Analyzed a number of recent deep learning algorithms for recommendation.
- - Reproducing published research is still challenging.
- - Most of the newer techniques can be outperformed by simpler algorithms.
- - Some even outperformed by TopPopular recommendation method.
- Reasons:
- Benchmarks not appropriate chosen.
- - Benchmarks not property tuned.
- - Not clear what is good benchmark.
- - Different data sets, data preprocessing and evaluation measures.
- - Tuning the parameters of new deep learning methid on test set.
- - Accuracy improvements often very marginal
- - Not all of the code is shared.

## Deep Learning: Conclusion
- Never start deep learning as your first approach for building a recommender system.
- Start with a k-nearest neighbor based method or extension thereof:
- - User-user collaborative filtering
- - Item-item collaborative filtering
