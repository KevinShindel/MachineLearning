### The convolutional architecture
- Our 'deep' MLP already did relatively well on a simple data set
- But:
- - Black-white image 
- - Small resolution and size
- - Only 10 classes
- How about a data set with pictures of 1000 classes?
- - Cats, Dogs, Cars, Boats ?
- - Increase number of layers? Hidden units?
- - Lots of weights to learn
- LeNet + Alexnet
- - Use of GPUs to train the model
- - Use a method to reduce overfitting (Dropout)
- - Use the rectified linear unit (ReLU) as activation function
- - Use of convolutional layers
- Filters apply an operation to the input image, defined by a kernel
- - Spatially local weighted sum
- Instead of defining a filter by hand, we'll let the network learn several filters
- A convolutional layers learns d filters (the depth) of given size
- - The output is given by sliding the filter over the input image (feature map or convolved image)
- Each filter is defined by a kernel
- - These are the weights
- - The same kernel is used for all positions in the image (parameter sharing)
- - Receptive size determine by user, depth depends on the channels or depth of the previous layer 
- Pooling applies down sampling 
- - Also works with a size and stride
- - Same depth as the layer before
- - Max pooling: take the maximum value in the window
- - Global pooling: downsize the feature map to a single value
- The traditional convolutional architecture:
- - Convolution -> Pooling -> Convolution -> Pooling -> Fully connected
- - A number of dense layers at the end
- - Final layer with softmax activation
- More details in notebook: 'dle_cnn_imagedigits.ipynb'

### ☞ Handwritten digits recognition with a CNN
- See notebook: 'dle_cnn_imagedigits.ipynb'

### Best practices
- The input layer (the image) should be divisible by 2 many times
- - Common numbers include 32 , 64, 96, 224, 384, 512
- Convolutional layers should use small filters (3x3 or 5x5), a stride of 1 and padding
- - A stride of 1 allows to leave all spatial down-sampling to pooling layers, with convolutional layers only transforming the input volume depth-wise
- - If convolutional layers were not padded, the size of the volumes would reduce by a small amount after each convolutional layer
- - If we used strides greater than 1 or zero-pad the input , we would also have to carefully keep track of the input volumes throughout the CNN architecture
- - Some do argue for a large convolutional filter size (e.g. 11x11 or 7x7) for the first convolutional layer
- Prefer stack of small filter convolutional over one layer with a large receptive field
- - More small filter is better that one large filter
- - The effective receptive field over the image will be similar
- The conventional paradigm of a liner list of layers has been challenged:
- - If Google's Inception architecture and features more intricate and different connectivity structures
- - Same for the more modern ResNet architecture
- The pooling layers oversee down sampling the spatial dimensions of the input:
- - A common setting is the to use max-pooling with 2x2 receptive fields and a stride of 2
- - Many dislike the pooling operation and think that we can get away without it
- - This also allows to construct FNC's which are more flexible as they do not impose the limitation of having to use a fixed input image size
- The amount of memory can grow quickly with a CNN:
- - Since GPU's are often bottleneck`ed by memory, it may be necessary to compromise
- - Using a black and white picture instead of a colored one can help, or downsizing the input layer
- Many CNN architectures perform convolutions in the lower layers and add a number of fully connected layer at the top, followed by a softmax layer
- - Bridges the convolutional structure with traditional neural networks classification
- - Treats the convolutional layers as feature extractors and the resulting features are classified in a traditional way
- However, the fully connected layers are prone to overfitting
- - Typically, dropout layers are added to resolve this issue
- Another strategy is 'global average pooling'
- - Replaces the fully connected layers
- - Generate one feature map for each corresponding category

### Dropout
- Dropout is another method to reduce overfitting
- - At each training stage, individual nodes are dropped out of the net with a given probability, so that a reduced network is trained
- - Only the reduced network is trained on the data in that stage
- - The removed nodes are then reinserted into network with their original weights
- By avoiding training all nodes on all training data, dropout decreases overfitting
- - The method also improves training speed
- - The techniques reduces node interactions, leading them to learn more robust features that better generalize to the new data
- - Finding alternative or redundant pathways through the deeper layers
- Typically , dropout is added on the fully connected layers, but other locations are valid as well
- Dropout is preferably applied after a non-linear activation function:
- - However, when using ReLU, it might make sense to apply dropout before the activation function (for reasons computation efficiency, produce the same result in this case)

### Batch normalization
- One often normalize the input layer by adjusting and scaling the activations
- - If the input layer is benefiting from it, why not to do whe same things also for the values in hidden layers
- Batch normalization reduces the amount by which the hidden unit values shift around
- - Allows to use higher learning rates, because the normalization makes sure there's no activation that's extreme 
- - Reduces overfitting because it has a slight regularization effect
- Some have argued against Dropout altogether, in favor of using Batch Normalization only
- Typically, batch normalization is typically performed before the non-linear activation layer, but is sometimes done afterwards as well
- Best before do Dropout (if present)

-> Transfer function -> Batch normalization -> Activation function -> Dropout
-> Transfer function -> Batch normalization -> Dropout -> ReLU

### Data augmentation
-

### ☞ Colored image classification with a CNN
-

### Opening the black box
-

### ☞ Interpretability examples with a CNN
-

### Further aspects
-

### Transfer learning
-

### ☞ Transfer learning with a CNN
-

### Variants
-

### ☞ Locating objects with a CNN
-

### Capsule networks
-

### Adversarial attacks
-

### Use cases
-

### Deep dream
-

### ☞ Deep dreaming example
-

### Artistic style transfer
-

### ☞ Artistic style transfer example
-
