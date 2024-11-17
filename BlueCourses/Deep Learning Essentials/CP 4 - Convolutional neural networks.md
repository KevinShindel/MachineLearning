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
- Increase size of training set through transformations
- - Some popular augmentations are grayscale, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more
- - By applying just a couple of these transformations to the training data, the model can learn to generalize to unseen data
- Force the network to focus on important aspects of the image (prevents overfitting)
- Sometimes also applied at prediction-time  to stabilize the output

### ☞ Colored image classification with a CNN
- looks at notebook 'dle_cnn_imagecifar.ipynb'

### Opening the black box
- Layers activation:
- - A straightforward visualization technique is to show activations of the network during forward pass
- - For ReLU CNN networks, activations usually start out looking relatively blobby and dense , but as training progresses become sparser and more localized
- Weight and feature visualization:
- - The second common strategy is to visualize the weights of the network
- - These are usually most interpretable in the first convolutional layer, which is looking directly at the raw pixel data, but it is possible to also show the filter weights deeper in the network    
- - However, since the weight matrices are generally small and hard to understand, a common strategy is to find an input through optimization (e.g. gradient ascent) that maximizes the response of a particular filter
- Retrieving images that maximally activate a neuron:
- - Similarly, we can take our given set of images, feed them through the network and keep track of which images maximally activates a neuron or set of neurons
- - We can then visualize those top-n images to get an understanding of what the neuron is looking for
- Image occlusion:
- - Suppose that a CNN classifiers an image as a dog
- - How can we be certain that it's picking up on the dog in the image and not some cues from the background or other object?
- - One way to investigate which part of the image a classification prediction comes from is by plotting the probability of the class of interest as a function of the position of an occluding objects.
- - Iterate over regions of the image, set a patch of the image to be grey, and look at the probability difference of the class
- - We can then visualize this as heat map. The darker the color, the more important the region is for the classification
- Salient maps:
- - An apporach with a similar goal is saliency maps
- - Again, based on determing changes to a given imnput image which have the largest effect on the output, which can be optimized through SGD
- - See -> https://github.com/raghakot/keras-vis
- Embedding with e.g t-SNE or UMAP:
- - CNNs can be interpreted as gradually transforming images into a representation in which the classes are separatable by a linear classifier
- - We can get a rough idea about the topology of this space by embedding images into two dimensions so that their low-dimensional representation has approx. equal distances that their high-dimensional representation.
- - To produce an embedding, we can take a set of images and use the CNN to extract the vector of outputs right before the final softmax layer
- - We can then plug these into t-SNE or UMAP and get 2-dimensional vector for each image
- LIME (Local Interpretable Model-agnostic Explanations) https://www.github.com/marcotcr/lime
- An explanation is a local linear approximation of the model's behavior
- While the model may be very complex globally, it is easier to approximate it around the vicinity of a particular instance
- While treating the model as a black box, LIME can explain any classifier

- SHAP (SHapley Additive exPlanations) https://www.github.com/slundberg/shap
- SHAP values provide a unified measure of feature importance
- A unified approach to explain the output of any machine learning model
- Connects game theory with local explanations

### ☞ Interpretability examples with a CNN
- see notebook: 'dle_cnn_interpretability.ipynb'

### Transfer learning
- A common misconception is that without huge amounts of data, you can't create effective deep learning models
- Transfer learning is the process of taking pre-trained model and fine-tuning it on your dataset.
- - Idea is that pre-trained model will act as feature extractor: you remove the last layer of the network and replace it with your own classifier and only retrain those 
    weights while keeping the rest of the network weights fixed
- - Or simply keep as it but retrain the last layer
- - Lower layers of the network will detect features like edges, corners, etc. that are common to all images
- https://www.teachablemachine.withgoogle.com/ - train an image detector in your browser
- Uses a pretrained model SqueezeNet on ImageNet (1000 classes)
- - Hence, we get an output vector of size 1000 for any image
- - When the user trains this system, simply store output vector for each image together with given class label
- For a new image , look at k-nearest neighbors in the output space
- - We then see for these neighbors which class they belong to (A, B, C) and simply derive the probabilities based on this set.
- Works surprisingly well, even for images that are not in ImageNet
- - The network might think of a lemon for a banana, but it will not think of a car
- - When we give another banana picture to predict the network will think  lemon again
- - We simply translate this to the user as 'its' a banana'

### ☞ Transfer learning with a CNN
- See notebook: 'dle_cnn_transferlearning.ipynb'
- https://www.norc.aut.ac.ir/its-data-repository/accident-images-analysis-dataset/
- Data-set: accident-detection
- - The aim of this is to detect the occurrence of accidents in images
- - Folder 1 includes 2500 images with label 'without_accident'
- - Folder 2 includes 23980 images with label 'with_accident'

### Variants
- LeNet: the first successful application of CNN, developed by Yann LeCun in 1998
- AlexNet: the first large-scale CNN that had a significant impact on the field, developed by Alex Krizhevsky in 2012
- ZFNet: a variant of AlexNet, developed by Matthew Zeiler and Rob Fergus in 2013
- GoogLeNet: a CNN developed by Google that won the ImageNet challenge in 2014
- VGGNet: a CNN developed by the Visual Graphics Group at Oxford University in 2014
- ResNet: a CNN developed by Microsoft Research in 2015 that won the ImageNet challenge
- SqeezeNet: a CNN developed by DeepScale in 2016
- Basic CNNs are useful for image classification, but there are many other tasks that can be solved with CNNs
- For object localization, the goal is to produce bounding boxes around objects in an image
- For object segmentation, the task is to output an outline of every object in an image
- Multiple such object might be present in the image, complicating the problem
- A basic CNN setup can also be used to localize objects of interest in preprocessing the data appropriately
- At prediction, the model is queried at multiple locations in the image to find the most likely location of the object
- One dimensional CNNs have been used for the text and time series data analysis as well

### ☞ Locating objects with a CNN
- See notebook 'dle_cnn_localization.ipynb'

### Capsule networks
- Geoffrey Hinton, the godfather of deep learning, has been working on a new type of neural network called capsule networks
- - Also, introduce an algorithm, 'dynamic routing between capsules', that allows to train such a network
- - Tries to remove standing issues of the traditional CNNs architecture
- CNN learn filters, but have trouble to lean 'pose', 'composition' or 'shapes'
- - Internal data representation of a CNN does not consider important spatial hierarchies between simple and complex objects
- - Not strong notion of three-dimensional representation of objects, just filters
- The idea of capsule networks is relatively simple
- - But: computers were just not powerful enough to train them
- - There was no algorithm to train them
- New technique: dynamic routing between capsules
- - Allows capsules to communicate with each other and create representations similar to scene graphs in computer graphics

### Adversarial attacks
- Adversarial examples are inputs to machine learning models that an attacker has intentionally designed to cause the model to make a mistake
- Recent studies have shown that any machine learning classifier can be tricked to gie incorrect predictions by adding small perturbations to the input
- Lots of research going into how to make neural nets more robust against these attacks


### Use cases
- Image classification, segmentation, object detection, and localization
- Face recognition and classification
- - Alibaba launched a facial recognition system for payments
- - Beijing's subway system uses facial recognition to pay for tickets
- - New AI can guess whether you're gay or straight from a photograph
- - Facial Recognition Technology is used to identify missing persons
- Pose and gait detection
- Business applications, e.g. in insurance (take a picture of a car accident and get an estimate)
- Stylistic and artistic applications (photo filters, deep dreaming, artistic style transfer)

### Deep dream
- Prediction is not always the end goal
- Uses a CNN to find and enhance patterns in images, creating a dream-like appearance in the deliberately over-processed images
- - Enhance an input image in such a way as to elicit a particular interpretation
- - Start with an image full of random noise, then gradually tweak it based on outputs of the convolutional layers
- Recall that convolutional layers outputs get excited (attain higher absolute values) when a corresponding pattern has been detected
- - We can choose some layers to maximize depends primarily on whether we want to focus on lower or higher-level feature representations (or perhaps a combination)
- Loss:
- - Activation maximization loss
- - Continuity loss 0 to give the image local coherence and avoid messy blurs
- - L2 norm loss - prevent pixels from taking very high values
- Start with an image as the original
- scipy.optimize.fmin_l_bfgs_b(func, x0, fprime=None, args=(), approx_grad=0, bounds=None, m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)
- - func: function to minimize
- - x0: initial guess
- - fprime: gradient of the function
- - args: additional arguments
- - approx_grad: if the gradient is approximated
- - bounds: bounds for the variables
- - m: number of corrections to approximate the inverse hessian
- - factr: termination factor
- Simple optimization function to minimize loss, based on outputs of the network
- - Broyden-Fletcher-Goldfarb-Shanno algorithm
- - An iterative method for solving unconstrained nonlinear optimization problems
- - With limited memory 
- - And bounded constraints
- We modify image by looking at the netwrok to maximize its convolutional features
- - Fine tuning based on desired output possible
- - Most pre-trained models can be used
- We use some random jitter in every step to ad variety to the image, but many variations exist 
- - Gaussian blur, contrast, etc.
- - No jitter can lead to a very noisy image
- - Use of octaves to generate high-resolution images

### ☞ Deep dreaming example
- See notebook: 'dle_cnn_deepdream.ipynb'

### Artistic style transfer
-

### ☞ Artistic style transfer example
-
