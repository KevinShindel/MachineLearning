## Phase 1: Setup and Initialization

### Input Preprocessing
|           | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|-----------|----------|----------|----------|----------|
| Feature 1 | x11      | x21      | x31      | x41      |
| Feature 2 | x12      | x22      | x32      | x42      |
| Feature 3 | x13      | x23      | x33      | x43      |
| Feature 4 | x14      | x24      | x34      | x44      |
| Target    | y1       | y2       | y3       | y4       |

    
### Splitting Input 
- Training set: Used to fit the parameters
- Validation set: used for model selection/tuning
- Test set: used to evaluate the model on final performance
- Usual split: 80:10:10


### Model Architecture and Hyper Parameters
- Select values for the model
  - Layers and nodes in the layer, activation functions
  - Hyper parameters
- Selection criteria
  - Initial selection based on intuition/reference
  - Adjustment based on results

### Weights Initialization

- All weights and bias parameters need to be initiated to some value before we start training
- Zero initialization: initialize to zeros, not recommended
- Random initialization: initialize to random values from a standard distribution (mean=0, SD=1)

### Phase 2: Forward Propagation

Input -> Weight and Biases -> Prediction Values (PV) == Actual Values (AV)
- Send each sample through the NN and obtain the value of PV
- Repeat for all samples and collect a set of PV
- Compare the values of PV to AV to obtain error rates

### Phase 3: Measuring Accuracy and Error

Error in Prediction = PV - AV
Lost and Cost Function
- A loss function measures the prediction error for a single sample
- A cost function measures the error across a set of samples

### Popular Cost Functions

| Cost Functions                 | Applications               | 
|--------------------------------|----------------------------|
| Mean Squared Error (MSE)       | Regression                 |
| Root Mean Squared Error (RMSE) | Regression                 |
| Binary Cross Entropy           | Binary Classification      |
| Categorical Cross Entropy      | Multi-Class Classification |

### Measuring Accuracy
- Send a set of samples through the ANN and predict outcome
- Estimate the prediction error between the predicted outcome and expected outcome using a cost function
- Use back propagation to adjust weights based on the error value

### Phase 4: Back Propagation
- Each node in neural network contributes to the overall error in prediction
- A node's contribution is driven by its weights and bias
- Weight and biases need to be adjusted to lower the error contribution by each node

> How Propagation Works
> - Back propagation works in reverse of the forward propagation
> - Start from the output layer
> - Compute a delta value based on the error found
> - Apply the delta to adjust the weights and biases in the layer
 
> How Back Propagation Works
> - Drive a new error value
> - Back propagate the new error to the previous layer and repeat


### Phase 5: Gradient Descent - is a method to minimize the cost function

> Repeat the learning process
> - Forward propagation
> - Estimate error
> - Backward propagate
> - Adjust weights and biases

### Phase 6: Batches and Epoch
> Batch 
> - A set of samples sent through the network at once
> - The training data set can be divided into one or more batches
> - Training data is sent to the ANN one batch at a time
> - Cost estimated and parameters updated after each batch
> - Batch gradient descent: batch size = training set size
> - Mini-batch gradient descent: batch size < training set size
> - Typical batch size are 32, 64, 128 ... etc.
> - Batch size is a hyper parameter

> Epoch
> - The number of times the entire training set is sent through the ANN
> - An epoch has one or more batches
> - The training process completes when all epoch is completed
> - Epoch size can be higher to achieve better accuracy
> - Epoch size is a hyper parameter

### Epoch and Batch Size Examples
- Training set size: 1000
- Batch size: 128
- Epoch size: 50
- Batches per epoch = ceil(1000/128) = 8
- Total iteration (passes) through ANN = 8 * 50 = 400
- Batch size and epoch are parameters they can be tuned to improve model accuracy

### Phase 7: Validation and Testing

> Validation
> - During learning, the predictions are obtained for the same data that is used to train the parameters (weights and biases)
> - After each epoch and corresponding parameter updates, the model can be used to predict for the validation data set
> - Accuracy and/or loss can be measured and investigated
> - Model can be fine-tuned and learning process repeated based on results

> Evaluation
> - After the model is completed and final model obtained, the test data set can be used to evaluate the model
> - Results obtained with test data set is used to measure the performance of the model


### Phase 8: ANN Model

> Parameters
> - Weights
> - Biases
> Hyper Parameters
> - Number of layers
> - Nodes in each layer
> - Activation functions
> - Cost 
> - Learning rate
> - Optimizer
> - Batch size
> - Epoch

#### Prediction Process

> Preprocess and prepare inputs
> Pass inputs to the first layer
> - Compute Y using weights, biases, activation function
> - Pass to the next layer
> Repeat until the output layer
> Post-process the output for prediction


### Phase 9: Reusing exising network architecture

> Network Architecture
> - Most NN implementations are NOT created from scratch
> - Knowledge and experience shared by the community
> - NN architecture papers
> - Implementation code available in open source
> - Shared ope-source models

> Popular Architectures
> - LeNet5
> - AlexNet
> - ResNet
> - VGGNet
> - LSTM
> - Transformers


### Phase 10: Using available open-source models

> Open Source Models
> - Fully trained models with parameters shared by the open-source community
> - Repos/Registries like Hugging face/GitHub
> - Easy download
> - Integrate quicly into PyTorc/TensorFlow

> Selection open-source models
> - Understand their purpose/ original use case
> - Learn about the data they are trained on
> - Explore popularity and usage
> - Review licensing and related requirements
> - Download nad build pipelines
> - Test with data specific to your use case