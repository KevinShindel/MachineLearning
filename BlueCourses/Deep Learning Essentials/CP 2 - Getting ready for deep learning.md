## Software
 - Use https://colab.research.google.com/ as platform for running the code.
## Setting up your environment
- Install Jypyter Notebook on your local machine.

## From If-then rules to deep learning.
- Expert-based rules
- - Obtained from business experts using knowledge elicitation techniques.
- - Subjective but not always to be frowned upon as inferior
- - Still remains popular in many settings.

- Data-based rules
- - Obtained from data using analytical techniques
- - Confirmatory vs. novel patterns
- - White vs Black box models, shallow vs deep models.

- Structured, 'tabular' data
- - Traditional 'shallow' modelling techniques usually perform well
- - Logistic regression, SVM, Decision Trees, Ensemble models

- Examples:
- - Churn prediction
- - Customer segmentation
- - CLV forecasting
- - Fraud detection

- DL not necessarily better in such cases
- As data complexity increases, artificial NN tend to be more performant in capturing patterns: Manual vs Auto feature engineering
- For very complex input data, ANN with a small number of hidden layers become inaccurate
- - E.g. images, sound, video, text...
- - For this type of data, deep ANNs, which are ANNs with more layers and more complex architectures are currently the most performant algorithms.

- Artificial NN have benn around for quite some time
- - Possibility to create 'deeper' networks was known about for quite some time

- Current retrieval triggered by:
- - Increase of computing power: emergence of GPUs/TPUs, parallel and cloud computing
- - Huge volumes of new data types: images, text, audio, video...
- - Software support: TensorFlow, PyTorch, Keras
- - New use cases: medical, finance, logistics, retail.

- The goal is to create algorithms that can take in very unstructured data, like images, audio waves or text blocks and predict the properties of those inputs.

### Comparing Deep Learning with Traditional Techniques
- Traditional algorithms best for small amount of data
- Deep learning is better for huge amount of data.

|                     | Traditional Algorithms                          | Deep Learning                                                     |
|---------------------|-------------------------------------------------|-------------------------------------------------------------------| 
| Accuracy            | Fair to good (on structured data)               | Good to excellent                                                 |
| Training time       | Short (seconds) to medium (hours)               | From medium to long (weeks)                                       | 
| Data Requirements   | Limited                                         | High                                                              |
| Feature engineering | manual trends features, windowing, aggregations | Auto, done by model                                               |
| Hyper-parameters    | Few to some (depending on algorithm )           | Many ( arch, num of layers, activation, optimizer )               |
| Interpretability    | High (white-box models) to reasonable           | Low (black-box model, though some explanations can be extracted ) |
| Cost and OpEx       | Low to reasonable                               | Reasonable to high (GPU, cloud, parralel computing )              |

### A Brief History of Deep Learning

- The beginnings: an electronic brain (1940s)
- The beginnings: the perceptron (1950s)
- From a golden age to an 'AI winter' (1960-1980s)
- Backpropagation to the rescue (1980-1990s)
- A second AI winter (1990-2000)
- Deep Learning (2000s)
- Deep Learning Unleashed (2010s)
- An Future of AI "Great AI awakening" (2020 and ahead)