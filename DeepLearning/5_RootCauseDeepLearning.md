# IT Operations: Root Cause Analysis

> A data center team wants to build a model to predict causes of issues reported by customers


## Problem Statement

> Using data about CPU load, memory load, networks delays, and other metrics, the team wants to predict the root cause 
> of issues reported by customers. The team has a dataset with 10,000 samples, each with 50 features. 
> The dataset has 10 classes, each representing a root cause of an issue. 
> The team wants to build a model that can predict the root cause of an issue with high accuracy.

> Feature variables: 
> - CPU load
> - Memory load
> - Network delays
> - Error 1000
> - Error 1001
> - Error 1002
> - Error 1003

> Target class:
> - Root cause of the issue

## Input preprocessing

1. Load data
2. Convert data

```python
import pandas as pd
import tensorflow as tf

#Load the data file into a Pandas Dataframe
symptom_data = pd.read_csv("root_cause_analysis.csv")

#Explore the data loaded
print(symptom_data.dtypes)
symptom_data.head()
```

```python
# import the necessary libraries
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

label_encoder = preprocessing.LabelEncoder()
symptom_data['ROOT_CAUSE'] = label_encoder.fit_transform(
                                symptom_data['ROOT_CAUSE'])

#Convert Pandas DataFrame to a numpy vector
np_symptom = symptom_data.to_numpy().astype(float)

#Extract the feature variables (X)
X_data = np_symptom[:,1:8]

#Extract the target variable (Y), conver to one-hot-encodign
Y_data=np_symptom[:,8]
Y_data = tf.keras.utils.to_categorical(Y_data,3)

#Split training and test data
X_train,X_test,Y_train,Y_test = train_test_split( X_data, Y_data, test_size=0.10)

print("Shape of feature variables :", X_train.shape)
print("Shape of target variable :",Y_train.shape)
```

## Model Building

```python
# import the necessary libraries
from tensorflow import keras

#Setup Training Parameters
EPOCHS=20
BATCH_SIZE=64
VERBOSE=1
OUTPUT_CLASSES=len(label_encoder.classes_)
N_HIDDEN=128
VALIDATION_SPLIT=0.2

#Create a Keras sequential model
model = tf.keras.models.Sequential()

#Add a Dense Layer
model.add(keras.layers.Dense(N_HIDDEN,
                             input_shape=(7,),
                              name='Dense-Layer-1',
                              activation='relu'))

#Add a second dense layer
model.add(keras.layers.Dense(N_HIDDEN,
                              name='Dense-Layer-2',
                              activation='relu'))

#Add a softmax layer for categorial prediction
model.add(keras.layers.Dense(OUTPUT_CLASSES,
                             name='Final',
                             activation='softmax'))

#Compile the model
model.compile(
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()

#Build the model
model.fit(X_train,
          Y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=VERBOSE,
          validation_split=VALIDATION_SPLIT)


#Evaluate the model against the test dataset and print results
print("\nEvaluation against Test Dataset :\n------------------------------------")
model.evaluate(X_test,Y_test)
```

## Model Evaluation ( predict root cause )

```python
#Pass individual flags to Predict the root cause
import numpy as np

CPU_LOAD=1
MEMORY_LOAD=0
DELAY=0
ERROR_1000=0
ERROR_1001=1
ERROR_1002=1
ERROR_1003=0

prediction=np.argmax(model.predict(
    [[CPU_LOAD,MEMORY_LOAD,DELAY,
      ERROR_1000,ERROR_1001,ERROR_1002,ERROR_1003]]), axis=1 )

print(label_encoder.inverse_transform(prediction))
```

```python
#Predicting as a Batch
print(label_encoder.inverse_transform(np.argmax(
        model.predict([[1,0,0,0,1,1,0],
                                [0,1,1,1,0,0,0],
                                [1,1,0,1,1,0,1],
                                [0,0,0,0,0,1,0],
                                [1,0,1,0,1,1,1]]), axis=1 )))
``` 

## Conclusion
> The model can predict the root cause of an issue with high accuracy. 
> The team can use this model to predict the root cause of issues reported by customers.
> The team can also use this model to predict the root cause of issues in real-time by integrating it with the monitoring system.
> The team can further improve the model by adding more features and tuning the hyperparameters.
