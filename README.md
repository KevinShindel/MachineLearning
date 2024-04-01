# Machine Learning Project Documentation

## Table of Contents
1. [EngX AI Learning](#module-1)
2. [Pandas ETL](#module-2)
3. [SciKit](#module-3)
4. [SparkML](#module-4)
5. [Deep Learning](#module-5)

## Module 1

In this module, we delve into the world of Language Learning Models (LLMs), exploring their potential to enhance software development processes. The module is divided into several key areas:

### Understanding LLMs
We start by understanding what LLMs are and how they function. LLMs are a type of machine learning model that uses large amounts of data to develop models for understanding text. They process natural language inputs and predict the next word based on what they have already seen. LLMs can also generate new texts based on a given prompt or context.

### Communicating with LLMs
We learn how to effectively communicate with conversational and inline tools, like ChatGPT and GitHub Copilot. This involves understanding how to craft prompts and provide context that the LLMs can understand and respond to effectively.

### Practical Applications of LLMs
We explore the practical applications of LLMs in software development. This includes understanding how they can be used to accelerate new feature creation, streamline maintenance, and improve refactoring processes.

### Hands-on Practice
Throughout the module, there are opportunities to practice what you have learned. This includes tasks that help you identify possible areas for growth in communication with AI tools skills.

By the end of this module, you will have a solid understanding of LLMs and how to leverage them in your software development processes.


## Module 2

This module focuses on the extraction, transformation, and loading (ETL) of data using the pandas library in Python. It consists of several scripts, each performing a specific ETL task.

### movies_film_join.py
This script reads raw data from CSV files, including user data, movie ratings, and movie details. It then merges these dataframes to create a combined dataframe. The script also calculates the mean rating for each movie by gender, filters movies with more than 250 ratings, and calculates the rating difference between genders. Finally, it calculates the standard deviation of the rating grouped by the movie title.

### usa_food_data_calories.py
This script investigates USA food data. It reads a JSON file containing food data, counts food groups, and plots the top 10 food groups. It also explodes the nutrition data and normalizes it. The script then merges the normalized nutrition data with the original food data, calculates the median value of each nutrient group, and plots the result.

### usa_naming_investigation.py
This script investigates USA naming data. It reads data from multiple CSV files, calculates the total number of births per year, and plots the result. The script also calculates the proportion of births for each name, gets the top 1000 names for each sex and year, and investigates tendencies in names. It also investigates the diversity of names in the top 50% of births, the last letter revolution, and names that have converted from male to female.

By the end of this module, you will have a solid understanding of how to perform ETL tasks using pandas in Python, and how to analyze and visualize the results.


# Module 3

This project focuses on the application of various machine learning models and techniques using the Scikit-learn library in Python. It consists of several scripts, each demonstrating a specific machine learning task.

## Logistic Regression: Titanic Training

This script (`logistic_regression/titanic_training.py`) applies logistic regression to the Titanic dataset. It includes data cleaning, feature engineering, model training, and prediction. The script also includes a function to predict the survival of a random passenger.

## Decision Tree: Zoo Predictor

The `d_tree/ZooPredictor.py` script uses a decision tree classifier to predict whether an animal is eatable based on various features. The script includes a class `ZooPredictor` that encapsulates the model, features, target, and data path. The class also includes methods for training the model and making predictions.

## Keepa Price Prediction

The `keepa_price_prediction.py` script demonstrates time series analysis and forecasting on the Keepa dataset. It includes data cleaning, outlier detection, seasonal decomposition, and forecasting using the Holt-Winters method.

Remember to replace the placeholder descriptions with actual descriptions of what each script does. This documentation provides a comprehensive overview of the `sci-kit_library` project. It can be further expanded with more specific details about each script as you progress through the project.


# Module 4

This project focuses on the application of various machine learning models and techniques using the SparkML library in Python. It consists of several scripts, each demonstrating a specific machine learning task.

## Linear Regression: House Rent Prediction

The `HouseRent.py` script applies linear regression to predict house rent based on various features. It includes data cleaning, feature engineering, model training, and prediction. The script also calculates the error in prediction and saves the predicted values along with the error in a CSV file.

## Linear Regression: Temperature Estimation

The `TemperatureEstimation.py` script uses linear regression to estimate the temperature in Fahrenheit based on the temperature in Celsius. It includes data cleaning, model training, and prediction. The script also calculates the accuracy of the model.

## Logistic Regression: Hepatitis Classifier

The `HepatitClassifier.py` script uses logistic regression to predict whether a patient has Hepatitis based on various features. It includes data cleaning, feature engineering, model training, and prediction. The script also calculates the accuracy, precision, recall, and F1 score of the model.

## Service: Spark Session Creation

The `service.py` script includes a function to create a Spark session with various configurations. This function is used in other scripts to create a Spark session.

Remember to replace the placeholder descriptions with actual descriptions of what each script does. This documentation provides a comprehensive overview of the `sparkml_examples` project. It can be further expanded with more specific details about each script as you progress through the project.

# Module 5

This module focuses on deep learning techniques and their applications in various domains. It consists of several scripts, each demonstrating a specific deep learning task.
