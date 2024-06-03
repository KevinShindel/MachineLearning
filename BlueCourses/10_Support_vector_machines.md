# Problems with NN
- Multimodal objective function: multiple local minima
- Higly paramaterized: choose number of hidden layers, number of neurons in each layer, activation function, learning rate, etc.

# Linear Programming
- Linear programming is a method to achieve the best outcome in a mathematical model whose requirements are represented by linear relationships.
- Minimize the sum of the absolute values of the deviations from the target values. (MSD)
- Popular in early credit scoring (Fair, Isaac and Company

# Linear Separable Case
- The goal is to find the hyperplane that separates the two classes.
- Consider hyperplane, which minimize the distance between the two classes.
- Large margin separating hyperplane
- Given a set of training data, the SVM algorithm outputs an optimal hyperplane which categorizes new examples.
- Assume that the data is linearly separable.
- Maximize or minimize the margin between the two classes.
- Optimization problem
- The classifier then becomes a linear combination of the support vectors.
- using Lagrangian optimization, a quadratic programming problem is obtained.
- Solution of QP problem is global: convex optimization problem.
- Training points that lie on one of the hyperplanes are called support vectors.

# Linear Non-Separable Case
- Allow for errors by introducing slack variables in the inequality constraints.
- The optimization problem then becomes a quadratic programming problem.
- The solution is a soft margin classifier.

# Non-Linear SVM Classifier
- 

# Kernel Functions

# NN interpretation of SVM Classifier

# Tuning the hyperparameters

# Benchmarking Study

# SVMs for Regression

# One-CLass SVMs

# Extension to SVMs

# Opening SMV Black box