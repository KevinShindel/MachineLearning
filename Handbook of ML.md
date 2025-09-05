
## 1. Data Exploration
- DataFrame Inspection (head, tail, info, describe)
- Data Types Conversion (astype)
- Summary Statistics (mean, median, mode, std, var, min, max, quantile)
- Data Visualization (histograms, box plots, scatter plots, pair plots, heatmaps)
- Correlation Analysis (correlation matrix, heatmap)
- Missing Values Analysis (isnull, notnull, heatmap)
- Outlier Detection (box plots, z-score, IQR)
- Distribution Analysis (KDE plots, Q-Q plots)

## 2. Preprocessing
- Missing Values Handling (Imputation, Deletion)
- Categorical Encoding (One-Hot Encoding, Label Encoding, Target Encoding, Frequency Encoding)
- Feature Scaling (Standardization, Normalization, Robust Scaling)
- Feature Engineering (Polynomial Features, Interaction Features, Binning)
- Dimensionality Reduction (PCA, LDA, t-SNE, UMAP)
- Data Augmentation (for images, text, etc.)
- Outlier Detection and Treatment (IQR, Z-score, Percentile Capping)
- Data Splitting (train_test_split)
- Create feature for stratification (pd.cut, pd.qcut)
- Balancing the Dataset (SMOTE, ADASYN, RandomOverSampler, RandomUnderSampler) ? TODO: check this

## 3. Scoring
### For **Regression task** common metrics:
- **RMSE** using for Normal Distribution Data without outliers.
- **MAE** using for Data with outliers.
- **R² Score** from sklearn.metrics - for measuring the proportion of variance explained by the model
- **Mean Absolute Percentage Error (MAPE)** from sklearn.metrics - for measuring prediction accuracy as a percentage
- **Mean Squared Error (MSE)** from sklearn.metrics - for measuring the average squared difference between predicted and actual values

### For **Binary classification** tasks common metrics:
- **classification_report** from sklearn.metrics - for detailed classification metrics
- **confusion_matrix** from sklearn.metrics - for visualizing the performance of a classification model
- **roc_auc_score** from sklearn.metrics - for calculating the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
- **precision_recall_curve** from sklearn.metrics - for plotting precision-recall curve

### For **Multiclass classification** tasks common metrics:
- **classification_report** from sklearn.metrics - for detailed classification metrics
- **confusion_matrix** from sklearn.metrics - for visualizing the performance of a classification model
- **cross_val_predict** from sklearn.model_selection - for generating cross-validated estimates for each input data point

## 4. Validation 
- Hold-out validation (train_test_split)
- K-Fold Cross Validation (KFold)
- Stratified K-Fold Cross Validation (StratifiedKFold)
- Leave-One-Out Cross Validation (LeaveOneOut)
- Time Series Split (Cross-validation for time series data)
- Nested Cross Validation (for hyperparameter tuning and model selection)
- Group K-Fold Cross Validation (GroupKFold for grouped data)
- Repeated K-Fold Cross Validation (RepeatedKFold for more robust estimates)
- Bootstrap Sampling (Bootstrap for estimating the distribution of a statistic)
- Monte Carlo Cross Validation (Randomly splitting the data multiple times)

## 5. Feature Selection
- Filter Methods (Variance Threshold, Correlation Coefficient, Chi-Squared Test, ANOVA)
- Wrapper Methods (Recursive Feature Elimination, Sequential Feature Selection)
- Embedded Methods (Lasso Regression, Ridge Regression, Elastic Net, Tree-based Feature Importance)
- Dimensionality Reduction (PCA, LDA)
- Mutual Information
- Feature Importance from Models (Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Boruta Algorithm
- SHAP Values
- Permutation Importance
- SelectKBest, SelectPercentile
- Feature Selection based on Model Performance (Forward Selection, Backward Elimination)
- Stability Selection
- Genetic Algorithms for Feature Selection
- Clustering-based Feature Selection (e.g., using K-Means to group similar features)
- Correlation Matrix Analysis (to identify and remove highly correlated features)
- VIF (Variance Inflation Factor) for multicollinearity detection
- Domain Knowledge (leveraging expert knowledge to select relevant features)
- Recursive Feature Addition (starting with no features and adding one at a time based on model performance

## 6. Hyperparameter Tuning
- Grid Search (GridSearchCV)
- Random Search (RandomizedSearchCV)
- Bayesian Optimization (e.g., using libraries like Hyperopt, Optuna)
- Genetic Algorithms (e.g., using libraries like DEAP)
- Hyperband (for efficient hyperparameter optimization)
- Early Stopping (to prevent overfitting during training)
- Cross-Validation (to evaluate model performance during tuning)
- Learning Rate Schedulers (for models like neural networks)

## 7. Model Selection
- Compare multiple models using cross-validation scores
- Use ensemble methods (bagging, boosting, stacking) to combine models
- Consider model interpretability and complexity
- Evaluate models on a validation set before final testing
- Use statistical tests (e.g., paired t-test) to compare model performances
- Consider computational efficiency and scalability
- Analyze residuals and error distributions
- Use domain knowledge to select models that align with the problem context
- Evaluate models based on business metrics and objectives