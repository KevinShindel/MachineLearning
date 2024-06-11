# Sample Bias
- "Every sample is biased!"
- Biased in terms of:
- - Timing
- - Geography
- - Strategy
- - Population lift
- Sample bias not a problem but should be documented:
- - Relates to model perimeter
- - E.g. churn model developed for Belgium customers during economic upturn and aggressive 
    competitor campaign 

# Model Risk
- "All models are wrong, but some are useful"
- Every ML model is outdated, before it is event implemented!
- Need for continuous model monitoring (aka backtesting)
- Model risk
- Model bias

# Deep Everything
- Even the best cook (ML) with the best kitchen (hardware/software) cannot make a decent dish 
  (ML model) if the ingredients (data) are of inferior quality.
- Get your data, and especially its quality, sorted first.

# Leader versus Follower
- Critically evaluate the TCO and ROI of each ML investment
- Example: the NoSQL massacre -> MongoDB: twitter, HealthCare

# Complexity versus Trust Trade-Off
- Create a sound corporate ML mindset.
- Key role of the citizen ML/analytics translator.
- Mitigate complexity by using visualization.
- Visual analytics: the best way to communicate ML results.

# Statistical Myopia
- Who cares about likelihood, p-values, AIC, etc.
- What is business value?
- - Interpretability
- Operational Efficiency
- Compliance
- Profit

# Profit Driven Machine Learning
- Model selection: which model is the most profitable? 
- Winning model is often selected based on accuracy related performance measures 
- Profit maximization not taken into account
- Idea - create performance metric that takes profits into account.
- Next, incorporate this measure into construction of the classifier.
- EMPC can be used to select most profitable model.
- Profit not directly integrated into model construction.
- Build logic regression and tree-classifier that incorporate EMPC in model construction.
- ProfLogit and ProfTree take profit maximization into account during training step: 
  generic/evolutionary algorithm.
- Profit maximizing classification trees using evolutionary algorithms.
- Goal: find classification tree which optimizes tradeoff between performance and complexity.