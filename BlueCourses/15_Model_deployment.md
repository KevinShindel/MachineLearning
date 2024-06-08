# Model Deployment
- Deploy model into production
- **Integrate with other business applications**:
- - risk based pricing (loan, insurance)
- - campaign management (marketing)
- - equity calculation (investment)
- - fraud detection (credit card, insurance)
- **Deployment strategies**:
- - direct changeover: replace old system by new analytical model, risky, shock effect
- - parallel run: run new model in parallel with old system, compare results, less risky
- - phased deployment: months 1-3: credit card, months 4-6: insurance, months 7-9: loan
- Only small fraction of actual machine code is real-world machine learning model
- Vast array of surrounding IT infrastructure and processes 
- Key challenges relate to data dependencies, model complexity, reproducibility, testing, 
  monitoring and dealing with changes in external world.
- Lineage of data dependencies: variuos data sources, metadata management, ensuring data + 
  preprocessing
- **Deployment context**:
- - ML model deployed as API, embedded in WEB/ mobile app
- - Scheduled to run every hour, day, week, month
- - ML development env (Jupiter, Anaconda) vs IT env (Java, C++, .NET)
- - Keep model changes in sync with app changes
- **Deployment governance**:
- - Training code well-documented, versioned, reproducible
- - Collaboration + versioning
- - Can training code be easily reproduced?
- - Runs on my machine phenomenon
- Model governance:
- - How the model deployed?
- - Is versioning supported for provided models?
- - Models as data: can output of one model be easily used in other models/projects?
- - Is metadata available?
- **Monitoring**:
- - Inputs, outputs, performance metrics
- - Do we know when data is changing, when output probabilities are changing?
- - An errors reported?
- Issues well known in traditional software engineering: 
- - testing, monitoring, logging.
- - Continuous development, logging, integration, CI/CD
- In ML production, many of these are hard to apply:
- - ML models degrade silently
- - Data definitions change, people take actions based on model output.
- - Models will continue to provide predictions, but as concept drift increases, generalization 
    power decreases
- - Solid model governance infrastructure is needed
- Models In-house - Uber's Michelangelo, Airbnb's Bighead, Netflix's Metaflow, Facebook's 
  FBLearner, Spotify's Luigi

# Model Governance
- Trust in analytical model across all management levels key to success
- Directors and senior management involved in implementation and monitoring process
- Senior management responsible for sound governance
- Board and senior management should have general understanding of analytical models
- Active involvement on ongoing basis, assign clear responsibilities.
- Outcome of monitoring and stress testing exercises must be communicated to senior management.
- Chief Analyst Officer (CAO) responsible for model governance.
- Centralized vs de-centralized model governance.
- Corporate wide Analytical Center of Excellence: platform, regulatory guidelines, privacy 
  standards.
- Small teams of data scientists (e.g. 3-5) embedded in respective business units.
- Model audit teams.

# Model Documentation
- All steps of model development and monitoring process should be adequately documented.
- Documentation should be transparent and comprehensive.
- Both for internal and external models.
- Use documented management system with appropriate versioning facilities.
- Documentation test: can new team use existing documentation to continue development or  
  production of existing analytical models?

# Model Backtesting
- Backtesting: 
- - testing model on historical data.
- - contrast ex-post observed really with ex-ante predictions
- - performance monitoring.

# Model Benchmarking
- Comparison of internal estimates with benchmark.
- Benchmark can be internal or external: data poolers, industry standard, rating agencies.
- Benchmark can be analytical or expert based.
- Champion-challenger approach: current model = champion, new model = challenger.
- 
# Model Stress Testing
- Analytical models more and more used for strategic dicisioning: pricing, provisioning, equity calc
- Stress testing: understand impact of adverse economic scenarios on analytical outputs.
- Sensitivity test:
- - Single factor: impact of competitor action on churn rates
- - Multifactor: impact of income drop and unemployment rise on credit scores.
- Scenario test:
- - Historical: 2008 financial crisis
- - Hypothetical: 3 periods of GPD contraction.
- Stress Testing Governance:
- - define scope
- - define ownership 
- - define contributors
- - present results to senior management
- - actions and strategies to mitigate risks
- - public disclosure
- - documentation

# Model Ethics
- Cannot use certain variables because of ethical reasons (cannot use gender, age, ethnicity... 
  to discriminate between good and bad credit risk)
- Be aware of latent discrimination: correlated variables, years client versus age, income 
  versus gender. 
- Need for white-box, interpretable models.
- Trade-off between model performance and model interpretability.
- Might depend on context (e.g. credit risk)
- Ethical impact:
- - developed countries: young people and immigrants face troubles because of lack of credit 
    history
- - developing countries: historical financial data often non-existent.
- - access to small credits has social impact.
- - call data provides an alternative for credit scoring.
- - financial inclusion.
- Privacy impact: 
- - data sharing
- - explicit authorization from user
- - possibility to opt out and right to be forgotten
- - GDPR

# Privacy and Security
- Business vs ML:
- - Business - ownership of ML model
- - ML: only data useful for ML models
- Data Security: set of policies and techniques to ensure confidentiality, availability and 
  integrity of data..
- Data privacy: parties accessing and using data can do so only in ways that comply with agreed 
  upon purposes of data use in their role.
- Security necessary instrument to guarantee data privacy.

- Data security: 
- - guaranteeing data integrity
- - guaranteeing data availability
- - authentication and access control
- - guaranteeing confidentiality
- - auditing
- - mitigating vulnerabilities

- Responsible: responsible for developing ML models.
- Accountable: delegate work and decide what should be done
- Consulted: experts who advise business and ML teams
- Informed: should be kept up-to-date.

- Access internal data via data access request
- Internal privacy commission investigates: which variables are sensitive, which variables and 
  instances should be shared, which user should be authorized to access data
- Actions: Anonymization, SQL views.

- Anonymization: transform sensitive data so exact value cannot be covered
- Types of keys: 
- - natural key: reveals identity of instance (e.g. VAT number)
- - technical key: conservation of natural key to protect true identity of instance.

- Anonymization techniques:
- - Aggregation: summary statistics
- - Dis-cretization: partition numeric variable into disjoint, mutually-exclusive classes.
- - Value distortion: return x+ e instead of x
- - Generalization: generalize into less specific but semantically consistent description.