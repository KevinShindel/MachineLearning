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
# Model Ethics

# Privacy and Security