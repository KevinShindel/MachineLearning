

### MLflow Optuna integration

1. Enables automatic logging of HPO for each trial, streaming exp. tracking
2. Parent-child Run organization by supporting for hierarchical run

```python
from optuna.integration.mlflow import MLflowCallback

mlflow_cb = MLflowCallback(
    tracking_uri = 'databricks',
    create_experiment = False,
    mlflow_kwargs = {
        "experiment_id" : <id>,
        "Nested": True
    }
)

study.optimize(
    objective,n_trials = N,
    callbacks = [mlflow_cb]
)
```


### Search Space Definitions 


```python
import optuna

trial = optuna.trial.Trial()

# Suggest integer hyperparameter (e.g., number of layers)
int_hyper_parameter = trial.suggest_int('n_layers', 1, 5)

# Suggest float hyperparameter (e.g., dropout rate)
float_hyper = trial.suggest_float('dropout_rate', 0.0, 0.5)

# Suggest uniform distribution (e.g., learning rate low-high)
uniform_hyper = trial.suggest_uniform('learning_rate', 0.0001, 0.1)

# Suggest log-uniform distribution (better for exponential ranges, e.g., batch size)
loguniform_hyper = trial.suggest_loguniform('batch_size', 8, 256)

# Suggest categorical choice (e.g., optimizer type)
categorical_hyper = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])

# Real case example: Neural Network hyperparameter tuning
def objective(trial):
    n_layers = trial.suggest_int('n_layers', 2, 5)  # 2 to 5 layers
    dropout = trial.suggest_float('dropout', 0.2, 0.5)  # 20% to 50% dropout
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-2)  # Log-uniform for learning rate
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])  # Choose optimizer
    
    model = build_model(n_layers=n_layers, dropout=dropout)
    accuracy = train_model(model, lr=learning_rate, optimizer=optimizer)
    return accuracy
```


### Pruning Strategy

Pruning stops unpromising trials early to save computational resources. Use `.should_prune()` method to check if trial should be pruned.

```python
import optuna

# ============ PRUNER STRATEGIES ============

# 1. NopPruner - No pruning (default baseline)
# Use when: Testing, you want all trials to complete, or have sufficient computational resources
pruner = optuna.pruners.NopPruner()

# 2. MedianPruner - Prunes trials with value below median of completed trials
# Use when: Standard HPO, looking for quick convergence, want simple & effective approach
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

# 3. PercentilePruner - Prunes trials below a certain percentile (e.g., 25th percentile)
# Use when: More aggressive pruning needed, want to focus on top-performing trials early
pruner = optuna.pruners.PercentilePruner(percentile=25, n_startup_trials=5)

# 4. SuccessiveHalvingPruner - Successive Halving algorithm (reduces candidates by half)
# Use when: Full budget is known in advance, resource allocation is uniform, want structured approach
pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, max_resource=100)

# 5. HyperbandPruner - Hyperband algorithm (adaptive resource allocation)
# Use when: Optimal pruning efficiency needed, budget is limited, want to balance exploration-exploitation
pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100)

# 6. PatientPruner - Waits for k consecutive trials before pruning (tolerates high variance)
# Use when: High variance objective function, early stopping is unreliable, need more patience with trials
pruner = optuna.pruners.PatientPruner(patience=3, wrapped_pruner=optuna.pruners.MedianPruner())

# 7. ThresholdPruner - Prunes trials that drop below absolute threshold
# Use when: You know the minimum acceptable performance threshold beforehand
pruner = optuna.pruners.ThresholdPruner(lower=0.5)

# 8. PBTSuccessiveHalvingPruner - Population Based Training with Successive Halving
# Use when: Distributed optimization with population-based approach, want adaptive hyperparameter updates
pruner = optuna.pruners.PopulationBasedTrainingPruner()

# ============ PRACTICAL EXAMPLE ============

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 2, 5)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    
    accuracy = train_model(n_layers=n_layers, dropout=dropout, lr=learning_rate)
    
    # Check if trial should be pruned (usually called at interval steps)
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return accuracy

# Create study with pruner strategy
study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(),  # Recommended for most cases
    direction='maximize'
)

study.optimize(objective, n_trials=100)
```