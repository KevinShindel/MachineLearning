| Model                       | Optimization Library    | Learning Time | MAE        | RMSE       | MSE        | R2         | # Evaluations | 
|-----------------------------|-------------------------|---------------|------------|------------|------------|------------|---------------|
| Random Forest Regressor     | N\A                     | 00:00:00      | 0.27       | 0.35       | 0.1243     | 0.989      | N/A           |
| Random Forest Regressor     | RandomSearchGrid        | 00:00:26      | 0.4886     | 0.5678     | 0.322      | 0.9735     | 150           |
| Random Forest Regressor     | GridSearchCV            | 00:04:43      | 0.2702     | 0.3414     | 0.1165     | 0.9904     | 1875          |
| Random Forest Regressor     | HalvingRandomSearchCV   | 00:00:24      | 0.733      | 0.8816     | 0.7773     | 0.9363     | ?             |
| **Random Forest Regressor** | **HalvingGridSearchCV** | **00:06:23**  | **0.2687** | **0.3387** | **0.1147** | **0.9905** | ?             |
| Random Forest Regressor     | Optuna                  | 00:00:29      | 0.28       | 0.351      | 0.123      | 0.9898     | 50            | 
| Random Forest Regressor     | Genetic Algorithm       | 00:01:22      | 0.2837     | 0.3483     | 0.1213     | 0.99       | 50            |
| Random Forest Regressor     | Hyperopt                | 00:01:09      | 0.3239     | 0.39       | 0.154      | 0.987      | 50            |