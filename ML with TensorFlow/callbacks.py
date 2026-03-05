import tensorflow as tf
import math

from tensorflow.keras import callbacks
import os
import time

log_folder = 'logs'
root_logdir = os.path.join(os.curdir, log_folder)
K = tf.keras.backend

checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=os.path.join(root_logdir, "models/my_keras_model.keras"),
    save_best_only=True
)

early_stopping_callback = callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)

def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

# run_logdir = get_run_logdir() # example: "logs/run_2024_06_01-12_00_00"

tensorboard_callback = callbacks.TensorBoard(
    log_dir=get_run_logdir(),
    histogram_freq=1,
    profile_batch=0
)

class ExponentialLearningRate(callbacks.Callback):

    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        # get current LR
        lr = K.get_value(self.model.optimizer.learning_rate)
        self.rates.append(lr)
        self.losses.append(logs.get("loss"))
        # compute new LR
        new_lr = lr * self.factor
        # set new LR (works with modern Keras)
        self.model.optimizer.learning_rate.assign(new_lr)


exp_learning_rate = ExponentialLearningRate(factor=1.005)


class StopDecayAfterEpoch(callbacks.Callback):

    def __init__(self, initial_lr, decay_rate, stop_epoch, verbose=1):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.stop_epoch = stop_epoch
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.stop_epoch:
            new_lr = self.initial_lr * (self.decay_rate ** epoch)
        else:
            new_lr = self.initial_lr * (self.decay_rate ** self.stop_epoch)

        # Keras 2.13+/Keras 3: use `learning_rate` (not `.lr`)
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "assign"):
            lr.assign(new_lr)
        else:
            K.set_value(lr, new_lr)

        if self.verbose:
            print(f"\nEpoch {epoch+1}: Learning rate set to {new_lr:.6f}")


stop_decay_after_epoch = StopDecayAfterEpoch(
    initial_lr=1e-3,
    decay_rate=0.9,
    stop_epoch=10,
    verbose=1
)


class CosineAnnealingStopAfterEpoch(tf.keras.callbacks.Callback):
    def __init__(
        self,
        max_lr,
        min_lr,
        total_epochs,
        stop_epoch,
        verbose=1
    ):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.frozen_lr = None

    def cosine_lr(self, epoch):
        cos_inner = math.pi * epoch / self.total_epochs
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(cos_inner))

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.stop_epoch:
            lr_value = self.cosine_lr(epoch)
            self.frozen_lr = lr_value
        else:
            lr_value = self.frozen_lr

        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "assign"):
            lr.assign(lr_value)
        else:
            tf.keras.backend.set_value(lr, lr_value)

        if self.verbose:
            print(f"\nEpoch {epoch+1}: Learning rate set to {lr_value:.6f}")

cosine_annealing_stop_after_epoch = CosineAnnealingStopAfterEpoch(
    max_lr=1e-3,
    min_lr=1e-5,
    total_epochs=50,
    stop_epoch=20,
    verbose=1
)

class CyclicalLRStopAfterStep(tf.keras.callbacks.Callback):
    def __init__(
        self,
        min_lr,
        max_lr,
        step_size,
        stop_step,
        mode="triangular",
        gamma=1.0,
        verbose=0
    ):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.stop_step = stop_step
        self.mode = mode
        self.gamma = gamma
        self.verbose = verbose

        self.iteration = 0
        self.frozen_lr = None

    def clr(self):
        cycle = math.floor(1 + self.iteration / (2 * self.step_size))
        x = abs(self.iteration / self.step_size - 2 * cycle + 1)
        scale = max(0, (1 - x))

        if self.mode == "triangular2":
            scale /= (2 ** (cycle - 1))
        elif self.mode == "exp_range":
            scale *= (self.gamma ** self.iteration)

        return self.min_lr + (self.max_lr - self.min_lr) * scale

    def on_train_batch_begin(self, batch, logs=None):
        if self.iteration < self.stop_step:
            lr_value = self.clr()
            self.frozen_lr = lr_value
        else:
            lr_value = self.frozen_lr

        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "assign"):
            lr.assign(lr_value)
        else:
            K.set_value(lr, lr_value)

        self.iteration += 1

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            print(f"Epoch {epoch+1} ended. Current LR: {K.get_value(self.model.optimizer.learning_rate):.6f}", end='\n')


def get_clr(X_train, batch_size):
    steps_per_epoch = len(X_train) // batch_size
    return CyclicalLRStopAfterStep(
        min_lr=1e-5,
        max_lr=1e-3,
        step_size=2 * steps_per_epoch,
        stop_step=10 * steps_per_epoch,
        mode="triangular",
        gamma=1.0,
        verbose=1
    )

cyclical_lr_stop_after_step = get_clr

reduce_on_plateau_callback = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)


class OneCycleScheduler(tf.keras.callbacks.Callback):
    def __init__(
        self,
        iterations: int,
        max_rate: float,
        start_rate: float | None = None,
        last_iterations: int | None = None,
        last_rate: float | None = None,
    ) -> None:
        super().__init__()
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10.0
        self.last_iterations = last_iterations or iterations // 10
        self.last_rate = last_rate or self.start_rate / 1000.0

        remaining = iterations - self.last_iterations
        self.half_iteration = remaining // 2

        self.phase1_end = self.half_iteration
        self.phase2_end = 2 * self.half_iteration
        self.phase3_end = self.phase2_end + self.last_iterations

        self.iteration = 0

    def _interpolate(self, i, i1, i2, r1, r2):
        return r1 + (r2 - r1) * (i - i1) / (i2 - i1)

    def on_train_batch_begin(self, batch, logs=None):
        i = min(self.iteration, self.iterations)

        if i < self.phase1_end:
            rate = self._interpolate(i, 0, self.phase1_end, self.start_rate, self.max_rate)
        elif i < self.phase2_end:
            rate = self._interpolate(
                i, self.phase1_end, self.phase2_end, self.max_rate, self.start_rate
            )
        else:
            rate = self._interpolate(
                i, self.phase2_end, self.phase3_end, self.start_rate, self.last_rate
            )

        self.iteration += 1

        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "assign"):
            lr.assign(rate)
        else:
            self.model.optimizer.learning_rate = rate

one_cycle_scheduler = OneCycleScheduler(
    iterations=1000,
    max_rate=1e-3,
    start_rate=1e-4,
    last_iterations=100,
    last_rate=1e-5
)