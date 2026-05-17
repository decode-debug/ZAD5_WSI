"""
Bayesian Optimisation for hyperparameter search.

What it does:
  1. Tries a few random configurations first (exploration phase).
  2. Fits a Gaussian Process (GP) surrogate model on the results so far.
  3. Uses the GP to predict which configuration is likely to be the best
     (this is the "acquisition" step — it balances exploration vs. exploitation).
  4. Trains a real neural network with that configuration and records validation accuracy.
  5. Repeats until the iteration budget is exhausted.
  6. Returns the best configuration found.

Search space (tunable at construction time):
  - num_hidden_layers  : how many hidden layers
  - nodes_per_layer    : neurons in each hidden layer
  - learning_rate      : step size for gradient descent
  - epochs             : training length
"""

import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Code.train.model import Model
from Code.train.training import TrainModel


# -----------------------------------------------------------------------
# Gaussian Process — a simple surrogate that learns "score vs. config"
# -----------------------------------------------------------------------
class GaussianProcess:
    def __init__(self, noise=1e-3):
        self.noise = noise   # small regularisation to keep the matrix invertible
        self.X_seen = []     # configs tried so far (as numeric vectors)
        self.y_seen = []     # scores observed so far

    def _rbf_kernel(self, a, b, length_scale=1.0):
        """Radial Basis Function kernel — measures similarity between two points."""
        diff = a - b
        return np.exp(-0.5 * np.dot(diff, diff) / (length_scale ** 2))

    def _kernel_matrix(self, X1, X2):
        """Build the full kernel matrix K(X1, X2)."""
        return np.array([
            [self._rbf_kernel(x1, x2) for x2 in X2]
            for x1 in X1
        ])

    def fit(self, X, y):
        """Remember all observed (config, score) pairs."""
        self.X_seen = np.array(X, dtype=float)
        self.y_seen = np.array(y, dtype=float)

    def predict(self, X_new):
        """
        Return (mean, std) for each candidate in X_new.
        mean  → expected score
        std   → uncertainty (high = we haven't explored here yet)
        """
        X_new = np.array(X_new, dtype=float)
        n = len(self.X_seen)

        # Kernel between all seen points
        K      = self._kernel_matrix(self.X_seen, self.X_seen)
        K_reg  = K + self.noise * np.eye(n)
        K_inv  = np.linalg.inv(K_reg)

        means, stds = [], []
        for x in X_new:
            # Kernel between new point and all seen points
            k = np.array([self._rbf_kernel(x, x_s) for x_s in self.X_seen])

            mean = float(k @ K_inv @ self.y_seen)
            var  = max(0.0, 1.0 - float(k @ K_inv @ k))
            means.append(mean)
            stds.append(np.sqrt(var))

        return np.array(means), np.array(stds)


# -----------------------------------------------------------------------
# Acquisition function — Upper Confidence Bound (UCB)
# Picks the candidate with the highest (mean + kappa * std).
# High kappa → more exploration; low kappa → more exploitation.
# -----------------------------------------------------------------------
def acquisition_ucb(means, stds, kappa=2.0):
    return means + kappa * stds


# -----------------------------------------------------------------------
# Main Bayesian Optimisation class
# -----------------------------------------------------------------------
class BayesOptimizer:
    def __init__(
        self,
        X_train, y_train,
        X_val,   y_val,
        # Search space bounds
        layers_range      = (1, 4),     # min/max hidden layers
        nodes_range       = (16, 256),  # min/max nodes per hidden layer
        lr_range          = (1e-3, 0.1),
        epochs_range      = (50, 200),
        # Budget
        n_random_starts   = 5,          # pure random trials before using GP
        n_iterations      = 20,         # total trials (random + GP-guided)
        # GP candidates sampled per iteration
        n_candidates      = 200,
    ):
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val

        self.bounds = {
            'num_layers': layers_range,
            'nodes'     : nodes_range,
            'lr'        : lr_range,
            'epochs'    : epochs_range,
        }
        self.max_layers      = layers_range[1]   # upper bound used for fixed-length encoding
        self.n_random_starts = n_random_starts
        self.n_iterations    = n_iterations
        self.n_candidates    = n_candidates

        self.gp = GaussianProcess()

        # History — list of dicts with config + score
        self.history = []

    # -------------------------------------------------------------------
    # Convert a config dict to a normalised numeric vector for the GP
    # -------------------------------------------------------------------
    def _encode(self, config):
        lo_l, hi_l = self.bounds['num_layers']
        lo_n, hi_n = self.bounds['nodes']
        lo_lr, hi_lr = self.bounds['lr']
        lo_e, hi_e = self.bounds['epochs']

        vec = [
            (config['num_layers'] - lo_l)  / max(hi_l  - lo_l,  1),
            (config['lr']         - lo_lr) / (hi_lr - lo_lr),
            (config['epochs']     - lo_e)  / max(hi_e  - lo_e,  1),
        ]
        for i in range(1, self.max_layers + 1):
            vec.append((config[f'nodes_{i}'] - lo_n) / max(hi_n - lo_n, 1))
        return np.array(vec)

    # -------------------------------------------------------------------
    # Draw a uniformly random config from the search space
    # -------------------------------------------------------------------
    def _random_config(self):
        cfg = {
            'num_layers': np.random.randint(*self.bounds['num_layers']),
            'lr'        : np.random.uniform(*self.bounds['lr']),
            'epochs'    : np.random.randint(*self.bounds['epochs']),
            'batch_size': np.random.randint(16, 129),
        }
        for i in range(1, self.max_layers + 1):
            cfg[f'nodes_{i}'] = np.random.randint(*self.bounds['nodes'])
        return cfg

    # -------------------------------------------------------------------
    # Train a model with a given config, return validation accuracy
    # -------------------------------------------------------------------
    def _evaluate(self, config):
        input_size  = self.X_train.shape[1]
        output_size = 10

        hidden_sizes = [config[f'nodes_{i+1}'] for i in range(config['num_layers'])]
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        activations = ['relu'] * config['num_layers'] + ['softmax']

        model   = Model(layer_sizes, activations)
        trainer = TrainModel(model, learning_rate=config['lr'])
        trainer.train(
            self.X_train, self.y_train,
            self.X_val,   self.y_val,
            epochs=config['epochs'],
            batch_size=config.get('batch_size', 32),
            verbose=False,   # suppress per-epoch output during search
        )
        score = trainer.model.accuracy(self.X_val, self.y_val)
        # Return both so the best trained model can be reused without re-training
        return score, trainer

    # -------------------------------------------------------------------
    # Run the full optimisation loop
    # -------------------------------------------------------------------
    def optimise(self):
        print("=" * 60)
        print("  Bayesian Optimisation — searching for best architecture")
        print("=" * 60)

        for iteration in range(1, self.n_iterations + 1):
            # --- Phase 1: random exploration ---
            if iteration <= self.n_random_starts:
                config = self._random_config()
                how    = "random"

            # --- Phase 2: GP-guided search ---
            else:
                # Generate many random candidates
                candidates = [self._random_config() for _ in range(self.n_candidates)]
                encoded    = np.array([self._encode(c) for c in candidates])

                # Ask the GP which candidate looks most promising
                X_hist = [self._encode(h['config']) for h in self.history]
                y_hist = [h['score']                for h in self.history]
                self.gp.fit(X_hist, y_hist)

                means, stds = self.gp.predict(encoded)
                ucb_scores  = acquisition_ucb(means, stds)
                config      = candidates[int(np.argmax(ucb_scores))]
                how         = "GP-guided"

            # --- Evaluate the chosen config ---
            nodes_str = ', '.join(str(config[f'nodes_{i+1}']) for i in range(config['num_layers']))
            print(
                f"\n[{iteration:>2}/{self.n_iterations}] {how} | "
                f"layers={config['num_layers']}, nodes=[{nodes_str}], "
                f"lr={config['lr']:.4f}, epochs={config['epochs']}"
            )
            score, trainer = self._evaluate(config)
            print(f"  → Val accuracy: {score:.4f}")

            self.history.append({'config': config, 'score': score, 'trainer': trainer})

        # --- Return the single best config and its already-trained model ---
        best = max(self.history, key=lambda h: h['score'])
        print("\n" + "=" * 60)
        print("  Best configuration found:")
        best_nodes_str = ', '.join(str(best['config'][f'nodes_{i+1}']) for i in range(best['config']['num_layers']))
        print(f"    Hidden layers : {best['config']['num_layers']}")
        print(f"    Nodes/layer   : [{best_nodes_str}]")
        print(f"    Learning rate : {best['config']['lr']:.5f}")
        print(f"    Epochs        : {best['config']['epochs']}")
        print(f"    Val accuracy  : {best['score']:.4f}")
        print("=" * 60)
        return best['config'], best['score'], best['trainer']
