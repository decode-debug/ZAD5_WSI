
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Code.train.model import Model
from Code.train.training import TrainModel
from Code.Import_training_data.import_traing_data import DataLoader
from Code.Optymalize_Model_Size.Bayes_optymalization import BayesOptimizer
from Code.Visualization.network_visualization import show_architecture, show_training_progress

# ----------------------------------------------------------------------
# Load data once — shared by both manual and auto paths
# ----------------------------------------------------------------------
loader = DataLoader()
X_train, y_train = loader.get_train_data()
X_val,   y_val   = loader.get_val_data()
X_test,  y_test  = loader.get_test_data()

input_size  = X_train.shape[1]   # 64 pixel features
output_size = 10                  # digits 0-9


def train_with_config(num_layers, nodes, lr, epochs):
    """Build a model from the given config, train it, print test accuracy, return trainer."""
    layer_sizes = [input_size] + [nodes] * num_layers + [output_size]
    activations = ['relu'] * num_layers + ['softmax']

    print(f"\nArchitecture: {' -> '.join(str(s) for s in layer_sizes)}")
    print(f"Epochs: {epochs}  |  Learning rate: {lr}\n")

    model   = Model(layer_sizes, activations)
    trainer = TrainModel(model, learning_rate=lr)
    trainer.train(X_train, y_train, X_val, y_val, epochs=epochs)

    test_acc = trainer.accuracy(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    return trainer, layer_sizes, activations


def visualize(trainer, layer_sizes, activations):
    """Show interactive Plotly charts for architecture and training progress."""
    show_architecture(layer_sizes, activations)
    show_training_progress(trainer.loss_history, trainer.val_acc_history)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Select mode:")
    print("  1 - Manual  (enter architecture yourself)")
    print("  2 - Auto    (Bayesian optimisation finds the best config)")
    mode = input("Choice [1/2]: ").strip()

    if mode == "2":
        # --- Bayesian optimisation ---
        raw_iter = input("Number of optimisation iterations [25]: ").strip()
        n_iter   = int(raw_iter) if raw_iter else 25

        optimizer = BayesOptimizer(
            X_train, y_train,
            X_val,   y_val,
            n_random_starts = max(5, n_iter // 5),
            n_iterations    = n_iter,
        )
        best_config, best_val_acc, best_trainer = optimizer.optimise()

        # Reuse the already-trained model — no re-training from scratch
        print("\nEvaluating best model found during optimisation...")
        test_acc = best_trainer.accuracy(X_test, y_test)
        print(f"Val  Accuracy (from optimisation): {best_val_acc:.4f}")
        print(f"Test Accuracy                    : {test_acc:.4f}")

        cfg = best_config
        layer_sizes = [input_size] + [cfg['nodes']] * cfg['num_layers'] + [output_size]
        activations = ['relu'] * cfg['num_layers'] + ['softmax']
        final_trainer = best_trainer

    else:
        # --- Manual mode ---
        num_hidden_layers = int(input("Number of hidden layers: "))
        nodes_per_layer   = int(input("Nodes per hidden layer:  "))
        raw_epochs = input("Epochs          [100]: ").strip()
        raw_lr     = input("Learning rate  [0.01]: ").strip()

        epochs = int(raw_epochs)  if raw_epochs else 100
        lr     = float(raw_lr)    if raw_lr     else 0.01

        final_trainer, layer_sizes, activations = train_with_config(
            num_hidden_layers, nodes_per_layer, lr, epochs
        )

    # --- Optional: show Plotly visualizations ---
    raw_viz = input("\nShow Plotly visualizations? [y/N]: ").strip().lower()
    if raw_viz == 'y':
        visualize(final_trainer, layer_sizes, activations)
