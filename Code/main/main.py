
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Code.train.model import Model
from Code.train.training import TrainModel
from Code.Import_training_data.import_traing_data import DataLoader
from Code.Optymalize_Model_Size.Bayes_optymalization import BayesOptimizer
from Code.Visualization.network_visualization import show_architecture, show_architecture_3d, show_training_progress

# ----------------------------------------------------------------------
# Load data once — shared by both manual and auto paths
# ----------------------------------------------------------------------
X_train, y_train = DataLoader.get_train_data()
X_val,   y_val   = DataLoader.get_val_data()
X_test,  y_test  = DataLoader.get_test_data()


input_size  = X_train.shape[1]    # 64 pixel features
output_size = 10                  # digits 0-9


def train_with_config(num_layers, nodes, lr, epochs, batch_size=32, config=0):
    """Build a model from the given config, train it, print test accuracy, return trainer."""
    if isinstance(nodes, list):
        hidden_sizes = nodes
    else:
        hidden_sizes = [nodes] * num_layers
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    activations = ['relu'] * num_layers + ['softmax']
    if config == 0:
        print(f"\nArchitecture: {' -> '.join(str(s) for s in layer_sizes)}")
        print(f"Epochs: {epochs}  |  Learning rate: {lr}  |  Batch size: {batch_size}\n")

    model   = Model(layer_sizes, activations)
    trainer = TrainModel(model, learning_rate=lr)
    trainer.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, verbose=False)

    test_acc = trainer.model.accuracy(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    return trainer, layer_sizes, activations


def visualize(trainer, layer_sizes, activations):
    """Show interactive Plotly charts for architecture and training progress."""
    print("\nSelect visualization:")
    print("  1 - 2D architecture diagram")
    print("  2 - 3D architecture  (drag to orbit, scroll to zoom)")
    print("  3 - Training progress (loss + accuracy curves)")
    print("  4 - All of the above")
    choice = input("Choice [1/2/3/4]: ").strip()

    if choice in ('1', '4'):
        show_architecture(layer_sizes, activations)
    if choice in ('2', '4'):
        show_architecture_3d(layer_sizes, activations)
    if choice in ('3', '4'):
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

        print("\nRe-training 5× with the best config to reduce randomness...")
        best_nodes = [best_config[f'nodes_{i+1}'] for i in range(best_config['num_layers'])]
        best_val_acc_retrain = -1
        for _ in range(5):
            trainer, _ls, _acts = train_with_config(
                best_config['num_layers'], best_nodes, best_config['lr'], best_config['epochs'],
                batch_size=best_config['batch_size'], config=1,
            )
            acc = trainer.model.accuracy(X_val, y_val)
            if acc > best_val_acc_retrain:
                best_val_acc_retrain = acc
                best_trainer         = trainer

        best_test_acc = best_trainer.model.accuracy(X_test, y_test)
        print(f"Val  Accuracy (best of 5 re-runs): {best_val_acc_retrain:.4f}")
        print(f"Test Accuracy                    : {best_test_acc:.4f}")
        print(f"Batch size                       : {best_config['batch_size']}")

        # --- Optional: save model to disk ---
        raw_save = input("\nSave best model to disk? [y/N]: ").strip().lower()
        if raw_save == 'y':
            MODELS_DIR = ROOT_DIR / 'saved_models'
            MODELS_DIR.mkdir(exist_ok=True)
            save_path = MODELS_DIR / 'bayes_best_model'
            best_trainer.model.save(str(save_path))
            print(f"Model saved to {save_path}.npz")

        cfg = best_config
        hidden = best_nodes
        layer_sizes = [input_size] + hidden + [output_size]
        activations = ['relu'] * cfg['num_layers'] + ['softmax']
        final_trainer = best_trainer

    else:
        # --- Manual mode ---
        num_hidden_layers = int(input("Number of hidden layers: "))

        raw_nodes = input(f"Nodes per layer (e.g. '128 64 32' for different sizes, or single '64' for all equal): ").strip()
        parts = raw_nodes.split()
        if len(parts) == 1:
            nodes_per_layer = int(parts[0])
        else:
            nodes_per_layer = [int(x) for x in parts]
            if len(nodes_per_layer) != num_hidden_layers:
                nodes_per_layer = (nodes_per_layer + [nodes_per_layer[-1]] * num_hidden_layers)[:num_hidden_layers]

        raw_epochs = input("Epochs          [100]: ").strip()
        raw_lr     = input("Learning rate  [0.01]: ").strip()

        epochs = int(raw_epochs)   if raw_epochs else 100
        lr     = float(raw_lr)     if raw_lr     else 0.01

        raw_batch_size = input("Batch size     [32]: ").strip()
        batch_size = int(raw_batch_size) if raw_batch_size else 32

        raw_runs = input("Number of training runs [1]: ").strip()
        n_runs   = int(raw_runs) if raw_runs else 1

        if n_runs == 1:
            final_trainer, layer_sizes, activations = train_with_config(
                num_hidden_layers, nodes_per_layer, lr, epochs, batch_size=batch_size
            )
        else:
            print(f"\nTraining {n_runs}× — keeping best by val accuracy...")
            best_val_acc_manual = -1
            for run in range(1, n_runs + 1):
                print(f"\n--- Run {run}/{n_runs} ---")
                trainer, layer_sizes, activations = train_with_config(
                    num_hidden_layers, nodes_per_layer, lr, epochs,
                    batch_size=batch_size, config=1,
                )
                acc = trainer.model.accuracy(X_val, y_val)
                print(f"Val Accuracy: {acc:.4f}")
                if acc > best_val_acc_manual:
                    best_val_acc_manual = acc
                    final_trainer       = trainer
            best_test_acc_manual = final_trainer.model.accuracy(X_test, y_test)
            print(f"\nBest Val  Accuracy (of {n_runs} runs): {best_val_acc_manual:.4f}")
            print(f"Test Accuracy                        : {best_test_acc_manual:.4f}")

        # --- Optional: save model to disk ---
        raw_save = input("\nSave best model to disk? [y/N]: ").strip().lower()
        if raw_save == 'y':
            MODELS_DIR = ROOT_DIR / 'saved_models'
            MODELS_DIR.mkdir(exist_ok=True)
            save_path = MODELS_DIR / 'manual_best_model'
            final_trainer.model.save(str(save_path))
            print(f"Model saved to {save_path}.npz")
    # --- Optional: show Plotly visualizations ---
    raw_viz = input("\nShow Plotly visualizations? [y/N]: ").strip().lower()
    if raw_viz == 'y':
        visualize(final_trainer, layer_sizes, activations)
