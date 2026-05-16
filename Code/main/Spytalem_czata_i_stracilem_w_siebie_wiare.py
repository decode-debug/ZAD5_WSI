import sys
from pathlib import Path
from sklearn.neural_network import MLPClassifier

# ----------------------------------------------------------------------
# 1. Setup Root Directory for Imports
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import ONLY the DataLoader (assuming it's the static version we built)
from Code.Import_training_data.import_traing_data import DataLoader

def main():
    # ----------------------------------------------------------------------
    # 2. Load Data (Augmentation and Normalization handled internally)
    # ----------------------------------------------------------------------
    print("Loading and preparing datasets...")
    X_train, y_train = DataLoader.get_train_data()
    X_val,   y_val   = DataLoader.get_val_data()
    X_test,  y_test  = DataLoader.get_test_data()

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")

    # ----------------------------------------------------------------------
    # 3. Define the Model
    # ----------------------------------------------------------------------
    print("\nInitializing MLPClassifier with 'lbfgs' solver...")
    model = MLPClassifier(
        solver='lbfgs',             # Optimal solver for small datasets (< 10k samples)
        hidden_layer_sizes=(128,),  # Expanding the bottleneck (64 input pixels -> 128 neurons)
        activation='relu',          # Activation function (consider testing 'tanh' later)
        max_iter=2000,              # 'lbfgs' often requires more iterations to fully converge
        tol=1e-4,                   # Tolerance for the optimization process
        alpha=0.0001,               # L2 penalty (regularization) to prevent overfitting
        random_state=42             # Seed for reproducible results
    )

    # ----------------------------------------------------------------------
    # 4. Train the Model
    # ----------------------------------------------------------------------
    print("Training the model (this might take a moment with lbfgs)...")
    model.fit(X_train, y_train)

    # ----------------------------------------------------------------------
    # 5. Evaluate the Model
    # ----------------------------------------------------------------------
    # Checking training accuracy helps determine if the model is memorizing data
    train_score = model.score(X_train, y_train)
    print(f"\n--- Results ---")
    print(f"Training Accuracy:   {train_score * 100:.2f}%")

    # Checking validation accuracy shows true generalization performance
    val_score = model.score(X_val, y_val)
    print(f"Validation Accuracy: {val_score * 100:.2f}%")

if __name__ == "__main__":
    main()