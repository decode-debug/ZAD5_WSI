
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Code.Import_training_data.import_traing_data import DataLoader

def main():
    data_loader = DataLoader()
    X_train, y_train = data_loader.get_train_data()
    X_val, y_val = data_loader.get_val_data()
    X_test, y_test = data_loader.get_test_data()

    print("Train set size:", X_train.shape[0])
    print("Validation set size:", X_val.shape[0])
    print("Test set size:", X_test.shape[0])

if __name__ == "__main__":
    main()