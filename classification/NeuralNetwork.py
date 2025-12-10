import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


class NeuralNetwork(BaseEstimator, ClassifierMixin):
    """
    A PyTorch-based neural network classifier for binary classification.

    This class implements a neural network with dropout regularization that is compatible
    with scikit-learn's pipeline and GridSearchCV. It uses BCE with logits loss and
    includes class balancing for imbalanced datasets. The model architecture consists
    of two hidden layers with ReLU activation and dropout, followed by a sigmoid output.
    """

    def __init__(self,
                 batch_size=10,
                 epochs=50,
                 dropout=0.2,
                 lr=0.001,
                 weight_decay=0.001):
        """
        Initialize the NeuralNetwork classifier.

        Args:
            batch_size (int, optional): Number of samples per batch. Defaults to 10.
            epochs (int, optional): Number of training epochs. Defaults to 50.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.2.
            lr (float, optional): Learning rate for Adam optimizer. Defaults to 0.001.
            weight_decay (float, optional): L2 regularization strength. Defaults to 0.001.
        """

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        This method is required for scikit-learn compatibility with GridSearchCV.

        Args:
            deep (bool, optional): If True, returns parameters for this estimator
                and contained subobjects. Defaults to True.

        Returns:
            dict: Dictionary of parameter names and values.
        """
        return {
            'dropout': self.dropout,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        This method is required for scikit-learn compatibility with GridSearchCV.

        Args:
            **params: Estimator parameters to set.

        Returns:
            NeuralNetwork: Returns self for method chaining.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def build_model(self, input_dim):
        """
        Build the neural network architecture.

        Creates a sequential neural network with two hidden layers (10 and 4 units),
        ReLU activation functions, dropout regularization, and a single output unit
        for binary classification.

        Args:
            input_dim (int): The number of input features.

        Returns:
            nn.Sequential: The built neural network model.
        """
        return nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(10, 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(4, 1)
        )

    def fit(self, X, y):
        """
        Train the neural network on the provided data.

        Performs the following steps:
        1. Converts input data to float32 numpy arrays
        2. Splits data into 80% training and 20% validation sets
        3. Builds the model with the correct input dimension
        4. Uses BCE with logits loss with class weights for balance
        5. Trains for the specified number of epochs with early stopping based on validation loss
        6. Uses learning rate scheduling to reduce LR on validation loss plateau
        7. Restores the best model weights after training

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Binary labels of shape (n_samples,).

        Returns:
            NeuralNetwork: Returns self for method chaining.
        """
        torch.manual_seed(42)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        # determine input dimension from data
        input_dim = X.shape[1]

        # internal validation split (from training data only)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # build model with correct input dimension
        self.model = self.build_model(input_dim)

        # class-balanced BCE with logits for numerical stability
        # Calculate pos_weight: ratio of negative samples to positive samples
        y_train_flat = y_train.flatten()
        n_neg = (y_train_flat == 0).sum()
        n_pos = (y_train_flat == 1).sum()
        pos_weight = torch.tensor(
            [n_neg / n_pos if n_pos > 0 else 1.0], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=0.3, patience=5)

        # DataLoaders
        train_ds = TensorDataset(torch.from_numpy(X_train),
                                 torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val),
                               torch.from_numpy(y_val))

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True)
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False)

        best_val_loss = float('inf')
        self.best_state = None

        for _ in range(self.epochs):
            self.model.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_X.size(0)

            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    logits = self.model(batch_X)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * batch_X.size(0)

                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    correct += (preds == batch_y).sum().item()
                    total += batch_y.size(0)

            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)

            # save best weights
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_state = {k: v.clone()
                                   for k, v in self.model.state_dict().items()}

        # load best weights into model
        self.model.load_state_dict(self.best_state)
        return self

    def predict(self, X):
        """
        Make binary predictions on the provided data.

        Converts the raw model logits to probabilities using sigmoid activation,
        then applies a 0.5 threshold to produce binary predictions.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Binary predictions (0 or 1) of shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(torch.from_numpy(X))
            probs = torch.sigmoid(logits).numpy().reshape(-1)
        return (probs >= 0.5).astype(int)
