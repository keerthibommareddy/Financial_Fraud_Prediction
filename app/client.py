# client.py

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import accuracy_score, f1_score
from model import LoanPredictorNN
from config import LOCAL_EPOCHS, LEARNING_RATE, BATCH_SIZE

class Client:
    def __init__(self, client_id, data, input_dim, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
        self.client_id = client_id
        self.batch_size = batch_size
        self.X, self.y = data
        self.has_data = len(self.X) > 0
        self.data_loader = None

        if self.has_data:
            dataset = torch.utils.data.TensorDataset(self.X, self.y)
            self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
            )

        self.model = LoanPredictorNN(input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs=LOCAL_EPOCHS):
        if not self.has_data or not self.data_loader:
            return 0.0

        self.model.train()
        epoch_loss = 0.0

        for epoch in range(epochs):
            current_epoch_loss = 0.0
            num_batches = 0

            for batch_X, batch_y in self.data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                current_epoch_loss += loss.item()
                num_batches += 1

            epoch_loss = current_epoch_loss / max(1, num_batches)

        return epoch_loss

    def get_model_params(self):
        return copy.deepcopy(self.model.state_dict()) if self.has_data else None

    def set_model_params(self, params):
        if params is not None:
            try:
                self.model.load_state_dict(params)
            except Exception as e:
                print(f"Error loading state for Client {self.client_id}: {e}")

    def evaluate_on_global_test(self, X_test_tensor, y_test_tensor, global_batch_size):
        if not self.has_data:
            return 0.0, 0.0

        self.model.eval()
        all_preds, all_labels = [], []

        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        is_bn = any(isinstance(m, nn.BatchNorm1d) for m in self.model.modules())
        drop_last = is_bn and global_batch_size > 1 and len(X_test_tensor) % global_batch_size == 1

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=global_batch_size, drop_last=drop_last
        )

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        return accuracy, f1
