# global_server.py

import torch
import torch.nn as nn
import copy
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from model import LoanPredictorNN
from config import BATCH_SIZE

class GlobalServer:
    def __init__(self, middle_servers, input_dim):
        self.middle_servers = middle_servers
        self.model = LoanPredictorNN(input_dim)
        print(f"Global Server initialized with {len(self.middle_servers)} middle servers.")

    def aggregate_middle_server_models(self, middle_params_list):
        if not self.middle_servers or not middle_params_list:
            return None

        valid_params = [p for p in middle_params_list if p is not None]
        if not valid_params:
            return None

        global_params = copy.deepcopy(valid_params[0])
        for key in global_params:
            tensors = [p[key].float().cpu() for p in valid_params]
            if tensors:
                global_params[key] = torch.stack(tensors, dim=0).mean(dim=0)

        try:
            self.model.load_state_dict(global_params)
            return global_params
        except Exception as e:
            print(f"Error loading aggregated state into Global Server: {e}")
            return None

    def distribute_model_to_middle_servers(self, params):
        if params is None:
            return
        for server in self.middle_servers:
            server.set_model_params(copy.deepcopy(params))

    def get_model_params(self):
        return copy.deepcopy(self.model.state_dict())

    def evaluate_global_model(self, X_test_tensor, y_test_tensor):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        all_preds, all_labels, all_probs_class1 = [], [], []
        test_loss = 0.0
        num_batches = 0

        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        test_batch_size = min(BATCH_SIZE * 4, len(X_test_tensor))
        is_bn = any(isinstance(m, nn.BatchNorm1d) for m in self.model.modules())
        drop_last = is_bn and test_batch_size > 1 and len(X_test_tensor) % test_batch_size == 1

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, drop_last=drop_last
        )

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs_class1.extend(probabilities[:, 1].cpu().numpy())
                num_batches += 1

        avg_loss = test_loss / max(1, num_batches)
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        accuracy = accuracy_score(all_labels_np, all_preds_np) * 100
        f1 = f1_score(all_labels_np, all_preds_np, average='binary', zero_division=0)

        auc = 0.0
        if len(np.unique(all_labels_np)) > 1:
            try:
                auc = roc_auc_score(all_labels_np, np.array(all_probs_class1))
            except Exception as e:
                print(f"  - AUC calculation error: {e}")

        return accuracy, avg_loss, f1, auc
