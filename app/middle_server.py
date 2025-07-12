# middle_server.py

import torch
import copy
from model import LoanPredictorNN

class MiddleServer:
    def __init__(self, server_id, input_dim):
        self.server_id = server_id
        self.clients = []
        self.has_clients = False
        self.model = LoanPredictorNN(input_dim)

    def assign_clients_for_round(self, clients):
        self.clients = [c for c in clients if c.has_data]
        self.has_clients = len(self.clients) > 0

    def aggregate_client_models(self, client_params_list):
        if not self.has_clients or not client_params_list:
            return None

        valid_params = [p for p in client_params_list if p is not None]
        if not valid_params:
            return None  # No valid models to aggregate from

        agg_params = copy.deepcopy(valid_params[0])
        for key in agg_params:
            tensors = [p[key].float().cpu() for p in valid_params]
            if tensors:
                agg_params[key] = torch.stack(tensors, dim=0).mean(dim=0)

        try:
            self.model.load_state_dict(agg_params)
            return agg_params
        except Exception as e:
            print(f"Error loading aggregated state into MiddleServer {self.server_id}: {e}")
            return None

    def distribute_model_to_clients(self, params):
        if not self.has_clients or params is None:
            return
        for client in self.clients:
            client.set_model_params(copy.deepcopy(params))

    def get_model_params(self):
        return copy.deepcopy(self.model.state_dict()) if self.has_clients else None

    def set_model_params(self, params):
        if params is not None:
            try:
                self.model.load_state_dict(params)
            except Exception as e:
                print(f"Error loading state into MiddleServer {self.server_id}: {e}")
