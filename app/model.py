# model.py

import torch.nn as nn

class LoanPredictorNN(nn.Module):
    def __init__(self, input_dim):
        super(LoanPredictorNN, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 256)
        self.relu_1 = nn.ReLU()
        self.batchnorm_1 = nn.BatchNorm1d(256)
        self.dropout_1 = nn.Dropout(0.5)

        self.layer_2 = nn.Linear(256, 128)
        self.relu_2 = nn.ReLU()
        self.batchnorm_2 = nn.BatchNorm1d(128)
        self.dropout_2 = nn.Dropout(0.4)

        self.layer_3 = nn.Linear(128, 64)
        self.relu_3 = nn.ReLU()
        self.batchnorm_3 = nn.BatchNorm1d(64)
        self.dropout_3 = nn.Dropout(0.3)

        self.output_layer = nn.Linear(64, 2)

    def forward(self, x):
        is_eval_batch_one = not self.training and x.shape[0] <= 1

        x = self.layer_1(x)
        x = self.relu_1(x) if is_eval_batch_one else self.relu_1(self.batchnorm_1(x))
        x = self.dropout_1(x)

        x = self.layer_2(x)
        x = self.relu_2(x) if is_eval_batch_one else self.relu_2(self.batchnorm_2(x))
        x = self.dropout_2(x)

        x = self.layer_3(x)
        x = self.relu_3(x) if is_eval_batch_one else self.relu_3(self.batchnorm_3(x))
        x = self.dropout_3(x)

        return self.output_layer(x)
