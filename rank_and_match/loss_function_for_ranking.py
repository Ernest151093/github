import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List

class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = torch.nn.Sequential(
                     torch.nn.Linear(num_input_features, self.hidden_dim),
                     torch.nn.ReLU(),
                     torch.nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
         X_test, y_test, self.query_ids_test) = self._get_data()

        X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)

        self.X_train = torch.Tensor(X_train)
        self.ys_train = torch.Tensor(y_train)

        self.X_test = torch.Tensor(X_test)
        self.ys_test = torch.Tensor(y_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        sc = StandardScaler()

        unique, indices = np.unique(inp_query_ids, return_index=True)

        for i in range(len(indices)):
            start = indices[i]
            if (i + 1) == len(indices):
                end = len(inp_feat_array)
            else:
                end = indices[i + 1]
            inp_feat_array[start:end] = sc.fit_transform(inp_feat_array[start:end])

        return inp_feat_array

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        net = ListNet(listnet_num_input_features, listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        mean_ndcg = []

        for i in range(self.n_epochs):
            print(f"Epoch: {i + 1}")
            self._train_one_epoch()
            mean_ndcg.append(self._eval_test_set())
        return mean_ndcg

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        batch_pred = torch.transpose(batch_pred, 1, 0)
        batch_pred = torch.squeeze(batch_pred)
        P_ys = torch.softmax(batch_ys, dim=0)
        P_yp = torch.softmax(batch_pred, dim=0)

        return -torch.sum(P_ys * torch.log(P_yp))

    def _train_one_epoch(self) -> None:
        self.model.train()

        uniq_number, indices = np.unique(self.query_ids_train, return_index=True)

        for i in range(len(indices)):
            start = indices[i]
            if (i + 1) == len(indices):
                end = len(self.X_train)
            else:
                end = indices[i + 1]
            self.optimizer.zero_grad()
            self.predict_t = self.model(self.X_train[start:end])
            loss_func = self._calc_loss(self.ys_train[start:end], self.predict_t)
            print(f"Loss function: {loss_func}")
            loss_func.backward(retain_graph=True)
            self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []

            uniq_number, indices = np.unique(self.query_ids_test, return_index=True)
            for i in range(len(indices)):
                start = indices[i]
                if (i + 1) == len(indices):
                    end = len(self.X_test)
                else:
                    end = indices[i + 1]
                self.predict_tt = self.model(self.X_test[start:end])
                ndcgs.append(self._ndcg_k(self.ys_test[start:end], self.predict_tt, self.ndcg_top_k))
            print(f"AVG ndcg: {np.mean(ndcgs)}")

            return np.mean(ndcgs)

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        dcg = 0
        idealdcg = 0
        ys_pred = torch.transpose(ys_pred, 1, 0)
        ys_pred = torch.squeeze(ys_pred)
        ys_true_desc = ys_true[ys_pred.sort(descending=True).indices]
        ys_true_desc_ideal, indices = ys_true.sort(descending=True)
        if len(ys_true) >= ndcg_top_k:
            for i in range(ndcg_top_k):
                dcg += float((2 ** ys_true_desc[i] - 1) / math.log2(i + 2))

            for i in range(ndcg_top_k):
                idealdcg += float((2 ** ys_true_desc_ideal[i] - 1) / math.log2(i + 2))

        else:
            for i in range(len(ys_true)):
                dcg += float((2 ** ys_true_desc[i] - 1) / math.log2(i + 2))

            for i in range(len(ys_true)):
                idealdcg += float((2 ** ys_true_desc_ideal[i] - 1) / math.log2(i + 2))

        return float(dcg / idealdcg)
