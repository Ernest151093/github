import numpy as np

class UpliftTreeRegressor:

    def __init__(self, max_depth=3, min_samples_leaf=1000, min_samples_leaf_treated=300, min_samples_leaf_control=300):
        self.max_depth = max_depth  # максимальная глубина дерева
        self.min_samples_leaf = min_samples_leaf  # минимально необходимое число обучающих объектов в листе дерева
        self.min_samples_leaf_treated = min_samples_leaf_treated  # минимально необходимое число обучающих объектов целевой группы
        self.min_samples_leaf_control = min_samples_leaf_control  # минимально необходимое число обучающих объектов контрольной группы

    def fit(self, X, treatment, y):
        self.n_features = X.shape[1]
        self.tree_ = self._grow_tree(X, y, treatment, depth=0)

    def _grow_tree(self, X, y, treatment, depth):
        num_t = treatment == 1  # маска для разделения целевой и контрольной групп
        uplift = np.sum(y[num_t]) / len(y[num_t]) - np.sum(y[~num_t]) / len(y[~num_t])  # разница средних ЦГ и КГ
        size_t_c = [X[num_t].shape[0], X[~num_t].shape[0]]

        node = Node(num_samples=y.size,
                    num_samples_per_group=size_t_c,
                    predicted_val=uplift
                    )

        if depth < self.max_depth:
            feat_idx, thr = self._best_split(X, y, treatment)
            if feat_idx is not None:
                node.feature_index = feat_idx
                node.threshold = thr
                indices_left = X[:, feat_idx] < thr
                X_left, y_left, treatment_left = X[indices_left], y[indices_left], treatment[indices_left]
                X_right, y_right, treatment_right = X[~indices_left], y[~indices_left], treatment[~indices_left]
                node.left = self._grow_tree(X_left, y_left, treatment_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, treatment_right, depth + 1)

        return node

    def _best_split(self, X, y, treatment):
        n_features = X.shape[1]
        m = y.size
        if m <= 1:
            return None, None

        if X.shape[0] < self.min_samples_leaf:
            return None, None

        num_t = treatment == 1
        if X[num_t].shape[0] < self.min_samples_leaf_treated or X[~num_t].shape[0] < self.min_samples_leaf_control:
            return None, None

        best_uplift = np.sum(y[num_t]) / len(y[num_t]) - np.sum(y[~num_t]) / len(y[~num_t])
        best_idx, best_thr = None, None
        best_delta_delta_p = 0

        for idx in range(n_features):
            thresholds, targets = zip(*sorted(zip(X[:, idx], y)))

            unique_values = np.unique(thresholds)
            if len(unique_values) > 10:
                percentiles = np.percentile(thresholds, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
            else:
                percentiles = np.percentile(thresholds, [10, 50, 90])
            threshold_options = np.unique(percentiles)

            for threshold in threshold_options:

                indices_left = X[:, idx] < threshold
                X_left, y_left, treatment_left = X[indices_left], y[indices_left], treatment[indices_left]
                X_right, y_right, treatment_right = X[~indices_left], y[~indices_left], treatment[~indices_left]
                if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
                    continue
                num_t_left = treatment_left == 1
                num_t_right = treatment_right == 1

                if len(X_left[num_t_left]) < self.min_samples_leaf_treated or len(
                        X_left[~num_t_left]) < self.min_samples_leaf_control or \
                        len(X_right[num_t_right]) < self.min_samples_leaf_treated or len(
                    X_right[~num_t_right]) < self.min_samples_leaf_control:
                    continue

                uplift_left = np.sum(y_left[num_t_left]) / len(y_left[num_t_left]) - np.sum(y_left[~num_t_left]) / len(
                    y_left[~num_t_left])
                uplift_right = np.sum(y_right[num_t_right]) / len(y_right[num_t_right]) - np.sum(
                    y_right[~num_t_right]) / len(y_right[~num_t_right])

                delta_delta_p = abs(uplift_left - uplift_right)

                if delta_delta_p > best_delta_delta_p:
                    best_delta_delta_p = delta_delta_p
                    best_idx = idx
                    best_thr = threshold

        return best_idx, best_thr

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        """Predict uplift for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_val
