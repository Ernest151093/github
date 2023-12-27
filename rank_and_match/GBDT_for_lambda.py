class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.22, ndcg_top_k: int = 10,
                 subsample: float = 0.612, colsample_bytree: float = 0.904,
                 max_depth: int = 7, min_samples_leaf: int = 15):
        self._prepare_data()
        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self.trees = []
        self.tree_idx = []

        self.num_input_feats = self.X_train.shape[1]
        self.num_input_obj_train = self.X_train.shape[0]
        self.num_input_obj_test = self.X_test.shape[0]

        self.col_feats = int(self.num_input_feats * colsample_bytree)
        self.objects = int(self.num_input_obj_train * subsample)

        self.best_ndcg = -1
        self.best_tree_idx = -1

        # допишите ваш код здесь

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
        # допишите ваш код здесь
        X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)

        y_train = y_train.reshape(len(y_train), -1)
        y_test = y_test.reshape(len(y_test), -1)

        self.X_train = torch.Tensor(X_train)
        self.ys_train = torch.Tensor(y_train).reshape(-1, 1)

        self.X_test = torch.Tensor(X_test)
        self.ys_test = torch.Tensor(y_test).reshape(-1, 1)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        # допишите ваш код здесь 
        for cur_id in np.unique(inp_query_ids):
            mask = inp_query_ids == cur_id
            tmp_array = inp_feat_array[mask]
            scaler = StandardScaler()
            inp_feat_array[mask] = scaler.fit_transform(tmp_array)

        return inp_feat_array

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        # допишите ваш код здесь

        lambdas = torch.zeros(self.num_input_obj_train, 1)

        for cur_id in np.unique(self.query_ids_train):
            mask = self.query_ids_train == cur_id
            lambdas[mask] = self._compute_lambdas(self.ys_train[mask], train_preds[mask])

        tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                     random_state=cur_tree_idx)

        this_tree_feats = np.random.choice(
            list(range(self.num_input_feats)), self.col_feats, replace=False)
        this_tree_objs = np.random.choice(
            list(range(self.num_input_obj_train)), self.objects, replace=False)

        tree.fit(
            self.X_train[this_tree_objs.reshape(-1)
            ][:, this_tree_feats].numpy(),
            lambdas[this_tree_objs.reshape(-1), :].numpy()
        )

        return tree, this_tree_feats

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        # допишите ваш код здесь
        calc_ndcg = []
        for cur_id in np.unique(queries_list):
            mask = queries_list == cur_id
            target_true = true_labels[mask]
            target_pred = preds[mask]
            calc_ndcg.append(self._ndcg_k(target_true, target_pred, self.ndcg_top_k))

        return np.mean(calc_ndcg)

    def fit(self):
        np.random.seed(0)
        # допишите ваш код здесь
        train_preds = torch.zeros(self.num_input_obj_train, 1)
        test_preds = torch.zeros(self.num_input_obj_test, 1)
        train_ndcgs, test_ndcgs = [], []
        pbar = tqdm(range(self.n_estimators))

        for cur_tree_idx in pbar:
            tree, idx_col = self._train_one_tree(cur_tree_idx, train_preds)

            self.trees.append(tree)
            self.tree_idx.append(idx_col)
            cur_train_data = self.X_train[:, idx_col].numpy()
            train_preds -= self.lr * torch.Tensor(tree.predict(cur_train_data)).reshape(-1, 1)

            train_ndcg = self._calc_data_ndcg(
                self.query_ids_train, self.ys_train, train_preds)
            print(f"Tree: {cur_tree_idx}; train_NDCG: {train_ndcg}", end=' ')

            cur_test_data = self.X_test[:, idx_col]
            test_preds -= self.lr * torch.Tensor(tree.predict(cur_test_data)).reshape(-1, 1)

            test_ndcg = self._calc_data_ndcg(
                self.query_ids_test, self.ys_test, test_preds)
            print(f"test_NDCG: {test_ndcg}")

            if self.best_ndcg < test_ndcg:
                self.best_ndcg = test_ndcg
                self.best_tree_idx = cur_tree_idx

            train_ndcgs.append(train_ndcg)
            test_ndcgs.append(test_ndcg)
            pbar.set_description_str(
                f'Test nDCG@{self.ndcg_top_k}={round(test_ndcg, 5)}')

        cut_idx = self.best_tree_idx + 1
        self.trees = self.trees[:cut_idx]
        self.tree_idx = self.tree_idx[:cut_idx]

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        # допишите ваш код здесь
        preds = torch.zeros(data.shape[0], 1)
        for cur_tree_idx in range(len(self.trees)):
            tree = self.trees[cur_tree_idx]
            feat_idx = self.tree_idx[cur_tree_idx]
            tmp_preds = tree.predict(data[:, feat_idx].numpy())
            preds += self.lr * torch.FloatTensor(tmp_preds).reshape(-1, 1)

        return preds

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:

        def compute_ideal_dcg(ys_true):
            idealdcg = 0
            ys_true = torch.squeeze(ys_true)
            ys_true_desc_ideal, indices = ys_true.sort(descending=True)
            for i in range(len(ys_true)):
                idealdcg += float((2 ** ys_true_desc_ideal[i] - 1) / math.log2(i + 2))

            return idealdcg

        def compute_labels_in_batch(y_true):
            # разница релевантностей каждого с каждым объектом
            rel_diff = y_true - y_true.t()
            # 1 в этой матрице - объект более релевантен
            pos_pairs = (rel_diff > 0).type(torch.float32)
            # 1 тут - объект менее релевантен
            neg_pairs = (rel_diff < 0).type(torch.float32)
            Sij = pos_pairs - neg_pairs

            return Sij

        def compute_gain_diff(y_true, gain_scheme):
            y_true = y_true.reshape(-1, 1)
            if gain_scheme == "exp2":
                gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
            elif gain_scheme == "diff":
                gain_diff = y_true - y_true.t()
            else:
                raise ValueError(f"{gain_scheme} method not supported")
            return gain_diff

        # находим идеальный dcg
        ideal_dcg = compute_ideal_dcg(y_true)

        if ideal_dcg == 0:
            N = 0
        else:
            N = 1 / ideal_dcg

        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        _, rank_order = torch.sort(y_true, descending=True, axis=0)
        rank_order += 1

        with torch.no_grad():
            # получаем все попарные разницы скоров в батче
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

            # поставим разметку для пар, 1 если первый документ релевантнее
            # -1 если второй документ релевантнее
            Sij = compute_labels_in_batch(y_true)
            # посчитаем изменение gain из-за перестановок
            gain_diff = compute_gain_diff(y_true, "exp2")

            # посчитаем изменение знаменателей-дискаунтеров
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            # посчитаем непосредственное изменение nDCG
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            # посчитаем лямбды
            lambda_update = (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

        return lambda_update

    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k) -> float:
        # допишите ваш код здесь
        dcg = 0
        idealdcg = 0
        # print(ys_pred.shape)
        ys_pred = torch.transpose(ys_pred, 1, 0)
        ys_pred = torch.squeeze(ys_pred)
        ys_true = torch.squeeze(ys_true)

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
        # print("dcg: ", dcg)
        # print("idealdcg: ", idealdcg)
        if idealdcg == 0:
            return 0
        else:
            return float(dcg / idealdcg)

    def save_model(self, path: str):
        state = {
            'trees': self.trees,
            'trees_feat_idxs': self.trees_feat_idxs,
            'best_ndcg': self.best_ndcg,
            'lr': self.lr
        }
        f = open(path, 'wb')
        pickle.dump(state, f)

    def load_model(self, path: str):
        f = open(path, 'rb')
        state = pickle.load(f)
        self.trees = state['trees']
        self.trees_feat_idxs = state['trees_feat_idxs']
        self.best_ndcg = state['best_ndcg']
        self.lr = state['lr']