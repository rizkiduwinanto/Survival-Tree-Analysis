import numpy as np
import cupy as cp
from node import TreeNode
from distribution import Weibull, LogLogistic, LogNormal, LogExtreme, GMM, GMM_New
from math_utils_gpu import norm_pdf, norm_cdf, logistic_pdf, logistic_cdf, extreme_pdf, extreme_cdf
from tree_utils import gpu_train_test_split
from lifelines.utils import concordance_index
import graphviz
import uuid
from concurrent.futures import ThreadPoolExecutor
import json
from sklearn.model_selection import train_test_split
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
from sklearn.metrics import mean_absolute_error
from collections import deque
from cupy.cuda import Stream

class AFTSurvivalTree():
    """
        Regression tree that implements AFTLoss
    """
    def __init__(
        self, 
        max_depth=5, 
        min_samples_split=5, 
        min_samples_leaf=5,
        sigma=0.5, 
        function="norm", 
        is_custom_dist=False,
        is_bootstrap=False,
        n_components=10,
        n_samples=1000,
        percent_len_sample=0.8,
        test_size=0.2,
        mode="bfs"
    ):
        self.tree = None
        self.max_depth = (2**31) - 1 if max_depth is None else max_depth

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.sigma = sigma 
        self.epsilon = 10e-12
        self.function = function.lower()
        self.custom_dist = None
        self.is_bootstrap = is_bootstrap
        self.is_custom_dist = is_custom_dist
        self.n_samples = n_samples
        self.percent_len_sample = percent_len_sample
        self.test_size = test_size
        self.mode = mode

        if is_custom_dist:
            if function == "weibull":
                self.custom_dist = Weibull()
            elif function == "logistic":
                self.custom_dist = LogLogistic()
            elif function == "norm":
                self.custom_dist = LogNormal()
            elif function == "extreme":
                self.custom_dist = LogExtreme()
            elif function == "gmm":
                self.custom_dist = GMM(n_components=n_components)
            elif function == "gmm_new":
                self.custom_dist = GMM_New(n_components=n_components)
            else:
                raise ValueError("Custom distribution not supported")

    def fit(self, X, y, random_state=None):
        if self.custom_dist is not None:
            if self.is_bootstrap:
                self.custom_dist.fit_bootstrap(y, n_samples=self.n_samples, percentage=self.percent_len_sample)

                X_gpu = cp.asarray(X)
                y_death_gpu = cp.asarray(y['death'])
                y_time_gpu = cp.asarray(y['d.time'])

            else:
                X_gpu = cp.asarray(X)
                y_death_gpu = cp.asarray(y['death'])
                y_time_gpu = cp.asarray(y['d.time'])

                X_train, y_death_train, y_time_train, X_test, y_death_test, y_time_test = gpu_train_test_split(X_gpu, y_death_gpu, y_time_gpu, test_size=self.test_size, random_state=random_state)

                y_death_test_cpu = cp.asnumpy(y_death_test)
                y_time_test_cpu = cp.asnumpy(y_time_test)

                y_dist = np.rec.fromarrays([y_death_test_cpu, y_time_test_cpu], names=['death', 'd.time'])

                self.custom_dist.fit(y_dist)

                X_gpu = X_train
                y_death_gpu = y_death_train
                y_time_gpu = y_time_train
        else:
            X_gpu = cp.asarray(X)
            y_death_gpu = cp.asarray(y['death'])
            y_time_gpu = cp.asarray(y['d.time'])

        if self.mode == "recursive":
            self.build_tree(X_gpu, y_death_gpu, y_time_gpu)
        elif self.mode == "bfs":
            self.build_tree_bfs(X_gpu, y_death_gpu, y_time_gpu)
        elif self.mode == "dfs":
            self.build_tree_dfs(X_gpu, y_death_gpu, y_time_gpu)
        return

    def build_tree(self, X, y_death, y_time, depth=0):   
        if depth > self.max_depth or len(y_time) < self.min_samples_split:
            node = TreeNode(None, None, self.mean_y(y_time), None, None, num_sample=len(y_time))
            if depth == 0: 
                self.tree = node
            return node
            
        split, feature, left_indices, right_indices, loss = self.get_best_split_vectorized(X, y_death, y_time)

        if split is None and feature is None:
            node = TreeNode(None, None, self.mean_y(y_time), None, None, num_sample=len(y_time))
            if depth == 0: 
                self.tree = node
            return node
        
        X_left = X[left_indices]
        y_death_left = y_death[left_indices]
        y_time_left = y_time[left_indices]
        X_right = X[right_indices]
        y_death_right = y_death[right_indices]
        y_time_right = y_time[right_indices]
        

        if len( y_time_left) == 0 or len(y_time_right) == 0:
            node = TreeNode(None, None, self.mean_y(y_time), None, None, num_sample=len(y_time))
            if depth == 0: 
                self.tree = node
            return node

        if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
            node = TreeNode(feature, None, self.mean_y(y_time), None, None, num_sample=len(y_time))
            if depth == 0:
                self.tree = node
            return node

        left = self.build_tree(X_left, y_death_left, y_time_left, depth+1)
        right = self.build_tree(X_right, y_death_right, y_time_right, depth+1)
    
        node = TreeNode(feature, split, None, left, right, loss=loss, num_sample=len(y_time))
        if depth == 0:
            self.tree = node
        return node

    def build_tree_bfs(self, X, y_death, y_time, depth=0):     
        queue = deque()
        root = TreeNode(None, None, None, None, None, num_sample=len(y_time))
        self.tree = root

        queue.append({
            'parent_node':root,
            'X': X,
            'y_death': y_death,
            'y_time': y_time,
            'depth': depth
        })

        while queue:
            current_node = queue.popleft()
            X_current = current_node['X']
            y_death_current = current_node['y_death']
            y_time_current = current_node['y_time']
            depth = current_node['depth']
            parent_node = current_node['parent_node']

            if depth > self.max_depth or len(y_time_current) < self.min_samples_split:
                parent_node.set_value(self.mean_y(y_time_current))
                continue
            
            split, feature, left_indices, right_indices, loss = self.get_best_split_vectorized_streams(X_current, y_death_current, y_time_current)

            if split is None and feature is None:
                parent_node.set_value(self.mean_y(y_time_current))
                continue

            X_left = X[left_indices]
            y_death_left = y_death[left_indices]
            y_time_left = y_time[left_indices]
            X_right = X[right_indices]
            y_death_right = y_death[right_indices]
            y_time_right = y_time[right_indices]

            if len(X_left) == 0 or len(X_right) == 0:
                parent_node.set_value(self.mean_y(y_time_current))
                continue

            if (len(y_time_left) < self.min_samples_leaf) or (len(y_time_right) < self.min_samples_leaf):
                parent_node.set_value(self.mean_y(y_time_current))
            else:
                parent_node.set_feature_index(feature)
                parent_node.set_threshold(split)
                parent_node.set_loss(loss)
                parent_node.set_num_sample(len(y_time_current))

                parent_node.set_left(TreeNode(None, None, None, None, None, num_sample=len(y_time_left)))
                parent_node.set_right(TreeNode(None, None, None, None, None, num_sample=len(y_time_right)))

                queue.append({
                    'parent_node':parent_node.left,
                    'X': X_left,
                    'y_death': y_death_left,
                    'y_time': y_time_left,
                    'depth': depth + 1
                })

                queue.append({
                    'parent_node':parent_node.right,
                    'X': X_right,
                    'y_death': y_death_right,
                    'y_time': y_time_right,
                    'depth': depth + 1
                })

        return self.tree  

    def build_tree_dfs(self, X, y_death, y_time, depth=0):     
        stack = []
        root = TreeNode(None, None, None, None, None, num_sample=len(y_time))
        self.tree = root

        stack.append({
            'parent_node':root,
            'X': X,
            'y_death': y_death,
            'y_time': y_time,
            'depth': depth
        })

        while stack:
            current_node = stack.pop()
            X_current = current_node['X']
            y_death_current = current_node['y_death']
            y_time_current = current_node['y_time']
            depth = current_node['depth']
            parent_node = current_node['parent_node']

            if depth > self.max_depth or len(y_time_current) < self.min_samples_split:
                parent_node.set_value(self.mean_y(y_time_current))
                continue
            
            split, feature, left_indices, right_indices, loss = self.get_best_split_vectorized(X_current, y_death_current, y_time_current)

            if split is None and feature is None:
                parent_node.set_value(self.mean_y(y_time_current))
                continue

            X_left = X[left_indices]
            y_death_left = y_death[left_indices]
            y_time_left = y_time[left_indices]
            X_right = X[right_indices]
            y_death_right = y_death[right_indices]
            y_time_right = y_time[right_indices]

            if (len(y_time_left) < self.min_samples_leaf) or (len(y_time_right) < self.min_samples_leaf):
                parent_node.set_value(self.mean_y(y_time_current))
            else:
                parent_node.set_feature_index(feature)
                parent_node.set_threshold(split)
                parent_node.set_loss(loss)
                parent_node.set_num_sample(len(y_time_current))

                parent_node.left = TreeNode(None, None, None, None, None, num_sample=len(y_time_left))
                parent_node.right = TreeNode(None, None, None, None, None, num_sample=len(y_time_right))

                stack.append({
                    'parent_node':parent_node.left,
                    'X': X_left,
                    'y_death': y_death_left,
                    'y_time': y_time_left,
                    'depth': depth + 1
                })

                stack.append({
                    'parent_node':parent_node.right,
                    'X': X_right,
                    'y_death': y_death_right,
                    'y_time': y_time_right,
                    'depth': depth + 1
                })

        return self.tree

    def get_best_split(self, X, y, feature):
        best_split = None
        best_feature = None
        best_left_indices = None
        best_right_indices = None
        best_loss = np.inf

        n_samples = len(X)

        sorted_indices = np.argsort(X[:, feature])
        sorted_X = X[sorted_indices, feature]
        sorted_y = y[sorted_indices]

        for i in range(n_samples - 1):
            if sorted_X[i] != sorted_X[i+1]:
                split = (sorted_X[i] + sorted_X[i+1]) / 2

                left_y = sorted_y[:i]
                right_y = sorted_y[i:]

                mean_y = self.mean_y(y)

                left_loss = self.calculate_loss(left_y, mean_y)
                right_loss = self.calculate_loss(right_y, mean_y)

                total_loss = left_loss + right_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_split = split
                    best_feature = feature
                    best_left_indices = sorted_indices[:i]
                    best_right_indices = sorted_indices[i:]

        return best_split, best_feature, best_left_indices, best_right_indices, best_loss

    def get_best_feature(self, X, y):
        best_split = None
        best_feature = None
        best_left_indices = None
        best_right_indices = None
        best_loss = np.inf

        n_features = len(X[0])

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_best_split, X, y, feature) for feature in range(n_features)]
            
            for future in futures:
                split, feature, left_indices, right_indices, loss = future.result()

                if loss < best_loss:
                    best_loss = loss
                    best_split = split
                    best_feature = feature
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        return best_split, best_feature, best_left_indices, best_right_indices, best_loss

    def get_best_split_vectorized_streams(self, X, y_death, y_time):
        best_split = None
        best_feature = None
        best_left_indices = None
        best_right_indices = None
        best_loss = cp.inf

        n_samples = len(X)
        n_features = len(X[0])

        mean_y = self.mean_y(y_time)

        n_streams = n_features
        streams = [cp.cuda.Stream() for _ in range(n_streams)]

        results = [{
            'best_loss': cp.inf,
            'best_split': None,
            'best_feature': None,
            'best_left_indices': None,
            'best_right_indices': None
        } for _ in range(n_streams)]

        def process_feature_stream(self, feature_idx, stream_idx):
            stream = streams[stream_idx]
            with stream:
                feature_values = X[:, feature_idx]
                unique_values = cp.unique(feature_values)
                
                if len(unique_values) <= 1:
                    return 

                thresholds = unique_values.reshape(-1, 1)
                left_mask = feature_values < thresholds
                right_mask = feature_values >= thresholds

                valid_splits = cp.any(left_mask, axis=1) & cp.any(right_mask, axis=1)
                thresholds = thresholds[valid_splits]
                left_mask = left_mask[valid_splits]
                right_mask = right_mask[valid_splits]

                if len(thresholds) == 0:
                    return

                left_loss = cp.array([self.calculate_loss_vectorized(y_death[mask], y_time[mask], pred=mean_y) for mask in left_mask])
                right_loss = cp.array([self.calculate_loss_vectorized(y_death[mask], y_time[mask], pred=mean_y) for mask in right_mask])

                total_loss = left_loss + right_loss

                best_idx = cp.argmin(total_loss).item()
                current_min_loss = total_loss[best_idx].item()

                if current_min_loss < results[stream_idx]['best_loss']:
                    results[stream_idx]['best_loss'] = current_min_loss
                    results[stream_idx]['best_split'] = thresholds[best_idx].item()
                    results[stream_idx]['best_feature'] = feature_idx
                    results[stream_idx]['best_left_indices'] = cp.where(left_mask[best_idx])[0]
                    results[stream_idx]['best_right_indices'] = cp.where(right_mask[best_idx])[0] 

        for feature_idx in range(n_features):
            process_feature_stream(self, feature_idx, feature_idx % n_streams)

        for stream in streams:
            stream.synchronize()

        for result in results:
            if result['best_loss'] < best_loss:
                best_loss = result['best_loss']
                best_split = result['best_split']
                best_feature = result['best_feature']
                best_left_indices = result['best_left_indices']
                best_right_indices = result['best_right_indices']

        return best_split, best_feature, best_left_indices, best_right_indices, best_loss

    def get_best_split_vectorized(self, X, y_death, y_time):
        best_split = None
        best_feature = None
        best_left_indices = None
        best_right_indices = None
        best_loss = cp.inf

        n_samples = len(X)
        n_features = len(X[0])

        mean_y = self.mean_y(y_time)

        for feature in range(n_features):
            feature_values = X[:, feature]
            unique_values = cp.unique(feature_values)
            
            if len(unique_values) <= 1:
                continue 

            thresholds = unique_values.reshape(-1, 1)
            left_mask = feature_values < thresholds
            right_mask = feature_values >= thresholds

            valid_splits = cp.any(left_mask, axis=1) & cp.any(right_mask, axis=1)
            thresholds = thresholds[valid_splits]
            left_mask = left_mask[valid_splits]
            right_mask = right_mask[valid_splits]

            if len(thresholds) == 0:
                continue

            left_loss = cp.array([self.calculate_loss_vectorized(y_death[mask], y_time[mask], pred=mean_y) for mask in left_mask])
            right_loss = cp.array([self.calculate_loss_vectorized(y_death[mask], y_time[mask], pred=mean_y) for mask in right_mask])

            total_loss = left_loss + right_loss

            best_idx = cp.argmin(total_loss).item()
            current_min_loss = total_loss[best_idx].item()

            if current_min_loss < best_loss:
                best_loss = current_min_loss
                best_split = thresholds[best_idx].item()
                best_feature = feature
                best_left_indices = cp.where(left_mask[best_idx])[0]
                best_right_indices = cp.where(right_mask[best_idx])[0]

        return best_split, best_feature, best_left_indices, best_right_indices, best_loss

    def broadcast_loss(self, y_death, y_time, pred_mask, pred=None):
        loss = self.calculate_loss(y_death, y_time, pred=pred)
        loss_broadcast = cp.broadcast_to(loss, pred_mask.shape)
        mask_loss = cp.where(pred_mask, loss_broadcast, 0)
        return cp.sum(mask_loss, axis=1)
            
    def calculate_loss(self, y_death, y_time, pred=None):
        if pred is None:
            pred = self.mean_y(y_time)

        loss = 0
        for i in range(len(y_time)):
            if y_death[i]:
                loss += self.get_censored_value(y_time[i], cp.inf, pred)
            else:
                loss += self.get_uncensored_value(y_time[i], pred)
        return loss

    def calculate_loss_vectorized(self, y_death, y_time, pred=None):
        if pred is None:
            pred = self.mean_y(y_time)

        is_censored = y_death.astype(bool)

        uncensored_loss = self.get_uncensored_value(y_time, pred)
        censored_loss = self.get_censored_value(y_time, cp.inf, pred)

        loss = cp.where(is_censored, uncensored_loss, censored_loss)

        return cp.sum(loss)
    
    def get_uncensored_value(self, y, pred):
        if self.custom_dist is not None:
            link_function = cp.log(y) - pred
            pdf = self.custom_dist.pdf_gpu(link_function)
        else:
            if self.function == "norm":
                pdf = norm_pdf(y, pred, self.sigma)
            elif self.function == "logistic":
                pdf = logistic_pdf(y, pred, self.sigma)
            elif self.function == "extreme":
                pdf = extreme_pdf(y, pred, self.sigma)
            else:
                raise ValueError("Distribution not supported")

        pdf = cp.maximum(pdf, self.epsilon)
        return -cp.log(pdf/(self.sigma*y))

    def get_censored_value(self, y_lower, y_upper, pred):
        if self.custom_dist is not None:
            link_function_lower = cp.log(y_lower) - pred
            link_function_upper = cp.log(y_upper) - pred
            cdf_diff = self.custom_dist.cdf_gpu(link_function_upper) - self.custom_dist.cdf_gpu(link_function_lower)
        else:
            if self.function == "norm":
                cdf_diff = norm_cdf(y_upper, pred, self.sigma) - norm_cdf(y_lower, pred, self.sigma)
            elif self.function == "logistic":
                cdf_diff = logistic_cdf(y_upper, pred, self.sigma) - logistic_cdf(y_lower, pred, self.sigma)
            elif self.function == "extreme":
                cdf_diff = extreme_cdf(y_upper, pred, self.sigma) - extreme_cdf(y_lower, pred, self.sigma)
            else:
                raise ValueError("Distribution not supported")

        cdf_diff = cp.maximum(cdf_diff, self.epsilon)
        return -np.log(cdf_diff)

    def mean_y(self, y_time):
        return cp.mean(y_time)

    def predict(self, X):
        if self.tree is None:
            raise ValueError("Tree has not been built. Call `fit` first.")
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            X = X.reshape(1, -1)
        predictions = [self.get_prediction(x, self.tree) for x in X]
        return predictions

    def get_prediction(self, X, tree):
        try:
            if tree.value is not None:
                return tree.value.get()
            else:
                feature_value = X[tree.feature_index]
                if feature_value <= tree.threshold:
                    return self.get_prediction(X, tree.left)
                else:
                    return self.get_prediction(X, tree.right)
        except:
            raise ValueError("Error in get_prediction")

    def _print(self):
        if self.tree is None:
            raise ValueError("Tree has not been built. Call `fit` first.")
        else:
            self.print_tree(self.tree)

    def print_tree(self, tree, indent=" "):
        if tree is None:
            print(f"{indent}None")
            return

        if tree.value is not None:
            print(f"{indent}value: {tree.value}")
        else:
            print(f"{indent}X_{tree.feature_index} <= {tree.threshold}")
            print(f"{indent}left:", end="")
            self.print_tree(tree.left, indent + "  ")
            print(f"{indent}right:", end="")
            self.print_tree(tree.right, indent + "  ")

    def _score(self, X, y_true):
        """
            Implement C-Index
        """
        times_pred = self.predict(X)
        event_true = [1 if not censored else 0 for censored, _ in y_true]
        times_true = [time for _, time in y_true]

        c_index = concordance_index(times_true, times_pred, event_true)
        return c_index

    def _brier(self, X, y):
        """
            Compute the Integrated Brier Score (IBS).
        """
        pred_times = self.predict(X)

        y_structured = np.array([(bool(not censor), float(time)) for censor, time in y], dtype=[('event', bool), ('time', float)])

        times_true = [time for _, time in y]
        min_time = min(times_true) 
        max_time = max(times_true)
        time_points = np.linspace(min_time, max_time * 0.999, 100)

        survival_probs = np.array([[1.0 if t < pred_time else 0.0 for t in time_points] 
                              for pred_time in pred_times])

        ibs = integrated_brier_score(y_structured, y_structured, survival_probs, time_points)
        return ibs

    def _auc(self, X, y):
        """
            Compute the Area Under the Curve (AUC).
        """
        pred_times = self.predict(X)

        y_structured = np.array([(bool(not censor), float(time)) for censor, time in y], dtype=[('event', bool), ('time', float)])

        times_true = [time for _, time in y]
        min_time = min(times_true) 
        max_time = max(times_true)
        time_points = np.linspace(min_time, max_time * 0.999, 100)

        survival_probs = np.array([[1.0 if t < pred_time else 0.0 for t in time_points] 
                              for pred_time in pred_times])

        auc, mean_auc = cumulative_dynamic_auc(y_structured, y_structured, survival_probs, time_points)
        return auc, mean_auc

    def _mae(self, X, y):
        """
            Compute the Mean Absolute Error (MAE).
        """
        pred_times = self.predict(X)

        event_true = [1 if not censored else 0 for censored, _ in y]
        times_true = [time for _, time in y]

        mae = mean_absolute_error(times_true, pred_times)
        return mae

    def _visualize(self):
        if self.tree is None:
            raise ValueError("Tree has not been built. Call `fit` first.")
        else:
            dot = graphviz.Digraph(comment='AFT Survival Tree')
            dot = self.visualize(dot, self.tree)
            dot.render('doctest-output/decision_tree').replace('\\', '/')

    def visualize(self, dot, tree, node_id=None):
        if node_id is None:
            node_id = str(uuid.uuid4())

        if tree.value is not None:
            dot.node(node_id, f"value: {np.round(tree.value, 3)} \n num_sample = {tree.num_sample}", shape='rectangle')
        else:
            dot.node(node_id, f"X_{tree.feature_index} <= {np.round(tree.threshold, 2)} \n loss = {np.round(tree.loss, 2)} \n num_sample = {tree.num_sample}", shape='rectangle')
            if tree.left is not None:
                node_left = str(uuid.uuid4())
                dot = self.visualize(dot, tree.left, node_left)
                dot.edge(node_id, node_left)
            if tree.right is not None:
                node_right = str(uuid.uuid4())
                dot = self.visualize(dot, tree.right, node_right)
                dot.edge(node_id, node_right)
            
        return dot

    def save(self, path):
        model_state = {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'sigma': float(self.sigma),
            'function': self.function,
            'epsilon': float(self.epsilon),
            'is_custom_dist': self.is_custom_dist,
            'is_bootstrap': self.is_bootstrap,
            'tree': self.tree.to_dict() if self.tree is not None else None
        }
        
        if self.custom_dist is not None:
            model_state['custom_dist_type'] = self.custom_dist.__class__.__name__
            model_state['custom_dist_params'] = self.custom_dist.get_params()
        
        with open(path, 'w') as f:
            json.dump(model_state, f, indent=4)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            model_state = json.load(f)
        
        model = cls(
            max_depth=model_state['max_depth'],
            min_samples_split=model_state['min_samples_split'],
            min_samples_leaf=model_state['min_samples_leaf'],
            sigma=model_state['sigma'],
            function=model_state['function'],
            is_custom_dist=model_state['is_custom_dist'],
            is_bootstrap=model_state['is_bootstrap']
        )
        
        model.epsilon = model_state['epsilon']
        
        if 'custom_dist_type' in model_state:
            dist_type = model_state['custom_dist_type']
            dist_params = model_state['custom_dist_params']
            
            if dist_type == 'Weibull':
                model.custom_dist = Weibull()
            elif dist_type == 'LogLogistic':
                model.custom_dist = LogLogistic()
            elif dist_type == 'LogNormal':
                model.custom_dist = LogNormal()
            elif dist_type == 'LogExtreme':
                model.custom_dist = LogExtreme()
            elif dist_type == 'GMM':
                model.custom_dist = GMM()

            model.custom_dist.set_params(dist_params)
        
        if model_state['tree'] is not None:
            model.tree = TreeNode.from_dict(model_state['tree'])
        
        return model


