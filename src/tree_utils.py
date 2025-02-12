import numba as nb
from utils import norm_pdf, norm_cdf, logistic_pdf, logistic_cdf, extreme_pdf, extreme_cdf

@nb.jit(nopython=True)
def get_best_split(self, X, y):
    best_split = None
    best_feature = None
    best_left_indices = None
    best_right_indices = None
    best_loss = np.inf

    n_samples, n_features = len(X), len(X[0])

    for feature in range(n_features):
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

@nb.jit(nopython=True)
def calculate_loss(self, y, pred=None):
    if pred is None:
        pred = self.mean_y(y)

    loss = 0
    for i in range(len(y)):
        censor, value = y[i]
        if censor:
            loss += self.get_censored_value(value, np.inf, pred)
        else:
            loss += self.get_uncensored_value(value, pred)
    return loss

@nb.jit(nopython=True)
def get_uncensored_value(self, y, pred):
    if self.function == "norm":
        pdf = norm_pdf(y, pred, self.sigma)
    elif self.function == "logistic":
        pdf = logistic_pdf(y, pred, self.sigma)
    else:
        pdf = extreme_pdf(y, pred, self.sigma)

    if pdf <= 0:
        pdf = self.epsilon
    return -np.log(pdf/(self.sigma*y))

@nb.jit(nopython=True)
def get_censored_value(self, y_lower, y_upper, pred):
    if self.function == "norm":
        cdf_diff = norm_cdf(y_upper, pred, self.sigma) - norm_cdf(y_lower, pred, self.sigma)
    elif self.function == "logistic":
        cdf_diff = logistic_cdf(y_upper, pred, self.sigma) - logistic_cdf(y_lower, pred, self.sigma)
    else:
        cdf_diff = extreme_cdf(y_upper, pred, self.sigma) - extreme_cdf(y_lower, pred, self.sigma)

    if cdf_diff <= 0:
        cdf_diff = self.epsilon
    return -np.log(cdf_diff)

@nb.jit(nopython=True)
def mean_y(self, y):
    return np.mean([value for _, value in y])