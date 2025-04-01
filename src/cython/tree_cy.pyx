from libcpp cimport bool
from distribution cimport Weibull

cdef class AFTSurvivalTree():
    cdef:
        int max_depth
        int min_samples_split
        int min_samples_leaf
        double sigma
        double epsilon
        bool is_bootstrap
        bool is_custom_dist
        object custom_dist
        char* function

    def __init__(
        self, 
        int max_depth=5, 
        int min_samples_split=5, 
        int min_samples_leaf=5,
        double sigma=0.5, 
        str function="norm", 
        bool is_custom_dist=False,
        bool is_bootstrap=False,
        int n_components=10
    ):
        self.tree = None
        self.max_depth = (2**31) - 1 if max_depth is None else max_depth

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.sigma = sigma 
        self.epsilon = 10e-12
        self.function = function
        self.custom_dist = None
        self.is_bootstrap = is_bootstrap
        self.is_custom_dist = is_custom_dist

        if is_custom_dist:
            if function == "weibull":
                self.custom_dist = Weibull()
            # elif function == "logistic":
            #     self.custom_dist = LogLogistic()
            # elif function == "norm":
            #     self.custom_dist = LogNormal()
            # elif function == "extreme":
            #     self.custom_dist = LogExtreme()
            # elif function == "gmm":
            #     self.custom_dist = GMM(n_components=n_components)
            else:
                raise ValueError("Custom distribution not supported")
