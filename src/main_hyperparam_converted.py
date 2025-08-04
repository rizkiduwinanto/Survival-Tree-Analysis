from utils.convert_hyperparam import convert_hyperparam

def run():
    idx = 0

    idx = convert_hyperparam(
        path_to_read="hyperparam/params_49.json",
        model="AFTForest",
        is_to_gmm=True,
        start_idx=idx,
        default_output_dir="converted_hyperparam"
    )

    idx = convert_hyperparam(
        path_to_read="hyperparam/params_54.json",
        model="AFTForest",
        is_to_gmm=False,
        start_idx=idx,
        default_output_dir="converted_hyperparam"
    )

    idx = convert_hyperparam(
        path_to_read="hyperparam/params_31.json",
        model="AFTForest",
        is_to_gmm=False,
        start_idx=idx,
        default_output_dir="converted_hyperparam"
    )

    idx = convert_hyperparam(
        path_to_read="hyperparam/params_19.json",
        model="AFTForest",
        is_to_gmm=True,
        start_idx=idx,
        default_output_dir="converted_hyperparam"
    )

    idx = convert_hyperparam(
        path_to_read="hyperparam/params_28.json",
        model="AFTForest",
        is_to_gmm=False,
        start_idx=idx,
        default_output_dir="converted_hyperparam"
    )

    idx = convert_hyperparam(
        path_to_read="hyperparam/params_8.json",
        model="AFTForest",
        is_to_gmm=False,
        start_idx=idx,
        default_output_dir="converted_hyperparam"
    )

if __name__ == "__main__":
    run()
