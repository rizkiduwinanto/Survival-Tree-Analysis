import argparse
from utils.generate_hyperparam import generate_hyperparam

def run(args):
    n_tries = args[0]
    basic = args[1] if args[1] is not None else False
    dry_run = args[2] if args[2] is not None else False
    model = args[3] if args[3] is not None else 'AFTForest'

    generate_hyperparam(
        n_tries=n_tries,
        dry_run=dry_run,
        basic=basic,
        model=model
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Hyperparameter Generation')
    parser.add_argument('--n_tries', type=int, default=10, help='Number of tries for hyperparameter generation')
    parser.add_argument('--basic', action=argparse.BooleanOptionalAction, help='Use basic hyperparameters')
    parser.add_argument('--dry_run', action=argparse.BooleanOptionalAction, help='Dry Run')
    parser.add_argument('--model', type=str, default='AFTForest', help='Model type for hyperparameter generation')  

    args = parser.parse_args()

    run([
        args.n_tries,
        args.basic,
        args.dry_run,
        args.model
    ])
