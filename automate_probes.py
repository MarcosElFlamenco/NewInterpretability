from train_test_chess import main

def train_and_test_probes_on_models(args):
    for model in args.models:


        # Simulate the command-line arguments in Python
        main([
            "--mode", "train",
            "--probe", "piece",
            "--probe_dataset", "random",
            "--model_name", model,
            "--training_config", "classic"
        ])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train or test chess probes on piece or skill data."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+'
)
    parser.add_argument(
        "--training-set",
        type=str,
    )
    parser.add_argument(
        "--testing-set",
        type=str,
    )
    args = parser.parse_args()
    train_and_test_probes_on_models(args)



