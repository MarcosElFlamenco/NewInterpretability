import train_test_chess as probes

def train_and_test_probes_on_models(args):
    for model in args.models:
        train_test_chess.py \
        --mode train \
        --probe piece \
        --probe_dataset random  \
        --model_dataset random \
        --training_config classic


2024-12-10 11:46:25,809 - probe_training_u

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

    train_and_test_probes_on_models(args)



