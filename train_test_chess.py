
from transformer_lens import HookedTransformer, HookedTransformerConfig
import einops
import argparse
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
import pandas as pd
import pickle
import logging
from typing import Optional
from torch import Tensor
import collections

from collections import deque
from jaxtyping import Int, Float, jaxtyped
from fancy_einsum import einsum
from beartype import beartype
import chess_utils
import einops
import othello_engine_utils
import othello_utils
from chess_utils import PlayerColor, Config
import probe_training_utils as utils
from probe_training_utils import TrainingParams, SingleProbe, SingleProbeCast, LinearProbeData, get_transformer_lens_model_utils, process_dataframe, get_othello_seqs_string, get_board_seqs_string, get_othello_seqs_int, get_board_seqs_int, get_skill_stack, get_othello_state_stack, prepare_data_batch, populate_probes_dict, TrainingParams, get_one_hot_range, init_logging_dict
import warnings
import os 

warnings.filterwarnings("ignore", category=FutureWarning)

debug = False

logger = logging.getLogger(__name__)

# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)

# Add handler to this logger if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)




MODEL_DIR = utils.MODEL_DIR
D_MODEL = utils.D_MODEL
N_HEADS = utils.N_HEADS
WANDB_PROJECT = utils.WANDB_PROJECT
BATCH_SIZE = utils.BATCH_SIZE
PROBE_DIR = utils.PROBE_DIR
DEVICE = utils.DEVICE
LOG_FREQUENCY = 10


DATA_DIR = "data/"
SAVED_PROBE_DIR = "linear_probes/"
WANDB_LOGGING = False

OTHELLO_SEQ_LEN = 59

@jaxtyped(typechecker=beartype)
def linear_probe_forward_pass(
    linear_probe,
    state_stack_one_hot_MBLRRC: Int[Tensor, "modes batch num_white_moves rows cols options"],
    resid_post_BLD: Float[Tensor, "batch num_white_moves d_model"],
    one_hot_range: int,
) -> tuple[Tensor, Tensor]:
    """Outputs are scalar tensors."""
    probe_out_MBLRRC = linear_probe.forward_pass(resid_post_BLD)

    predictions = probe_out_MBLRRC[0].argmax(-1)
    ground_truth = state_stack_one_hot_MBLRRC[0].argmax(-1)
    accuracy_grid = (predictions == ground_truth)
    accuracy = accuracy_grid.float().mean()

    probe_log_probs_MBLRRC = probe_out_MBLRRC.log_softmax(-1)
    probe_correct_log_probs_MLRR = (
        einops.reduce(
            probe_log_probs_MBLRRC * state_stack_one_hot_MBLRRC,
            "modes batch pos rows cols options -> modes pos rows cols",
            "mean",
        )
        * one_hot_range
    )  # Multiply to correct for the mean over one_hot_range
    # probe_correct_log_probs shape (modes, num_white_moves, num_rows, num_cols)
    loss = -probe_correct_log_probs_MLRR[0, :].mean(0).sum()

    return loss, accuracy


# helps estimate an arbitrarily accurate loss over either split using many batches
# This is mainly useful for checking that the probe isn't overfitting to the train set
# Note that I'm not doing a proper train/val split here, this was just a quick and dirty way to check for overfitting
# You could disable this if using a training set with over 5k games, as there shouldn't be any overfitting
@torch.no_grad()
def estimate_loss(
    train_games: int,
    val_games: int,
    probes: dict[int, SingleProbe],
    probe_data: LinearProbeData,
    config: Config,
    one_hot_range: int,
    layers: list[int],
    train_params: TrainingParams,
) -> dict[int, dict[str, dict[str, float]]]:
    out = {}

    for layer in probes:
        out[layer] = {
            "train": {"loss": 0.0, "accuracy": 0.0},
            "val": {"loss": 0.0, "accuracy": 0.0},
        }

    eval_iters = (train_params.eval_iters // BATCH_SIZE) * BATCH_SIZE

    train_indices = torch.randperm(train_games)[:eval_iters]
    val_indices = torch.randperm(val_games) + train_games  # to avoid overlap
    val_indices = val_indices[:eval_iters]

    split_indices = {"train": train_indices, "val": val_indices}
    for split in split_indices:
        losses: dict[int, list[float]] = {}
        accuracies: dict[int, list[float]] = {}
        for layer in probes:
            losses[layer] = []
            accuracies[layer] = []
        for k in range(0, eval_iters, BATCH_SIZE):
            indices = split_indices[split][k : k + BATCH_SIZE]

            state_stack_one_hot_MBLRRC, resid_post_dict_BLD = prepare_data_batch(
                indices, probe_data, config, layers
            )

            for layer in probes:
                loss, accuracy = linear_probe_forward_pass(
                    probes[layer],
                    state_stack_one_hot_MBLRRC,
                    resid_post_dict_BLD[layer],
                    one_hot_range,
                )
                losses[layer].append(loss.item())
                accuracies[layer].append(accuracy.item())
        for layer in layers:
            out[layer][split]["loss"] = sum(losses[layer]) / len(losses[layer])
            out[layer][split]["accuracy"] = sum(accuracies[layer]) / len(accuracies[layer])
    return out


def train_linear_probe_cross_entropy(
    probes: dict[int, SingleProbe],
    probe_data: LinearProbeData,
    config: Config,
    train_params: TrainingParams,
    model_name: str,
    config_name: str,
    probe_dataset: str,
    max_games: int,
) -> dict[int, float]:
    """Trains a linear probe on the train set, contained in probe_data. Saves all probes to disk.
    Returns a dict of layer: final avg_acc over the last 1,000 iterations.
    This dict is also used as an end to end test for the function."""
    first_layer = min(probes.keys())
    layers = list(probes.keys())
    all_layers_str = "_".join([str(layer) for layer in layers])
    assert probes[first_layer].logging_dict["split"] == "train", "Don't train on the test set"

    val_games = (train_params.max_val_games // BATCH_SIZE) * BATCH_SIZE
    train_games = (train_params.max_train_games // BATCH_SIZE) * BATCH_SIZE

     
    num_games = train_games + val_games

    if len(probe_data.board_seqs_int) < num_games:
        raise ValueError("Not enough games to train on")
        # We raise an error so it doesn't fail silently. If we want to use less games, we can comment the error out
        # and add some logic to set train and val games to the number of games we have

    one_hot_range = get_one_hot_range(config)

    if WANDB_LOGGING:
        import wandb

        wandb.init(
            project=WANDB_PROJECT,
            name=f"layers:{all_layers_str}_" + probes[first_layer].logging_dict["wandb_run_name"],
            config=probes[first_layer].logging_dict,
        )

    current_iter = 0
    ## note that probes is a single probe (it should be a list of size 1)
    one_probe = probes[layers[0]]
    #print(f"the probe that is being trained looks like this: {one_probe}")
    starting_epoch = one_probe.epoch
    print(f"val games {val_games}, train_games {train_games}, num games {num_games}")

    print(f"Training to {train_params.num_epochs} epochs, {starting_epoch} already trained, {train_params.num_epochs - starting_epoch} left")

    for epoch in range(starting_epoch,train_params.num_epochs):
        full_train_indices = torch.randperm(train_games)
        for i in tqdm(range(0, train_games, BATCH_SIZE)):
            indices_B = full_train_indices[i : i + BATCH_SIZE]  # shape batch_size

            state_stack_one_hot_MBLRRC, resid_post_dict_BLD = prepare_data_batch(
                indices_B, probe_data, config, layers
            )

            for layer in probes:
                probes[layer].loss, probes[layer].accuracy = linear_probe_forward_pass(
                    probes[layer],
                    state_stack_one_hot_MBLRRC,
                    resid_post_dict_BLD[layer],
                    one_hot_range,
                )

                probes[layer].loss.backward()
                probes[layer].optimiser.step()
                probes[layer].optimiser.zero_grad()

                probes[layer].accuracy_queue.append(round(probes[layer].accuracy.item(),2))

            if (i+1) % LOG_FREQUENCY == 0:
                if WANDB_LOGGING:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "iter": current_iter,
                        }
                    )
                for layer in probes:
                    avg_acc = sum(probes[layer].accuracy_queue) / len(probes[layer].accuracy_queue)
                    logger.info(
                        f"epoch {epoch}, iter {i}, layer {layer}, acc {probes[layer].accuracy:.3f}, loss {probes[layer].loss:.3f}, avg acc {avg_acc:.3f} /n the length of the accuracy list is {len(probes[layer].accuracy_queue)}"
                        
                    )
                    if WANDB_LOGGING:
                        wandb.log(
                            {
                                f"layer_{layer}_loss": probes[layer].loss,
                                f"layer_{layer}_acc": probes[layer].accuracy,
                                f"layer_{layer}_avg_acc": avg_acc,
                            }
                        )

            if current_iter % 1000 == 0:
                losses = estimate_loss(
                    train_games,
                    val_games,
                    probes,
                    probe_data,
                    config,
                    one_hot_range,
                    layers,
                    train_params,
                )
                for layer in probes:
                    logger.info(
                        f"epoch {epoch}, layer {layer}, train loss: {losses[layer]['train']['loss']:.3f}, val loss: {losses[layer]['val']['loss']:.3f}, train acc: {losses[layer]['train']['accuracy']:.3f}, val acc: {losses[layer]['val']['accuracy']:.3f}"
                    )
                    if WANDB_LOGGING:
                        wandb.log(
                            {
                                f"layer_{layer}_train_loss": losses[layer]["train"]["loss"],
                                f"layer_{layer}_train_acc": losses[layer]["train"]["accuracy"],
                                f"layer_{layer}_val_loss": losses[layer]["val"]["loss"],
                                f"layer_{layer}_val_acc": losses[layer]["val"]["accuracy"],
                            }
                        )
            current_iter += BATCH_SIZE
    final_accs = {}
    for layer in probes:
        checkpoint = {
            "final_loss": probes[layer].loss,
            "iters": current_iter,
            "model_name": model_name,
            "training_config": config_name,
            "probe_dataset": probe_dataset,
            "max_games": max_games,
            "num_epochs": train_params.num_epochs,
            "accuracy_queue": probes[layer].accuracy_queue,
            "accuracy": probes[layer].accuracy,
        }
        if isinstance(probes[layer],SingleProbe):
            checkpoint["linear_probe_MDRRC"] = probes[layer].linear_probe_MDRRC
        elif isinstance(probes[layer],SingleProbeCast):
            checkpoint["linear_DN"] = probes[layer].linear_DN
            checkpoint["linear_MNRRC"] = probes[layer].linear_MNRRC


        # Update the checkpoint dictionary with the contents of logging_dict
        checkpoint.update(probes[layer].logging_dict)
        torch.save(checkpoint, probes[layer].probe_name)
        if(probes[layer].accuracy_queue):
            final_accs[layer] = sum(probes[layer].accuracy_queue) / len(probes[layer].accuracy_queue)
            logger.info(f"layer {layer}, final acc: {final_accs[layer]}")
        else:
            logger.info(f"layer {layer}, no final acc, because no training")
    return final_accs


def construct_linear_probe_data( 
    input_dataframe_file: str,
    dataset_prefix: str,
    n_layers: int,
    model_name: str,
    config: Config,
    max_games: int,
    device: torch.device,
    verbose: bool,
) -> LinearProbeData:
    """We need the following data to train or test a linear probe:
    - The layer to probe in the GPT
    - The GPT model in transformer_lens format
    - The number of layers in the GPT
    - board_seqs_int: the integer sequences representing the chess games, encoded using meta.pkl
    - board_seqs_string: the string sequences representing the chess games
    - custom_indices: the indices of the moves we want to probe on. By default, these are the indices of every "."
    - skill_stack: the skill levels of the players in the games (only used if probing for skill)
    """

    model = get_transformer_lens_model_utils(model_name, n_layers, device)
    user_state_dict_one_hot_mapping, df = process_dataframe(input_dataframe_file, config)
    if verbose:
        print(f"user state dict one hot {user_state_dict_one_hot_mapping}")
    #this is none and probably shouldnt be
    
    df = df[:max_games]
    board_seqs_string_Bl = get_board_seqs_string(df)
    if verbose:
        print(f'seqs string {board_seqs_string_Bl}')
    board_seqs_int_Bl = get_board_seqs_int(df)
    if verbose:
        print(f'seqs int {board_seqs_int_Bl}')
    skill_stack_B = None
    if config.probing_for_skill:
        skill_stack_B = get_skill_stack(config, df)
    custom_indices = chess_utils.find_custom_indices(
        config.custom_indexing_function, board_seqs_string_Bl
    )

    pgn_str_length = len(board_seqs_string_Bl[0])
    num_games = len(board_seqs_string_Bl)

    assert board_seqs_int_Bl.shape == (num_games, pgn_str_length)

    if skill_stack_B is not None:
        assert skill_stack_B.shape == (num_games,)

    _, shortest_game_length_in_moves = custom_indices.shape
    assert custom_indices.shape == (num_games, shortest_game_length_in_moves)

    if not config.pos_end:
        config.pos_end = shortest_game_length_in_moves

    probe_data = LinearProbeData(
        model=model,
        custom_indices=custom_indices,
        board_seqs_int=board_seqs_int_Bl,
        board_seqs_string=board_seqs_string_Bl,
        skill_stack=skill_stack_B,
        user_state_dict_one_hot_mapping=user_state_dict_one_hot_mapping,
    )

    return probe_data



@torch.no_grad()
def test_linear_probe_cross_entropy(
    linear_probe_name: str,
    probe_data: LinearProbeData,
    config: Config,
    logging_dict: dict,
    train_params: TrainingParams,
    test_games_dataset: str,
) -> float:
    """Takes a linear probe and tests it on the test set, contained in probe_data. Saves the results to a pickle file.
    Returns a float representing the average accuracy of the probe on the test set. This is also used as an end to end test for the function.
    """
    assert logging_dict["split"] == "test", "Don't test on the train set"

    num_games = (train_params.max_test_games // BATCH_SIZE) * BATCH_SIZE

    if (len(probe_data.board_seqs_int) // BATCH_SIZE) * BATCH_SIZE < num_games:
        raise ValueError("Not enough games to test on")
        # We raise an error so it doesn't fail silently. If we want to use less games, we can comment the error out
        num_games = (len(probe_data.board_seqs_int) // BATCH_SIZE) * BATCH_SIZE

    one_hot_range = get_one_hot_range(config)

    logging_dict["num_games"] = num_games

    checkpoint = torch.load(linear_probe_name, map_location=DEVICE)
    if config.probe_type == 'vanilla':
        linear_probe_MDRRC = checkpoint["linear_probe_MDRRC"]
        probe = SingleProbe(
            linear_probe_MDRRC=linear_probe_MDRRC,
            probe_name="dummy_probe",
            optimiser=None,  # Not needed for inference
            logging_dict={},
            epoch=0,
            loss=torch.tensor(0.0),
            accuracy=torch.tensor(0.0),
            accuracy_queue=deque(maxlen=1000),
        )

        logger.info(f"linear_probe shape: {linear_probe_MDRRC.shape}")
    elif config.probe_type == 'cast':
        linear_DN = checkpoint["linear_DN"]
        linear_MNRRC = checkpoint["linear_MNRRC"]
        probe = SingleProbeCast(
            linear_DN=linear_DN,
            linear_MNRRC=linear_MNRRC,
            probe_name="dummy_probe",
            optimiser=None,  # Not needed for inference
            logging_dict={},
            epoch=0,
            loss=torch.tensor(0.0),
            accuracy=torch.tensor(0.0),
            accuracy_queue=deque(maxlen=1000),
        )





    layer = logging_dict["layer"]

    current_iter = 0
    accuracy_list = []
    loss_list = []
    full_test_indices = torch.arange(0, num_games)
    for i in tqdm(range(0, num_games, BATCH_SIZE)):
        indices_B = full_test_indices[i : i + BATCH_SIZE]  # shape batch_size

        state_stack_one_hot_MBLRRC, resid_post_dict_BLD = prepare_data_batch(
            indices_B, probe_data, config, [layer]
        )


        loss, accuracy = linear_probe_forward_pass(
            probe,
            state_stack_one_hot_MBLRRC,
            resid_post_dict_BLD[layer],
            one_hot_range,
        )
        accuracy_list.append(accuracy.item())
        loss_list.append(loss.item())

        if i % 100 == 0:
            average_accuracy = sum(accuracy_list) / len(accuracy_list)
            logger.info(
                f"batch {i}, average accuracy: {average_accuracy}, acc {accuracy}, loss {loss}"
            )

        current_iter += BATCH_SIZE
    data = {
        "accuracy": accuracy_list,
        "loss": loss_list,
    }

    output_probe_data_name = linear_probe_name.split("/")[-1].split(".")[0]
    output_location = f"{PROBE_DIR}test_data/{output_probe_data_name}_{test_games_dataset}.pkl"

    logger.info(f"Saving test data to {output_location}")
    average_accuracy = sum(accuracy_list) / len(accuracy_list)
    logger.info(f"Average accuracy: {average_accuracy}")

    with open(output_location, "wb") as f:
        pickle.dump(data, f)

    return average_accuracy


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train or test chess probes on piece or skill data."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "train"],
        default="train",
        help='Mode to run the script in: "test" or "train".',
    )
    parser.add_argument(
        "--probe",
        type=str,
        choices=["piece", "skill"],
        default="piece",
        help='Type of probe to use: "piece" for piece board state or "skill" for player skill level.',
    )
    parser.add_argument(
        "--wandb_logging",
        action="store_true",
        help="Enable logging to Weights & Biases. Default is False.",
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        help='model path'
    )
    parser.add_argument(
        '--layers_to_train',
        nargs= "+",
        type=int,
        help='you may specify as many integers as you want, those are the layers the probes will train on'
    )
    
    parser.add_argument(
        '--probe_dataset', 
        type=str, 
        help='The dataset to train the probes on'
    )
    
    parser.add_argument(
        '--test_games_dataset', 
        type=str, 
        help='the dataset to test the probes on'
    )
 
    parser.add_argument(
        '--training_config', 
        type=str, 
        help='the kind of training configuration/probe type'
    )
    parser.add_argument(
        '--max_train_games', 
        type=int, 
        help='maximum games that will be read from the dataset file to be trained on'
    )
    parser.add_argument(
        '--verbose', 
        action="store_true",
        help='prints verbose'
    )
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        help='num epochs in training, each epoch is one pass over the whole dataset'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    WANDB_LOGGING = args.wandb_logging
    args.tf_model_name = "tf_lens_" + args.model_name
    args.model_path = "tf_lens_models/tf_lens_" + args.model_name

    TRAIN_PARAMS = TrainingParams(max_train_games=args.max_train_games,num_epochs=args.num_epochs)
    if args.training_config == 'classic':
        config = chess_utils.piece_config
    elif args.training_config == 'custom':
        config = chess_utils.custom_piece_config
    elif args.training_config == 'cast64':
        config = chess_utils.cast64_piece_config
    elif args.training_config == 'cast32':
        config = chess_utils.cast32_piece_config
    elif args.training_config == 'cast16':
        config = chess_utils.cast16_piece_config
    elif args.training_config == 'cast8':
        config = chess_utils.cast8_piece_config
    elif args.training_config == 'cast1':
        config = chess_utils.cast1_piece_config
    elif args.probe == "skill":
        config = chess_utils.skill_config



    if args.mode == "test":
        print('TESTING PROBES')
        # saved_probes = [
        #     file
        #     for file in os.listdir(SAVED_PROBE_DIR)
        #     if os.path.isfile(os.path.join(SAVED_PROBE_DIR, file))
        # ]
        saved_probes = []

        print(f"[FROM MAIN CODE] testing: probe_dataset {args.probe_dataset} model_name {args.model_name} test_dataset {args.test_games_dataset}")
        
        if args.probe == 'piece':
            for i in range(1):
                linear_probe_name = (
                f"{args.model_name}_{args.training_config}_{args.probe_dataset}_{args.max_train_games}_{config.linear_probe_name}.pth"
                )
                linear_probe_location = f"{PROBE_DIR}{linear_probe_name}"
                if os.path.exists(linear_probe_location):
                   saved_probes.append(linear_probe_name)
                else:
                    print(f"We did not find {linear_probe_name}, skipping testing")
        elif args.probe == "skill":
            saved_probes = [
                "tf_lens_lichess_8layers_ckpt_no_optimizer_chess_skill_probe_layer_5.pth"
            ]

        # NOTE: This is very inefficient. The expensive part is forwarding the GPT, which we should only have to do once.
        # With little effort, we could test probes on all layers at once. This would be much faster.
        # But I can test the probes in 20 minutes and it was a one-off thing, so I didn't bother.
        # My strategy for development / hyperparameter testing was to iterate on the train side, then do the final test on the test side.
        # As long as you have a reasonable training dataset size, you should be able to get a good idea of final test accuracy
        # by looking at the training accuracy after a few epochs.
        for probe_to_test in saved_probes:
            probe_file_location = f"{PROBE_DIR}{probe_to_test}"
            # We will populate all parameters using information in the probe state dict
            with open(probe_file_location, "rb") as f:
                state_dict = torch.load(f, map_location=torch.device(DEVICE))
                #config = chess_utils.find_config_by_name(state_dict["config_name"])
                layer = state_dict["layer"]
                #model_name = state_dict["model_name"]
                n_layers = 8
                dataset_prefix = state_dict["dataset_prefix"]
                print(f'retrieved the following dataset prefix from probe: {dataset_prefix} \
                    \n We will now gladly ignore this information')
                config.pos_start = state_dict["pos_start"]
                levels_of_interest = None
                if "levels_of_interest" in state_dict.keys():
                    levels_of_interest = state_dict["levels_of_interest"]
                config.levels_of_interest = levels_of_interest
                n_layers = state_dict["n_layers"]
                split = "test"
                input_dataframe_file = f"{DATA_DIR}{args.test_games_dataset}_{split}.csv"

                config = chess_utils.set_config_min_max_vals_and_column_name(
                    config, input_dataframe_file, args.test_games_dataset,args.verbose 
                )
                probe_data = construct_linear_probe_data(
                    input_dataframe_file,
                    args.test_games_dataset,
                    n_layers,
                    args.model_path,
                    config,
                    TRAIN_PARAMS.max_test_games,
                    DEVICE,
                    args.verbose
                )

                logging_dict = init_logging_dict(
                    layer, config, split, dataset_prefix, args.model_path, n_layers, TRAIN_PARAMS, config.probe_type
                )

                avg_accuracy = test_linear_probe_cross_entropy(
                    probe_file_location, probe_data, config, logging_dict, TRAIN_PARAMS, args.test_games_dataset
                )

    elif args.mode == "train":
        print('TRAINING PROBE')

        player_color = PlayerColor.WHITE
        first_layer = 0
        last_layer = 7

        # When training a probe, you have to set all parameters such as model name, dataset prefix, etc.
#        dataset_prefix = "lichess_"


        split = "train"
        n_layers = 8
        indexing_function = None


        input_dataframe_file = f"{DATA_DIR}{args.probe_dataset}_{split}.csv"
        config = chess_utils.set_config_min_max_vals_and_column_name(
            config, input_dataframe_file, args.probe_dataset, args.verbose
        )
        config = chess_utils.update_config_using_player_color(
            player_color, config, indexing_function
        )

        max_games = TRAIN_PARAMS.max_train_games + TRAIN_PARAMS.max_val_games
        probe_data = construct_linear_probe_data(
            input_dataframe_file,
            args.probe_dataset,
            n_layers,
            args.model_path,
            config,
            max_games,
            DEVICE,
            args.verbose
        )

#        layers = list(range(first_layer, last_layer + 1))
        layers = args.layers_to_train
        probes = populate_probes_dict(
            layers,
            config,
            TRAIN_PARAMS,
            split,
            args.probe_dataset,
            args.model_name,
            n_layers,
            args
        )

        train_linear_probe_cross_entropy(probes, probe_data, config, TRAIN_PARAMS,args.model_name,args.training_config,args.probe_dataset, max_games)
