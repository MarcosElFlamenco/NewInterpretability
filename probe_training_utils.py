import torch
import pandas as pd
from torch import Tensor
from dataclasses import dataclass, field
import collections
from typing import Optional
from transformer_lens import HookedTransformer, HookedTransformerConfig
from chess_utils import PlayerColor, Config
import logging
from fancy_einsum import einsum
from jaxtyping import Int, Float, jaxtyped
from beartype import beartype
import chess_utils
import pickle
import einops

MODEL_DIR = "models/"
D_MODEL = 512
N_HEADS = 8
WANDB_PROJECT = "chess_linear_probes"
BATCH_SIZE = 2
PROBE_DIR = "linear_probes/"

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

# meta is used to encode the string pgn strings into integer sequences
with open(f"{MODEL_DIR}meta.pkl", "rb") as f:
    meta = pickle.load(f)

logger.info(meta)

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
logger.info(f"Using device: {DEVICE}")


stoi, itos = meta["stoi"], meta["itos"]
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

meta_round_trip_input = "1.e4 e6 2.Nf3"
logger.info(encode(meta_round_trip_input))
logger.info("Performing round trip test on meta")
assert decode(encode(meta_round_trip_input)) == meta_round_trip_input


@dataclass
class TrainingParams:
    modes: int = 1
    # modes currently doesn't do anything, but it is used and adds a dimension to the tensors
    # In the future, modes could be used to do clever things like training multiple probes at once, such as a black piece probe and a white piece probe
    wd: float = 0.01
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.99
    max_train_games: int = 10000
    max_test_games: int = 10000
    max_val_games: int = 1000
    max_iters: int = 50000
    eval_iters: int = 50
    num_epochs: int = max_iters // max_train_games

TRAIN_PARAMS = TrainingParams()

@dataclass
class SingleProbe:
    linear_probe: torch.Tensor
    probe_name: str
    optimiser: torch.optim.AdamW
    logging_dict: dict
    loss: torch.Tensor = torch.tensor(0.0)
    accuracy: torch.Tensor = torch.tensor(0.0)
    accuracy_queue: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=1000)
    )


@dataclass
class LinearProbeData:
    model: HookedTransformer
    custom_indices: torch.Tensor
    board_seqs_int: torch.Tensor
    board_seqs_string: list[str]
    skill_stack: torch.Tensor
    user_state_dict_one_hot_mapping: Optional[dict[int, int]] = None


def init_logging_dict(
    layer: int,
    config: Config,
    split: str,
    dataset_prefix: str,
    model_name: str,
    n_layers: int,
    train_params: TrainingParams,
) -> dict:

    indexing_function_name = config.custom_indexing_function.__name__

    wandb_run_name = f"{config.linear_probe_name}_{model_name}_layer_{layer}_indexing_{indexing_function_name}_max_games_{train_params.max_train_games}"
    if config.levels_of_interest is not None:
        wandb_run_name += "_levels"
        for level in config.levels_of_interest:
            wandb_run_name += f"_{level}"

    logging_dict = {
        "linear_probe_name": config.linear_probe_name,
        "layer": layer,
        "indexing_function_name": indexing_function_name,
        "batch_size": BATCH_SIZE,
        "lr": train_params.lr,
        "wd": train_params.wd,
        "pos_start": config.pos_start,
        "num_epochs": train_params.num_epochs,
        "num_games": train_params.max_train_games,
        "modes": train_params.modes,
        "wandb_project": WANDB_PROJECT,
        "config_name": config.linear_probe_name,
        "column_name": config.column_name,
        "levels_of_interest": config.levels_of_interest,
        "split": split,
        "dataset_prefix": dataset_prefix,
        "model_name": model_name,
        "n_layers": n_layers,
        "wandb_run_name": wandb_run_name,
        "player_color": config.player_color.value,
    }

    return logging_dict


def get_transformer_lens_model_utils(
    model_name: str, n_layers: int, device: torch.device
) -> HookedTransformer:

    if model_name == "Baidicoot/Othello-GPT-Transformer-Lens":
        return HookedTransformer.from_pretrained(model_name).to(device)

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=D_MODEL,
        d_head=int(D_MODEL / N_HEADS),
        n_heads=N_HEADS,
        d_mlp=D_MODEL * 4,
        d_vocab=32,
        n_ctx=1023,
        act_fn="gelu",
        normalization_type="LNPre",
    )
    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(f"{MODEL_DIR}{model_name}.pth"))
    model.to(device)
    return model

def process_dataframe(
    input_dataframe_file: str,
    config: Config,
) -> tuple[Optional[dict], pd.DataFrame]:
    """This is used if we want to have our model do classification on a subset of the Elo bins.
    There are 6 Elo bins. If we want our model to classify between bin 0 and bin 5, we can use this function to
    filter the DataFrame to only include games from these bins."""
    df = pd.read_csv(input_dataframe_file)
    user_state_dict_one_hot_mapping = None

    if config.levels_of_interest is not None:
        user_state_dict_one_hot_mapping = {}
        for i in range(len(config.levels_of_interest)):
            user_state_dict_one_hot_mapping[config.levels_of_interest[i]] = i

        matches = {number for number in config.levels_of_interest}
        logger.info(f"Levels to be used in probe dataset: {matches}")

        # Filter the DataFrame based on these matches
        cn = config.column_name
        column = df[cn]
        df = df[column.isin(matches)]
        logger.info(f"Number of games in filtered dataset: {len(df)}")

        df = df.reset_index(drop=True)

    return user_state_dict_one_hot_mapping, df

def get_othello_seqs_string(df: pd.DataFrame) -> list[str]:
    key = "tokens"

    # convert every string to a list of integers
    if isinstance(df[key].iloc[0], str):
        df[key] = df[key].apply(lambda x: list(map(int, x.strip("[] ").split(","))))

    board_seqs_string_Bl = df[key].tolist()
    board_seqs_string_Bl = othello_engine_utils.to_string(board_seqs_string_Bl)

    for i in range(len(board_seqs_string_Bl)):
        board_seqs_string_Bl[i] = board_seqs_string_Bl[i][:OTHELLO_SEQ_LEN]

    logger.info(
        f"Number of games: {len(board_seqs_string_Bl)},length of a game in chars: {len(board_seqs_string_Bl[0])}"
    )
    return board_seqs_string_Bl

def get_board_seqs_string(df: pd.DataFrame) -> list[str]:

    if "tokens" in df.columns:
        return get_othello_seqs_string(df)

    key = "transcript"
    row_length = len(df[key].iloc[0])

    assert all(
        df[key].apply(lambda x: len(x) == row_length)
    ), "Not all transcripts are of length {}".format(row_length)

    board_seqs_string_Bl = df[key]

    logger.info(
        f"Number of games: {len(board_seqs_string_Bl)},length of a game in chars: {len(board_seqs_string_Bl[0])}"
    )
    return board_seqs_string_Bl

def get_othello_seqs_int(df: pd.DataFrame) -> Int[Tensor, "num_games pgn_str_length"]:
    tokens_list = df["tokens"].tolist()
    tokens_tensor_Bl = torch.tensor(tokens_list)
    tokens_tensor_Bl = tokens_tensor_Bl[:, :OTHELLO_SEQ_LEN]  # OthelloGPT has context length of 59
    logger.info(f"tokens_tensor shape: {tokens_tensor_Bl.shape}")
    return tokens_tensor_Bl


@jaxtyped(typechecker=beartype)
def get_board_seqs_int(df: pd.DataFrame) -> Int[Tensor, "num_games pgn_str_length"]:
    if "tokens" in df.columns:
        return get_othello_seqs_int(df)

    encoded_df = df["transcript"].apply(encode)
    logger.info(encoded_df.head())
    board_seqs_int_Bl = torch.tensor(encoded_df.apply(list).tolist())
    logger.info(f"board_seqs_int shape: {board_seqs_int_Bl.shape}")
    return board_seqs_int_Bl


@jaxtyped(typechecker=beartype)
def get_skill_stack(config: Config, df: pd.DataFrame) -> Int[Tensor, "num_games"]:
    skill_levels_list = df[config.column_name].tolist()

    skill_stack_B = torch.tensor(skill_levels_list)
    logger.info(f"Unique values in skill_stack: {skill_stack_B.unique()}")
    logger.info(f"skill_stack shape: {skill_stack_B.shape}")
    return skill_stack_B


def get_othello_state_stack(
    config: chess_utils.Config, games_str_Bl: list[str]
) -> Int[Tensor, "modes batch num_white_moves num_rows num_cols"]:
    state_stack_MBLRRC = config.custom_board_state_function(games_str_Bl)
    return state_stack_MBLRRC

def get_one_hot_range(config: Config) -> int:
    one_hot_range = config.max_val - config.min_val + 1
    if config.levels_of_interest is not None:
        one_hot_range = len(config.levels_of_interest)
    return one_hot_range


@jaxtyped(typechecker=beartype)
def prepare_data_batch(
    indices: Int[Tensor, "batch_size"],
    probe_data: LinearProbeData,
    config: Config,
    layers: list[int],
) -> tuple[
    Int[Tensor, "modes batch_size num_white_moves num_rows num_cols num_options"],
    dict[int, Float[Tensor, "batch_size num_white_moves d_model"]],
]:
    list_of_indices = indices.tolist()  # For indexing into the board_seqs_string list of strings
    games_int_Bl = probe_data.board_seqs_int[
        indices
    ]  # games_int shape (batch_size, pgn_str_length)
    games_str_Bl = [probe_data.board_seqs_string[idx] for idx in list_of_indices]
    games_str_Bl = [s[:] for s in games_str_Bl]
    games_dots_BL = probe_data.custom_indices[indices]
    games_dots_BL = games_dots_BL[
        :, config.pos_start : config.pos_end
    ]  # games_dots shape (batch_size, num_white_moves)

    if config.probing_for_skill:
        games_skill_B = probe_data.skill_stack[indices]  # games_skill shape (batch_size,)
    else:
        games_skill_B = None

    if config.othello:
        state_stack_one_hot_BlRRC = othello_utils.games_batch_to_state_stack_mine_yours_BLRRC(
            games_str_Bl
        ).to(DEVICE)
        state_stack_one_hot_MBlRRC = einops.repeat(
            state_stack_one_hot_BlRRC, "B L R1 R2 C -> M B L R1 R2 C", M=TRAIN_PARAMS.modes
        )

    else:
        state_stack_MBlRR = chess_utils.create_state_stacks(
            games_str_Bl, config.custom_board_state_function, games_skill_B
        )  # shape (modes, batch_size, pgn_str_length, num_rows, num_cols)

        state_stack_one_hot_MBlRRC = chess_utils.state_stack_to_one_hot(
            TRAIN_PARAMS.modes,
            config.num_rows,
            config.num_cols,
            config.min_val,
            config.max_val,
            DEVICE,
            state_stack_MBlRR,
            probe_data.user_state_dict_one_hot_mapping,
        ).to(DEVICE)

    indexed_state_stacks_one_hot_MBLRRC = []

    for batch_idx in range(BATCH_SIZE):
        # Get the indices for the current batch
        dots_indices_for_batch_L = games_dots_BL[batch_idx]

        # Index the state_stack for the current batch
        print(f"the selected indices are {dots_indices_for_batch_L}")
        indexed_state_stack_one_hot_MLRRC = state_stack_one_hot_MBlRRC[
            :, batch_idx, dots_indices_for_batch_L, :, :, :
        ]

        # Append the result to the list
        indexed_state_stacks_one_hot_MBLRRC.append(indexed_state_stack_one_hot_MLRRC)

    # Stack the indexed state stacks along the first dimension
    state_stack_one_hot_BMLRRC = torch.stack(indexed_state_stacks_one_hot_MBLRRC)

    # Use einops to rearrange the dimensions after stacking
    state_stack_one_hot_MBLRRC = einops.rearrange(
        state_stack_one_hot_BMLRRC,
        "batch modes pos row col classes -> modes batch pos row col classes",
    )

    resid_post_dict_BLD = {}

    with torch.inference_mode():
        _, cache = probe_data.model.run_with_cache(games_int_Bl.to(DEVICE)[:, :], return_type=None)
        for layer in layers:
            resid_post_dict_BLD[layer] = cache["resid_post", layer][
                :, :
            ]  # shape (batch_size, pgn_str_length - 1, d_model)

    # Not the most efficient way to do this, but it's clear and readable
    for layer in layers:
        resid_post_BLD = resid_post_dict_BLD[layer]
        # Initialize a list to hold the indexed state stacks
        indexed_resid_posts_BLD = []

        for batch_idx in range(games_dots_BL.size(0)):
            # Get the indices for the current batch
            dots_indices_for_batch_L = games_dots_BL[batch_idx]

            # Index the state_stack for the current batch
            indexed_resid_post_LD = resid_post_BLD[batch_idx, dots_indices_for_batch_L]

            # Append the result to the list
            indexed_resid_posts_BLD.append(indexed_resid_post_LD)

        # Stack the indexed state stacks along the first dimension
        resid_post_dict_BLD[layer] = torch.stack(
            indexed_resid_posts_BLD
        )  # shape (batch_size, num_white_moves, d_model)

    return state_stack_one_hot_MBLRRC, resid_post_dict_BLD


def populate_probes_dict(
    layers: list[int],
    config: Config,
    train_params: TrainingParams,
    split,
    dataset_prefix,
    model_name,
    n_layers,
) -> dict[int, SingleProbe]:
    probes = {}
    for layer in layers:
        logging_dict = init_logging_dict(
            layer, config, split, dataset_prefix, model_name, n_layers, TRAIN_PARAMS
        )
        linear_probe_name = (
            f"{PROBE_DIR}{logging_dict['model_name']}_{config.linear_probe_name}_layer_{layer}.pth"
        )
        linear_probe_MDRRC = torch.randn(
            train_params.modes,
            D_MODEL,
            config.num_rows,
            config.num_cols,
            get_one_hot_range(config),
            requires_grad=False,
            device=DEVICE,
        ) / torch.sqrt(torch.tensor(D_MODEL))
        linear_probe_MDRRC.requires_grad = True
        logger.info(f"linear_probe shape: {linear_probe_MDRRC.shape}")

        optimiser = torch.optim.AdamW(
            [linear_probe_MDRRC],
            lr=train_params.lr,
            betas=(train_params.beta1, train_params.beta2),
            weight_decay=train_params.wd,
        )
        probes[layer] = SingleProbe(
            linear_probe=linear_probe_MDRRC,
            probe_name=linear_probe_name,
            optimiser=optimiser,
            logging_dict=logging_dict,
        )
    return probes

