PYTHON := python3
SETUP := model_setup.py
SETUPs3:= model_setup_s3.py
TEST := train_test_chess.py
FILTER := lichess_data_filtering.ipynb
BUCKET := go-bucket-craft
LOCAL_PROBES := linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_0.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_1.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_2.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_3.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_4.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_5.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_6.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_7.pth
S3_PROBES := tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_0.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_1.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_2.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_3.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_4.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_5.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_6.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_7.pth


##PROBING SETTINGS
RANDOM_MODEL_NAME := big_random16M_vocab32_50K
LICHESS_MODEL_NAME := tf_lens_lichess9gb_vocab32_175K
OLD_MODEL_NAME := tf_lens_random_8layers_ckpt_no_optimizer_Bonus
PROBE_DATASET := random
PROBE_CONTROL_DATASET := dummy
TEST_GAMES_DATASET := random
TRAINING_CONFIG := classic

MODEL1 := tf_lens_random_8layers_ckpt_no_optimizer.pth
MAX_ITERS := 20000
MAX_TRAIN_GAMES := 10000


B1 := skypilot-workdir-oscar-3286bef5


setup: $(SETUP)
	$(PYTHON) $(SETUP) \
		--model_name $(RANDOM_MODEL_NAME)

train_probe:
	$(PYTHON) $(TEST) \
		--mode train \
		--probe piece \
		--probe_dataset $(PROBE_DATASET) \
		--model_name $(RANDOM_MODEL_NAME) \
		--training_config $(TRAINING_CONFIG) \
		--max_iters $(MAX_ITERS) \
		--max_train_games $(MAX_TRAIN_GAMES)


test_probe: $(TEST)
	$(PYTHON) $(TEST) \
		--mode test \
		--probe piece \
		--probe_dataset $(PROBE_DATASET) \
		--model_name $(RANDOM_MODEL_NAME) \
		--test_games_dataset $(TEST_GAMES_DATASET)
		--verbose

test_control_probe:
	$(PYTHON) $(TEST) \
		--mode test \
		--probe piece \
		--model_name $(MODEL_NAME) \
		--probe_dataset $(PROBE_CONTROL_DATASET) \
		--test_games_dataset $(TEST_GAMES_DATASET)


##this function is not debuged yet
##TODO check that this works as required

run_probe_experiments:
	$(PYTHON) run_experiments.py \
		--models big_random16M_vocab32_50K big_random16M_vocab32_100K big_random16M_vocab32_200K big_random16M_vocab32_300K \
		--probe_datasets random \
		--training_configs $(TRAINING_CONFIG) \
		--test_games_datasets random \
		--max_iters $(MAX_ITERS) \
		--max_train_games $(MAX_TRAIN_GAMES) \
		--verbose



remote_train_probe:
	sky jobs launch -c boardCluster remote/train_probes.yaml

remote_test_probe:
	sky jobs launch -c boardCluster remote/test_probes.yaml

remote_sanity_check_probe:
	sky jobs launch -c boardCluster remote/sanity_probes.yaml

dummy_probes:
	python3 saveDummyProbe.py

upload_trained_probes:
	$(PYTHON) remote/upload_file_to_s3.py \
		--bucket_name $(BUCKET) \
		--file_paths $(LOCAL_PROBES)

download_trained_probes:
	$(PYTHON) remote/download_file_from_s3.py \
		--bucket_name $(BUCKET) \
		--s3_keys $(S3_PROBES) \
		--download_paths $(LOCAL_PROBES)

filter: $(FILTER)
	$(PYTHON) $(FILTER)


sups3:
	aws s3 rm s3://$(B1) --recursive
	aws s3 rb s3://$(B1)
	aws s3 rm s3://$(B2) --recursive
	aws s3 rb s3://$(B2)
	aws s3 rm s3://$(B3) --recursive
	aws s3 rb s3://$(B3)
	aws s3 rm s3://$(B4) --recursive
	aws s3 rb s3://$(B4)
	aws s3 rm s3://$(B5) --recursive
	aws s3 rb s3://$(B5)
	aws s3 rm s3://$(B6) --recursive
	aws s3 rb s3://$(B6)
	aws s3 rm s3://$(B7) --recursive
	aws s3 rb s3://$(B7)
	aws s3 rm s3://$(B8) --recursive
	aws s3 rb s3://$(B8)
	aws s3 rm s3://$(B9) --recursive
	aws s3 rb s3://$(B9)
	aws s3 rm s3://$(B0) --recursive
	aws s3 rb s3://$(B0)
	aws s3 rm s3://$(Ba) --recursive
	aws s3 rb s3://$(Ba)
	aws s3 rm s3://$(Bb) --recursive
	aws s3 rb s3://$(Bb)
	aws s3 rm s3://$(Bc) --recursive
	aws s3 rb s3://$(Bc)
	aws s3 rm s3://$(Bd) --recursive
	aws s3 rb s3://$(Bd)
	aws s3 rm s3://$(Be) --recursive
	aws s3 rb s3://$(Be)
	aws s3 rm s3://$(Bf) --recursive
	aws s3 rb s3://$(Bf)








