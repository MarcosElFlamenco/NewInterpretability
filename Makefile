PYTHON := python3
SETUP := model_setup.py
TEST := train_test_chess.py

##PROBING SETTINGS
RANDOM_MODEL_NAME := random_karvhypNS_400K
LICHESS_MODEL_NAME := lichess_karvhyp_10K
MODEL := tf_lens_random_8layers_ckpt_no_optimizer.pth

PROBE_DATASET := lichess 
TEST_GAMES_DATASET := random

TRAINING_CONFIG := classic
MAX_TRAIN_GAMES := 10000
NUM_EPOCHS := 3

happy:
	$(PYTHON) run_experiments.py \
		--models lichess_karvhyp_10K \
		--probe_datasets $(PROBE_DATASET) \
		--training_configs $(TRAINING_CONFIG) \
		--test_games_datasets lichess \
		--max_train_games $(MAX_TRAIN_GAMES) \
		--num_epochs $(NUM_EPOCHS) \



train_probe:
	$(PYTHON) $(TEST) \
		--mode train \
		--probe piece \
		--probe_dataset random \
		--model_name $(LICHESS_MODEL_NAME) \
		--training_config $(TRAINING_CONFIG) \
		--max_train_games $(MAX_TRAIN_GAMES) \
		--num_epochs $(NUM_EPOCHS)

test_probe: $(TEST)
	$(PYTHON) $(TEST) \
		--mode test \
		--probe piece \
		--probe_dataset random \
		--model_name lichess_karvhyp_600K \
		--training_config classic \
		--max_train_games $(MAX_TRAIN_GAMES) \
		--test_games_dataset lichess \
		--verbose

test_control_probe:
	$(PYTHON) $(TEST) \
		--mode test \
		--probe piece \
		--model_name $(MODEL_NAME) \
		--probe_dataset $(PROBE_CONTROL_DATASET) \
		--test_games_dataset $(TEST_GAMES_DATASET)

ALL_RANDOM_MODELS := random_karvhypNS_450K random_karvhypNS_400K random_karvhypNS_350K random_karvhypNS_300K random_karvhypNS_250K random_karvhypNS_200K random_karvhypNS_150K random_karvhypNS_100K 
ALL_LICHESS_MODELS := lichess_karvhyp_600K lichess_karvhyp_500K lichess_karvhyp_400K lichess_karvhyp_300K lichess_karvhyp_200K lichess_karvhyp_100K lichess_karvhyp_550K lichess_karvhyp_450K lichess_karvhyp_350K lichess_karvhyp_250K lichess_karvhyp_150K lichess_karvhyp_50K



run_probe_long:
	$(PYTHON) run_experiments.py \
		--models $(ALL_RANDOM_MODELS)
		--probe_datasets $(PROBE_DATASET) \
		--training_configs $(TRAINING_CONFIG) \
		--test_games_datasets lichess \
		--max_train_games $(MAX_TRAIN_GAMES) \
		--num_epochs $(NUM_EPOCHS) \

run_probe_experiments_nightime:
	$(PYTHON) run_experiments.py \
		--models $(ALL_LICHESS_MODELS) $(ALL_RANDOM_MODELS)\
		--probe_datasets random lichess \
		--training_configs $(TRAINING_CONFIG) \
		--test_games_datasets lichess random \
		--max_train_games $(MAX_TRAIN_GAMES) \
		--num_epochs 3 \
		--verbose \
		--test



setup: $(SETUP)
	$(PYTHON) $(SETUP) \
		--model_name $(RANDOM_MODEL_NAME)



##remote


BUCKET := go-bucket-craft
B1 := skypilot-workdir-oscar-3286bef5

remote_train_probe:
	sky jobs launch -c boardCluster remote/train_probes.yaml

remote_test_probe:
	sky jobs launch -c boardCluster remote/test_probes.yaml

remote_sanity_check_probe:
	sky jobs launch -c boardCluster remote/sanity_probes.yaml

upload_trained_probes:
	$(PYTHON) remote/upload_file_to_s3.py \
		--bucket_name $(BUCKET) \
		--file_paths $(LOCAL_PROBES)

download_trained_probes:
	$(PYTHON) remote/download_file_from_s3.py \
		--bucket_name $(BUCKET) \
		--s3_keys $(S3_PROBES) \
		--download_paths $(LOCAL_PROBES)

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

