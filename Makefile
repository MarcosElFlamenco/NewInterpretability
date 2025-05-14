PYTHON := python3
SETUP := model_setup.py
TEST := train_test_chess.py

##PROBING SETTINGS
RANDOM_MODEL_NAME := random_karvhypNS_400K
LICHESS_MODEL_NAME := lichess_karvhyp_10K
MODEL := tf_lens_random_8layers_ckpt_no_optimizer.pth

PROBE_DATASET := lichess 
TEST_GAMES_DATASET := random

TRAINING_CONFIG := cast32
MAX_TRAIN_GAMES := 10000
NUM_EPOCHS := 3

# this trains a probe on specified dataset
train_probe:
	$(PYTHON) $(TEST) \
		--mode train \
		--probe piece \
		--probe_dataset random \
		--num_epochs 1 \
		--layers_to_train 2 3 6 \
		--model_name big_random16M_vocab32_200K \
		--training_config classic \
		--max_train_games $(MAX_TRAIN_GAMES) \
		--test_games_dataset random \
		
# this evaluates probe accuracy on a seperate dataset of games
test_probe: $(TEST)
	$(PYTHON) $(TEST) \
		--mode test \
		--probe piece \
		--probe_dataset random \
		--model_name big_random16M_vocab32_200K \
		--training_config classic \
		--max_train_games $(MAX_TRAIN_GAMES) \
		--test_games_dataset random \
		--verbose

#this can contain multiple models
MODELS := ../models/2_random_600_595K.pth ../models/gm_karvhyp/gm_karvhyp_100K.pth
#training configs include classic, cast16, ect...

# this run experiments script automates probe training and testing
# and does so over all combinations of configurations you specify
# there's also a cool little json file that stores all the combinations that have already been run so they dont run again
# if you ask for them again or if you stop and restart the program
train_all_classic_probes:
	$(PYTHON) run_experiments.py \
		--models $(MODELS) \
		--probe_datasets random \
		--training_configs classic \
		--test_games_datasets random \
		--max_train_games $(MAX_TRAIN_GAMES) \
		--num_epochs 3 \
		--verbose \
		--test

# this is the csv -> bin program from the other repo, not sure you need it but ill leave it here jic
setup: $(SETUP)
	$(PYTHON) $(SETUP) \
		--model_name $(RANDOM_MODEL_NAME)




B1 := some_s3_bucket
#delete your buckets with command line using this command, it's much simpler than on the s3 interface
sups3:
	aws s3 rm s3://$(B1) --recursive
	aws s3 rb s3://$(B1)

#I didnt implement probe training online because the computer is just strong enough that it can do so locally,
# but i would recomend sending them online, this solution is kind of long and complicated
