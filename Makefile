PYTHON := python3
SETUP := model_setup.py
SETUPs3:= model_setup_s3.py
TEST := train_test_chess.py
FILTER := lichess_data_filtering.ipynb
BUCKET := go-bucket-craft
DATATYPE := dummy
LOCAL_PROBES := linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_0.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_1.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_2.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_3.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_4.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_5.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_6.pth linear_probes/saved_probes/tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_7.pth
S3_PROBES := tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_0.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_1.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_2.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_3.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_4.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_5.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_6.pth tf_lens_$(DATATYPE)_8layers_ckpt_no_optimizer_chess_piece_probe_layer_7.pth

B1 := skypilot-workdir-oscar-f41b825a

setup: $(SETUP)
	$(PYTHON) $(SETUP)

test_probe: $(TEST)
	$(PYTHON) $(TEST) \
		--mode test \
		--probe piece

train_probe:
	$(PYTHON) $(TEST)

remote_train_probe:
	sky jobs launch -c boardCluster remote/train_probes.yaml

remote_test_probe:
	sky jobs launch -c boardCluster remote/test_probes.yaml


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
