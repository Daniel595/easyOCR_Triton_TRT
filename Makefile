build_container:
	docker images easyocr_tritonserver:latest | grep easyocr_tritonserver || docker compose build tritonserver

pull_models:
	wget https://github.com/Daniel595/easyOCR_Triton_TRT/releases/download/0.0.1/craft.onnx
	mv craft.onnx triton-server/model_repo/ocr_craft/1
	wget https://github.com/Daniel595/easyOCR_Triton_TRT/releases/download/0.0.1/crnn.onnx
	mv crnn.onnx triton-server/model_repo/ocr_crnn/1

build_models:
	docker run -it --gpus all --shm-size=512m --rm -v $(MODEL_DIR):/models easyocr_tritonserver \
		bash -c "\
			cd /models/ocr_craft/1 && bash trtexec.sh && \
			cd /models/ocr_crnn/1 && bash trtexec.sh
		"

setup_model_repo:
	make build_container
	make pull_models
	make build_models
