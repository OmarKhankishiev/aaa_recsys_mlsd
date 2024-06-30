IMAGE:=homework-pipeline-khankishiev

build:
	docker build -t ${IMAGE} .

run:
	docker run -it ${IMAGE} python script.py

run_2:
	docker run -it ${IMAGE} python script.py --model_name LFM --optimizer_name SGD

run_all:
	docker run -it ${IMAGE} python script.py --model_name baseline_LFM
	docker run -it ${IMAGE} python script.py --model_name baseline_top_popular
	docker run -it ${IMAGE} python script.py --model_name LFM
	docker run -it ${IMAGE} python script.py --model_name ALS
	docker run -it ${IMAGE} python script.py --model_name optuna_LFM_search
	docker run -it ${IMAGE} python script.py --model_name optuna_ALS_search
	docker run -it ${IMAGE} python script.py --model_name best_model
