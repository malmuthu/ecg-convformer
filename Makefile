download:
	python -m src.data.download

preprocess:
	python -m src.data.preprocess

train:
	python -m src.training.trainer --config configs/default.yaml

evaluate:
	python -m src.evaluation.evaluate --checkpoint results/checkpoints/convformer_interpatient_v1.pt

transfer:
	python -m src.evaluation.transfer --checkpoint results/checkpoints/convformer_interpatient_v1.pt --mode both

figures:
	python generate_figures.py
