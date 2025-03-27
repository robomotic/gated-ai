# Makefile for Neural Network Architecture Comparison

.PHONY: setup train train-iwmn test test-iwmn compare clean

# Setup the environment
setup:
	bash setup.sh

# Train the MoE model
train:
	python src/train.py

# Train the IWMN model
train-iwmn:
	python src/train_iwmn.py

# Test the MoE model
test:
	python src/test.py

# Test the IWMN model
test-iwmn:
	python src/test.py --model iwmn

# Run performance comparison between MoE and IWMN
compare:
	python src/comparison.py

# Clean generated files
clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf model_checkpoints/*
	rm -rf data/*
	rm -rf comparison_results/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
