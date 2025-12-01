# Deep Learning Portfolio

![CI](https://github.com/ericknyatenya/deep-learning-portfolio/actions/workflows/ci.yml/badge.svg)

A collection of notebooks and simple model implementations for learning deep learning fundamentals.

Repository structure:

```
deep-learning-portfolio/
├── notebooks/            # Educational notebooks
│   ├── 01_neuron_forward_backward.ipynb
│   ├── 02_nn_from_scratch.ipynb
│   ├── 03_cnn_tensorflow.ipynb
│   └── 04_text_classification.ipynb
│
├── src/                  # Source code
│   ├── scratch/          # NumPy scratch implementations (layers, activations)
│   │   ├── layers.py
│   │   └── activations.py
│   ├── tf_models/        # TensorFlow model definitions
│   └── torch_models/     # PyTorch model definitions
│
├── data/
│   └── processed/
│
├── experiments/         # Experiment folders (mnist, cifar10, text)
│
├── models/
│   └── saved/           # Saved weights / checkpoints
│
├── requirements.txt
└── README.md
```

Getting started:

1. Create a virtual environment:
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Open the notebooks with Jupyter:
```bash
jupyter notebook notebooks/
```

3. Explore `src/scratch` to learn the building blocks.

