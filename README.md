# NLP Project Part One: Emotion and Empathy Prediction in Conversations

## Introduction

This project explores the application of NLP techniques for the task of emotion and empathy prediction in conversational contexts. Using the WASSA 2024 Track-2 dataset, we analyze multi-turn dialogues to predict emotion intensity, emotion polarity, and empathy levels in conversational exchanges.

## Dataset

The project uses the **WASSA 2024 Track-2** dataset, which contains:
- Multi-turn dialogues annotated with emotion intensity scores
- Emotion polarity scores
- Empathy intensity scores

The dataset is based on participants' empathic reactions to news stories, where multiple readers read the same article and then engage in conversations about it. These dialogues capture how people express emotions and empathy when reacting to situations involving harm to a person, group, or community.

### Dataset Structure
```
Dataset/
├── trac2_CONVT_train.csv
├── trac2_CONVT_dev.csv
└── trac2_CONVT_test.csv
```

## Task Description

Given a conversational turn, the goal is to predict:
1. **Emotion Intensity**: The intensity level of emotions expressed
2. **Emotion Polarity**: The positive or negative orientation of emotions
3. **Empathy Level**: The degree of empathic response

This challenge requires understanding both:
- The semantics of individual utterances
- The broader dynamics of conversations

## Project Structure

```
NLPproject1/
├── Dataset/              # Training, validation, and test datasets (gitignored)
├── Notebooks/            # Jupyter notebooks for experiments
│   └── main_notebook.ipynb
├── Scripts/              # Python modules
│   ├── ann_model.py      # Neural network model definitions
│   ├── dataset.py        # Dataset loading and processing
│   └── preprocessing.py  # Data preprocessing utilities
├── Report/               # Experiment results and reports (gitignored)
├── Results/              # Output files and predictions
├── Saved Models/         # Trained model checkpoints (gitignored)
└── README.md
```

## Approaches

This project implements various NLP techniques, including:

- **Embeddings**: Word and sentence-level representations
- **Recurrent Neural Networks (RNNs)**: For sequential processing of conversational data
- **Transformers**: State-of-the-art architectures for context understanding
- **LLM Prompting**: Leveraging large language models for prediction tasks

The project analyzes the strengths and weaknesses of each approach in the domain of emotion and empathy modeling.

## Models

The repository includes several trained models with different architectures:
- Baseline model with ReLU activation and dropout
- Deep architecture with GELU activation
- Shallow architecture with LeakyReLU and low dropout

## Getting Started

### Prerequisites

```bash
# Python 3.x required
# Install dependencies (create requirements.txt as needed)
pip install torch numpy pandas scikit-learn transformers
```

### Running the Code

1. Ensure the dataset files are placed in the `Dataset/` folder
2. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook Notebooks/main_notebook.ipynb
   ```
3. Or use the Python scripts directly:
   ```bash
   python Scripts/preprocessing.py
   python Scripts/ann_model.py
   ```

## Objectives

This project provides hands-on experience with:
- Working with real-world conversational datasets
- Implementing and comparing multiple NLP architectures
- Understanding emotion and empathy modeling challenges
- Combining representation learning, context modeling, and classification
- Analyzing model performance and trade-offs

## Results

Experimental results and model performance metrics are stored in the `Report/` directory (not tracked in git).

## License

This is an academic project for educational purposes.

## Acknowledgments

- WASSA 2024 Shared Task organizers for the dataset
- Based on research in empathic reactions to news stories
