# ITOps-LLM-Prediction Model Repository

This repository provides a complete machine learning pipeline for anomaly detection in IT operations logs(linux; /var/log/messages
), leveraging a combination of time-series analysis with TS2Vec and natural language processing with a GPT-2 based model. Below is the structure and content of the repository.

## Project Structure

```
itops-llm-prediction/
├── src/
│   ├── data_preprocess.py
│   ├── embedding.py
│   ├── ts2vec_train.py
│   ├── gpt2_predictor.py
│   ├── evaluate.py
│   └── types/
│       └── __init__.py
├── ts2vec/
│   └── (ts2vec library files)
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

-   Python 3.8+
-   PyTorch
-   A CUDA-enabled GPU is recommended for training.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/shiw2/ITOpts-LLM-Prediction-model
    cd ITOpts-LLM-Prediction-model
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To run the full pipeline, you will need to execute the scripts in the following order. Ensure your raw data file (e.g., `messages-20250602`) is placed in a `data/sourcedata/` directory at the root of the project.

1.  **Data Preprocessing**:
    ```bash
    python -m src.data_preprocess
    ```
    This script will load the raw data, perform cleaning and down-sampling, and save the processed training and testing sets to `data/sampledatasets/`.

2.  **Embeddings and TS2Vec Training**:
    ```bash
    python -m src.ts2vec_train
    ```
    This will generate text and label embeddings, train the TS2Vec model, and save the final embedded data to `data/embeddata/`.

3.  **GPT-2 Predictor Training and Evaluation**:
    ```bash
    python -m src.evaluate
    ```
    This script will train the GPT-2 based predictor, evaluate its performance, and compare it with other baseline models (SVM, Random Forest, Decision Tree). The results, including performance metrics and plots, will be displayed.

## Model Performance

The ITOps-LLM-Prediction model demonstrates strong performance in identifying anomalies, outperforming several traditional machine learning models.

| Model                       | Accuracy | Macro Recall | Macro Precision | Macro F1 |
| --------------------------- | -------- | ------------ | --------------- | -------- |
| ITOpts-LLM-Prediction model | 0.9800   | 0.9613       | 0.9757          | 0.9683   |
| SVM                         | 0.9500   | 0.8788       | 0.9651          | 0.9142   |
| RF                          | 0.9140   | 0.7925       | 0.9369          | 0.8408   |
| DT                          | 0.8780   | 0.7662       | 0.8270          | 0.7906   |
