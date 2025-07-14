# ITOps-LLM Prediction Model

This repository contains the implementation of the ITOps-LLM prediction model, which utilizes advanced embedding techniques and transformer models to predict operational issues in IT environments.

## Project Structure

```
itops-llm-prediction
├── src
│   ├── data_preprocess.py       # Functions for loading and preprocessing the dataset
│   ├── embedding.py              # Handles text and label embedding
│   ├── ts2vec_train.py           # Responsible for training the TS2Vec model
│   ├── gpt2_predictor.py         # Defines the GPT2VecPredictor class for predictions
│   ├── evaluate.py               # Functions for evaluating model performance
│   └── types
│       └── __init__.py           # Custom types or interfaces
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/shiw2/ITOpts-LLM-Prediction-model.git
cd ITOpts-LLM-Prediction-model
pip install -r requirements.txt
```

## Dependencies

The project requires the following libraries:

- pandas
- numpy
- torch
- transformers
- sentence-transformers
- ts2vec
- scikit-learn
- matplotlib
- seaborn

## Usage

1. **Data Preprocessing**: Use `data_preprocess.py` to load and preprocess your dataset. This includes loading data from JSON files, normalizing timestamps, and encoding labels.

2. **Embedding**: Utilize `embedding.py` to convert your text and labels into vector representations using the SentenceTransformer model.

3. **Training the Model**: Train the TS2Vec model using `ts2vec_train.py`. This script will fit the model on your training data and encode it into representations.

4. **Making Predictions**: Use the `gpt2_predictor.py` to create an instance of the `GPT2VecPredictor` class, which will allow you to map input data to predictions based on the trained model.

5. **Evaluation**: Evaluate the model's performance using the functions defined in `evaluate.py`. This includes calculating accuracy, precision, recall, and F1-score.

## Model Links

- **TS2Vec**: The TS2Vec model can be found at [ts2vec GitHub repository](https://github.com/zhihanyue/ts2vec).
- **GPT-2**: The GPT-2 model is available on [Hugging Face GPT2](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt2).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
