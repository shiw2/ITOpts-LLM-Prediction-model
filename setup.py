from setuptools import setup, find_packages

setup(
    name='itops_llm_prediction_model',
    version='0.1.0',
    author='shiw2',
    description='ITOps-LLM: TS2Vec + GPT2-based prediction model',
    url='https://github.com/shiw2/ITOpts-LLM-Prediction-model',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'torch>=1.8',
        'transformers>=4.0.0',
        'ts2vec @ git+https://github.com/zhihanyue/ts2vec.git',
    ],
    package_data={
        'itops_llm': ['classifier.pkl', 'ts2vec_ckpt/*']
    },
    python_requires='>=3.7',
)
