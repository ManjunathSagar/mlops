# ml_pipeline/train_crf.py
import mlflow
import nltk
import joblib
import pandas as pd
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics
#from ml_pipeline.model_utils import sentence2features, sentence2labels
from model_utils import sentence2features, sentence2labels
#from ml_pipeline.data_prep import load_dataset, split_sentences
from data_prep import load_dataset, split_sentences

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

def train():
    mlflow.set_experiment("CRF-NER")

    with mlflow.start_run():
        sentences = load_dataset("data/ner_dataset.csv")
        train_data, val_data, test_data = split_sentences(sentences)

        X_train = [sentence2features(s) for s in train_data]
        y_train = [sentence2labels(s) for s in train_data]
        X_val = [sentence2features(s) for s in val_data]
        y_val = [sentence2labels(s) for s in val_data]
        X_test = [sentence2features(s) for s in test_data]
        y_test = [sentence2labels(s) for s in test_data]

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X_train, y_train)

        val_f1 = metrics.flat_f1_score(y_val, crf.predict(X_val), average='weighted')
        test_f1 = metrics.flat_f1_score(y_test, crf.predict(X_test), average='weighted')

        mlflow.log_param("algorithm", "lbfgs")
        mlflow.log_param("max_iterations", 100)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("test_f1", test_f1)

        model_path = "ml_pipeline/crf_model.joblib"
        joblib.dump(crf, model_path)
        mlflow.log_artifact(model_path)

        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
