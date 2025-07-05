from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import nltk
from ml_pipeline.model_utils import word2features

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

app = FastAPI()
crf = joblib.load("ml_pipeline/crf_model.joblib")

class TextRequest(BaseModel):
    sentence: str

@app.post("/predict")
def predict_entities(data: TextRequest):
    tokens = nltk.word_tokenize(data.sentence)
    pos_tags = nltk.pos_tag(tokens)
    features = [word2features(pos_tags, i) for i in range(len(pos_tags))]
    tags = crf.predict([features])[0]
    return {"entities": list(zip(tokens, tags))}
