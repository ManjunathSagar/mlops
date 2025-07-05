#ml_pipeline/model_utils.py

def word2features(sent, i):
    word, postag = sent[i][0], sent[i][1]
    word = str(word) if not isinstance(word, str) else word

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.istitle()': word.istitle(),
        'word.isupper()': word.isupper(),
        'word.isdigit()': word.isdigit(),
        'postag': postag
    }
    if i > 0:
        word1, postag1 = sent[i-1][0], sent[i-1][1]
        word1 = str(word1) if not isinstance(word1, str) else word1 
        features.update({'-1:word.lower()': word1.lower(), '-1:postag': postag1})
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word1, postag1 = sent[i+1][0], sent[i+1][1]
        word1 = str(word1) if not isinstance(word1, str) else word1 
        features.update({'+1:word.lower()': word1.lower(), '+1:postag': postag1})
    else:
        features['EOS'] = True
    return features

def sentence2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sentence2labels(sent):
    return [label for _, _, label in sent]
