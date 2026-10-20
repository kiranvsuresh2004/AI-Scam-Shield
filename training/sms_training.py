from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / 'dataset' / 'SMSSpamCollection.csv'
MODEL_DIR = PROJECT_DIR / 'model'


def load_sms_dataset():
    raw_data = pd.read_csv(
        DATASET_PATH,
        header=None,
        names=['combined'],
        dtype=str,
        keep_default_na=False,
    )
    parts = raw_data['combined'].str.split('\t', n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError('SMS dataset rows must contain a label and message separated by a tab.')

    data = parts.copy()
    data.columns = ['label', 'message']
    data['label'] = data['label'].str.strip().str.strip('"').str.lower()
    data['message'] = data['message'].str.strip().str.strip('"')
    invalid_labels = set(data['label'].unique()) - {'ham', 'spam'}
    if invalid_labels:
        raise ValueError(f'Unexpected SMS dataset labels: {sorted(invalid_labels)}')

    data['label'] = data['label'].map({'ham': 0, 'spam': 1}).astype(int)
    data['dedupe_key'] = (
        data['message'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    )
    conflicting = data.groupby('dedupe_key')['label'].nunique().gt(1)
    if conflicting.any():
        raise ValueError('Duplicate SMS text has conflicting labels; resolve those rows before training.')
    data = data.drop_duplicates(subset='dedupe_key').drop(columns='dedupe_key')
    data = data[data['message'].str.strip().ne('')]
    return data.reset_index(drop=True)


def report_model(name, model, X_test, y_test):
    prediction = model.predict(X_test)
    print(f'\n{name} classification report:')
    print(classification_report(
        y_test,
        prediction,
        labels=[0, 1],
        target_names=['Genuine', 'Spam'],
        digits=4,
        zero_division=0,
    ))
    print(f'{name} confusion matrix (Genuine, Spam):')
    print(confusion_matrix(y_test, prediction, labels=[0, 1]))


def main():
    data = load_sms_dataset()
    print(f'SMS training rows after deduplication: {len(data)}')
    print(f'Label counts (0=Genuine, 1=Spam): {data["label"].value_counts().sort_index().to_dict()}')

    X_train, X_test, y_train, y_test = train_test_split(
        data['message'],
        data['label'],
        test_size=0.30,
        random_state=42,
        stratify=data['label'],
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        strip_accents='unicode',
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f'TF-IDF training shape: {X_train_tfidf.shape}')
    print(f'TF-IDF test shape: {X_test_tfidf.shape}')

    models = {
        'Naive Bayes': (
            MultinomialNB(alpha=2.0),
            'sms_nb_model.pkl',
        ),
        'Logistic Regression': (
            LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced'),
            'sms_lr_model.pkl',
        ),
        'Linear SVM': (
            LinearSVC(C=0.1, class_weight='balanced'),
            'sms_svm_model.pkl',
        ),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, (model, file_name) in models.items():
        model.fit(X_train_tfidf, y_train)
        report_model(name, model, X_test_tfidf, y_test)
        joblib.dump(model, MODEL_DIR / file_name)

    joblib.dump(vectorizer, MODEL_DIR / 'sms_vectorizer.pkl')
    print(f'\nSaved SMS model artifacts to {MODEL_DIR}')


if __name__ == '__main__':
    main()
