import time
import streamlit as st
from tqdm import tqdm
from src.preprocessing import run_preprocess
from src.load_dataset import load_df, split_dataset
from src.config import DatasetConfig
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed

class SentimentModel:
    def __init__(self, model_type='dt'):
        self.models = {
            'dt': DecisionTreeClassifier(criterion='entropy', random_state=42, ccp_alpha=0.0),
            'rf': RandomForestClassifier(random_state=42, min_samples_leaf=1, max_features='sqrt'),
            'ada': AdaBoostClassifier(random_state=42, learning_rate=1.0),
            'gb': GradientBoostingClassifier(random_state=42, learning_rate=0.4)
        }
        self.model = self.models.get(model_type)
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        
    def train(self, x_train, y_train):
        print('Start training...')
        x_train_encoded = self.tfidf_vectorizer.fit_transform(x_train)
        self.model.fit(x_train_encoded, y_train)
        print('Training completed!')
        
    def evaluate(self, x_test, y_test):
        x_test_encoded = self.tfidf_vectorizer.transform(x_test)
        y_pred = self.model.predict(x_test_encoded)
        accuracy = accuracy_score(y_pred, y_test)
        
        return accuracy
    
    def predict(self, text):
        text_encoded = self.tfidf_vectorizer.transform([text])
        prediction = self.model.predict(text_encoded)
        return "Positive" if prediction[0] == 1 else "Negative"
    
@st.cache_data(max_entries=1000, ttl=3600)
def train_and_cache_models(model_type):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = load_df(DatasetConfig.DATASET_PATH)
            x_data, y_data = run_preprocess(df)
            x_train, y_train, x_test, y_test = split_dataset(x_data, y_data)

            model = SentimentModel(model_type)
            model.train(x_train, y_train)
            accuracy = model.evaluate(x_test, y_test)
            
            results = {
                'model': model.model.__class__.__name__,
                'accuracy': accuracy
            }
            return model, results
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed. Error: {str(e)}. Retrying...")
                time.sleep(2)  # Đợi 2 giây trước khi thử lại
            else:
                raise

@st.cache_data(max_entries=1000)
def train_models_parallel(model_types):
    models = {}
    results = {}

    with ThreadPoolExecutor() as executor:
        future_to_model = {
            executor.submit(train_and_cache_models, model_type): model_type for model_type in model_types
        }

        for future in tqdm(as_completed(future_to_model), total=len(model_types), desc="Training models"):
            model_type = future_to_model[future]
            model, result = future.result()
            models[model_type] = model
            results[model_type] = result

    return models, results