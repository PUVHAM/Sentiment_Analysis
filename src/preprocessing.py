import re
import string
import threading
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import contractions

from sklearn.preprocessing import LabelEncoder

stop = set(stopwords.words('english'))

# Expanding constractions
def expand_contractions(text):
    return contractions.fix(text)

# Function to clean data
def preprocess_text(text):
    
    wl = WordNetLemmatizer()
    
    soup = BeautifulSoup(text, "html.parser") # Removing html tags
    text = soup.get_text()
    
    emoji_clean = re.compile("["
        u"\U0001F600-\U0001F64F" # Emoticons
        u"\U0001F300-\U0001F5FF" # Symbols & pictographs
        u"\U0001F680-\U0001F6FF" # Transport & map symbols
        u"\U0001F1E0-\U0001F1FF" # Flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)
    
    text = emoji_clean.sub(r'', text)
    text = re.sub(r'\.(?=\S)', '. ', text) # Add space after full stop
    text = re.sub(r'http\S+', '', text) # Remove urls
    text = "".join([
        word.lower() for word in text if word not in string.punctuation
    ]) # Remove punctuation and make text lowercase
    text = " ".join([
        wl.lemmatize(word) for word in text.split() if word not in stop and word.isalpha()
    ]) # Lemmatize
    
    return text

nltk_lock = threading.Lock()

def safe_preprocess(text):
    with nltk_lock:
        return preprocess_text(text)

def run_preprocess(df):
    x_data = df['review'].apply(safe_preprocess)
    
    label_encode = LabelEncoder()
    y_data = label_encode.fit_transform(df['sentiment'])
    
    return x_data, y_data