#%% Import libraries

import pandas as pd # Data manipulation
import string # To detect punctuation
import re # To use regular expressions
import nltk # Matural Language Processing Toolkit
nltk.download('stopwords') # Download list of stop words
from nltk.tokenize import TweetTokenizer # Used to tokenize words
from nltk.stem import WordNetLemmatizer # Used for lemmatization
nltk.download('wordnet')# Download lemmatization databse
tw_tknzr=TweetTokenizer(strip_handles=True, reduce_len=True) # Initialize tokenizer
from nltk.corpus import stopwords # Import stop words list
from sklearn.feature_extraction.text import TfidfVectorizer # Import TFIDF vectorizer
from sklearn.feature_extraction.text import CountVectorizer # Import BoW vectorizer
from sklearn.model_selection import train_test_split # Split into training and testing
from sklearn import svm #Support vector machine classification model
#from sklearn.ensemble import RandomForestClassifier # Random forest classification model
#from sklearn.naive_bayes import GaussianNB # Naive Bayes classification model
from sklearn.metrics import confusion_matrix, accuracy_score # To measure model performance
import pickle # Saving and loading model

#%% Import dataset

df=pd.read_excel('Dataset.xlsx')
#df # View dataset

#%% Cleaning

corpus = [] # Initialize corpus as empty list
for i in range(0, len(df)): # Iterate over enter dataset
    df_original=re.sub(r'\$\w*','',df['text'][i]) # Remove tickers (twitter username)
    tw_tknzr=TweetTokenizer(strip_handles=True, reduce_len=True)
    df_tokenized = tw_tknzr.tokenize(df_original)
    df_stopwords=[i for i in df_tokenized if i.lower() not in set(stopwords.words('english'))] # Remove stopwords
    df_hyperlinks=[re.sub(r'https?:\/\/.*\/\w*','',i) for i in df_stopwords] # Remove hyperlinks
    df_hashtags=[re.sub(r'#', '', i) for i in df_hyperlinks] # Remove hashtags
    df_punctuation=[re.sub(r'['+string.punctuation+']+', ' ', i) for i in df_hashtags] # Remove Punctuation and split 's, 't, 've with a space for filter
    df_whitespace = ' '.join(df_punctuation) # Remove multiple whitespace
    lemmatizer = WordNetLemmatizer()
    df_lemma = lemmatizer.lemmatize(df_whitespace) # Lemmatize
    df_lemma_tokenized = tw_tknzr.tokenize(df_lemma) # Remove any words with 2 or fewer letters (after removing punctuation)
    df_lemma_shortwords = [re.sub(r'^\w\w?$', '', i) for i in df_lemma_tokenized]
    df_lemma_whitespace =' '.join(df_lemma_shortwords)
    df_lemma_multiplewhitespace = re.sub(r'\s\s+', ' ', df_lemma_whitespace)
    df_clean = df_lemma_multiplewhitespace.lstrip(' ') #Remove any whitespace at the front of the sentence
    corpus.append(df_clean)
print("Data cleaned!")

#corpus #View cleaned dataset

#%% Model creation (Support Vector Machine with TF-IDF)

mfeatures = 5000 # Number of features
tsize = 0.2 # Percentage size of training set
Regularization = 1.0 # Regularization parameter, must be positive
svmkernel = 'rbf' # SVM Kernel
polydegree = 3 # Degree for polynomial kernel
kernelcoeff = 'scale' #Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. Either 'auto', 'scale', or a float
kernelterm = 0.0 # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

vec = TfidfVectorizer(max_features=mfeatures) # Initialize TFIDF vectorizer
#vec = CountVectorizer(max_features = mfeatures) # Initialize BoW vectorizer

X = vec.fit_transform(corpus).toarray() # Transform cleaned tweets using TF-IDF
y = df.iloc[:, 0].values # Set target as first column of dataset
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = tsize) # Splitting the dataset into the Training set and Test set


#classifier = RandomForestClassifier(max_depth=2, random_state=0) # Initialize Random forest classifier
#classifier = GaussianNB() # Initialize Naive Bayes classifier
classifier = svm.SVC(C=Regularization , kernel=svmkernel, degree=polydegree, gamma=kernelcoeff, coef0=kernelterm, probability=False) # Initialize SVM classifier
classifier.fit(X_train, y_train) # Train classifier with the Training set

y_pred = classifier.predict(X_test) # Predicting with testing set

cm = confusion_matrix(y_test, y_pred) # Generate Confusion Matrix
print(cm)

acc = accuracy_score(y_test, y_pred) # Calculate Accuracy
print(acc)
#%% Saving model

with open("model.pkl", 'wb') as file: # Save model to local disk
    pickle.dump(classifier, file)

with open("vec.pkl", 'wb') as file:  # Save vectorizer to local disk
    pickle.dump(vec, file)

#%% Loading Model

with open("model.pkl", 'rb') as file: # Load model from local disk
    imported_model = pickle.load(file)

with open("vec.pkl", 'rb') as file: # Load model from local disk
    imported_vec = pickle.load(file)

#%% Test with sample message

def hatespeech(chat_msg):
    df_original=re.sub(r'\$\w*','',chat_msg) # Remove tickers (twitter username)
    tw_tknzr=TweetTokenizer(strip_handles=True, reduce_len=True)
    df_tokenized = tw_tknzr.tokenize(df_original)
    df_stopwords=[i for i in df_tokenized if i.lower() not in set(stopwords.words('english'))] # Remove stopwords
    df_hyperlinks=[re.sub(r'https?:\/\/.*\/\w*','',i) for i in df_stopwords] # Remove hyperlinks
    df_hashtags=[re.sub(r'#', '', i) for i in df_hyperlinks] # Remove hashtags
    df_punctuation=[re.sub(r'['+string.punctuation+']+', ' ', i) for i in df_hashtags] # Remove Punctuation and split 's, 't, 've with a space for filter
    df_whitespace = ' '.join(df_punctuation) # Remove multiple whitespace
    lemmatizer = WordNetLemmatizer()
    df_lemma = lemmatizer.lemmatize(df_whitespace) # Lemmatize
    df_lemma_tokenized = tw_tknzr.tokenize(df_lemma) # Remove any words with 2 or fewer letters (after removing punctuation)
    df_lemma_shortwords = [re.sub(r'^\w\w?$', '', i) for i in df_lemma_tokenized]
    df_lemma_whitespace =' '.join(df_lemma_shortwords)
    df_lemma_multiplewhitespace = re.sub(r'\s\s+', ' ', df_lemma_whitespace)
    df_clean = df_lemma_multiplewhitespace.lstrip(' ') #Remove any whitespace at the front of the sentence

    c=[] # Initialize temporary empty list
    c.append(df_clean) # Append cleaned user input

    final_msg= imported_vec.transform(c).toarray() # Vectorize input
    import_pred = imported_model.predict(final_msg) # Predict class of input
    if import_pred[0] == 1:
        print("HATE SPEECH DETECTED")
    else:
        print("NOT HATE ")
#%%

chat_msg = "Chocolate sauce on pizza tastes good" # Test user input, change this

hatespeech(chat_msg) # Function call