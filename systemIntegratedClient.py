# Import prediction model and vectorizer
import pickle

with open("model.pkl", 'rb') as file:
    imported_model = pickle.load(file)

with open("vec.pkl", 'rb') as file:
    imported_vec = pickle.load(file)

# Define prediction function
import string # To detect punctuation
import re # To use regular expressions
import nltk # Matural Language Processing Toolkit
nltk.download('stopwords') # Download list of stop words
from nltk.tokenize import TweetTokenizer # Used to tokenize words
from nltk.stem import WordNetLemmatizer # Used for lemmatization
nltk.download('wordnet')# Download lemmatization databse
tw_tknzr=TweetTokenizer(strip_handles=True, reduce_len=True) # Initialize tokenizer
from nltk.corpus import stopwords # Import stop words list

def hatespeech(chat_msg): # Hate speech detection function

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

    return import_pred[0]

# Client
from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread
import tkinter

# Handles receiving of messages
def receive():
    while True:
        try:
            msg = client_socket.recv(BUFSIZ).decode("utf8")
            msg_list.insert(tkinter.END, msg)
        except OSError:  # Possibly client has left the chat.
            break

# Handles sending of messages
def send(event=None):  # event is passed by binders.
    msg = my_msg.get()
    moderated_msg = hatespeech(msg)
    if moderated_msg == 1:
        msg = "MESSAGE REDACTED DUE TO PROFANITY"
    my_msg.set("")  # Clears input field.
    client_socket.send(bytes(msg, "utf8"))
    if msg == "{quit}":
        client_socket.close()
        top.quit()

# This function is to be called when the window is closed
def on_closing(event=None):
    my_msg.set("{quit}")
    send()

# Define GUI window
top = tkinter.Tk()
top.title("Seen There, Done Chat")

messages_frame = tkinter.Frame(top)
my_msg = tkinter.StringVar()  # For the messages to be sent.
my_msg.set("Type your messages here.")
scrollbar = tkinter.Scrollbar(messages_frame)  # To navigate through past messages.
# Following will contain the messages.
msg_list = tkinter.Listbox(messages_frame, height=15, width=50, yscrollcommand=scrollbar.set)
scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
msg_list.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
msg_list.pack()
messages_frame.pack()

entry_field = tkinter.Entry(top, textvariable=my_msg)
entry_field.bind("<Return>", send)
entry_field.pack()
send_button = tkinter.Button(top, text="Send", command=send)
send_button.pack()

top.protocol("WM_DELETE_WINDOW", on_closing)

# Socket coneection setup
#HOST = input('Enter host: ')
#PORT = input('Enter port: ')

HOST = ('127.0.0.1')
PORT = ('33000')

if not PORT:
    PORT = 33000
else:
    PORT = int(PORT)

BUFSIZ = 1024
ADDR = (HOST, PORT)

client_socket = socket(AF_INET, SOCK_STREAM)
client_socket.connect(ADDR)

receive_thread = Thread(target=receive)
receive_thread.start()
tkinter.mainloop()  # Starts GUI execution.
