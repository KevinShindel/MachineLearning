import matplotlib.pyplot as plt
# import the necessary packages
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# constants
NB_CLASSES = 2
N_HIDDEN = 32
RANDOM_STATE = datetime.now().timestamp()
TEST_SIZE = 0.2
BATCH_SIZE = 256
EPOCHS = 10
VERBOSE = 1


def custom_tokenizer(text):
    # create a word lemmatizer
    lemmatizer = WordNetLemmatizer()

    # tokenize the text
    words = word_tokenize(text)

    # remove the stop words
    stopwords_list = stopwords.words('english')
    words = filter(lambda word: word not in stopwords_list, words)

    # lemmatize the words
    lematized = list(map(lemmatizer.lemmatize, words))

    return lematized


def main():
    # load data
    spam_data = pd.read_csv('../files/Spam-Classification.csv',
                            encoding='latin-1')

    # show dtypes
    print(spam_data.dtypes)

    # print describe
    print(spam_data.describe())

    spam_classes_raw = spam_data['CLASS']
    spam_messages = spam_data['SMS']

    # build a TF-IDF vectorizer model

    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)

    # transform feature input to TF-IDF
    tf_idf = vectorizer.fit_transform(spam_messages)

    # convert TF_IDF to numpy array
    tf_idf_array = tf_idf.toarray()

    # build a label encoder for target variable to convert strings to integers
    label_encoder = LabelEncoder()
    spam_classes = label_encoder.fit_transform(spam_classes_raw)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tf_idf_array, spam_classes,
                                                        test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)

    # create model
    model = Sequential()

    # add hidden lvl 1
    model.add(Dense(N_HIDDEN, input_dim=X_train.shape[1], activation='relu', name='hidden1'))

    # add hidden lvl 2
    model.add(Dense(N_HIDDEN, activation='relu', name='hidden2'))

    # add output layer
    model.add(Dense(NB_CLASSES, activation='softmax', name='output'))

    # compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # collect summary of the model
    summary = model.summary()
    print(summary)

    # train model
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    # show history
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.title('Model performance throughout training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    # evaluate model
    model.evaluate(X_test, y_test)

    # predict for text
    text = 'Congratulations! You have been selected as a winner. Text WON to 4422 to claim your prize.'
    text_vector = vectorizer.transform([text]).toarray()

    # predict
    model_prediction = model.predict(text_vector)
    prediction = model_prediction.argmax()
    print(f'The text is a {label_encoder.inverse_transform([prediction])[0]} message')
    print('Prediction Output: ', prediction)
    print('Prediction Classes are: ', label_encoder.inverse_transform(prediction))

if __name__ == '__main__':
    main()