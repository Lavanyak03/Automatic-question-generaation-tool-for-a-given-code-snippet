from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import simpledialog
from tkinter import filedialog
import string
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.models import model_from_json
import pickle
import os

main = tkinter.Tk()
main.title("Generating Question Titles for Stack Overflow from Mined Code Snippets") #designing main screen
main.geometry("1300x1200")

global filename
global code
global question
global code_text_tokenizer
global code_token, question_token, code_length, question_length
global enc_dec_model
global max_code_len, code_vocab, max_question_len, question_vocab,code_pad_sentence, question_pad_sentence, question_text_tokenizer

def clean_sentence(sentence):
    lower_case_sent = sentence.lower()
    string_punctuation = string.punctuation + "¡" + '¿'
    clean_sentence = lower_case_sent.translate(str.maketrans('', '', string_punctuation))
    return clean_sentence

def tokenize(sentences):
    text_tokenizer = Tokenizer()
    text_tokenizer.fit_on_texts(sentences)
    return text_tokenizer.texts_to_sequences(sentences), text_tokenizer


def readDataset(fileName, dataList):
    with open(fileName, "r") as file:
        for line in file:
            line = line.strip('\n')
            line = line.strip()
            dataList.append(line)
    file.close()
    return dataList

def upload():
    global filename, code, question
    code = []
    question = []
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    text.insert(END,filename+" loaded\n\n")
    code = readDataset("data_sample/src-train.txt", code)
    question = readDataset("data_sample/tgt-train.txt", question)
    text.insert(END,"Total questions found in dataset: "+str(len(question))+"\n")
    text.insert(END,"Total code found in dataset: "+str(len(code))+"\n")

def Preprocessing():
    global max_code_len, code_vocab, max_question_len, question_vocab,code_pad_sentence, question_pad_sentence, question_text_tokenizer
    text.delete('1.0', END)
    global code, question
    global code_text_tokenizer
    global code_token, question_token, code_length, question_length

    code_text_tokenized, code_text_tokenizer = tokenize(code)
    question_text_tokenized, question_text_tokenizer = tokenize(question)
    code_token = len(max(code_text_tokenized,key=len))
    question_token = len(max(question_text_tokenized,key=len))

    code_vocab = len(code_text_tokenizer.word_index) + 1
    question_vocab = len(question_text_tokenizer.word_index) + 1

    max_code_len = int(len(max(code_text_tokenized,key=len)))
    max_question_len = int(len(max(question_text_tokenized,key=len)))
    code_length = max_code_len
    question_length = max_question_len

    code_pad_sentence = pad_sequences(code_text_tokenized, max_code_len, padding = "post")
    question_pad_sentence = pad_sequences(question_text_tokenized, max_question_len, padding = "post")
    question_pad_sentence = question_pad_sentence.reshape(*question_pad_sentence.shape, 1)
    print("code tokens ==> ",str(code_token))
    print("Question Tokens ==>",str(question_token))
    print("Average Code Length ==>",str(max_code_len/100))
    print("Average Question Length ==>",str(max_question_len/100))
    print("Code Vocabulary ==>",str(code_vocab))
    print("Question Vocabulary ==>",str(question_vocab))
    text.insert(END,"Code Tokens             : "+str(code_token)+"\n")
    text.insert(END,"Question Tokens         : "+str(question_token)+"\n")
    text.insert(END,"Average Code Length     : "+str(max_code_len/100)+"\n")
    text.insert(END,"Average Question Length : "+str(max_question_len/100)+"\n\n")
    text.insert(END,"Code Vocabulary: "+str(code_vocab)+"\n\n")
    text.insert(END,"Question Vocabulary: "+str(question_vocab))


def trainLSTM():
    global max_code_len, code_vocab, max_question_len, question_vocab,code_pad_sentence, question_pad_sentence, question_text_tokenizer
    text.delete('1.0', END)
    global enc_dec_model
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            enc_dec_model = model_from_json(loaded_model_json)
        json_file.close()
        enc_dec_model.load_weights("model/model_weights.h5")
        enc_dec_model._make_predict_function()
    else:
        input_sequence = Input(shape=(max_code_len,))
        embedding = Embedding(input_dim=code_vocab, output_dim=128,)(input_sequence)
        encoder = LSTM(32, return_sequences=False)(embedding)
        r_vec = RepeatVector(max_question_len)(encoder)
        decoder = LSTM(32, return_sequences=True, dropout=0.2)(r_vec)
        logits = TimeDistributed(Dense(question_vocab))(decoder)
        enc_dec_model = Model(input_sequence, Activation('softmax')(logits))
        enc_dec_model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(1e-3), metrics=['accuracy'])
        enc_dec_model.summary()
        hist = enc_dec_model.fit(code_pad_sentence, question_pad_sentence, batch_size=8, epochs=5000)
        enc_dec_model.save_weights('model/model_weights.h5')
        model_json = enc_dec_model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(enc_dec_model.summary())
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[4999] * 100
    text.insert(END,"LSTM training Process Completed with Final Accuracy: "+str(accuracy)


def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['accuracy']
    loss = data['loss']
    accuracy = accuracy[4500:4999]
    loss = loss[4500:4999]

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('LSTM Sequence 2 Sequence Encoder & Decoder Accuracy & Loss Graph')
    plt.show()


def tokenGraph():
    global code_token, question_token, code_length, question_length

    height = [code_token, question_token, code_length, question_length]
    bars = ('Code Tokens', 'Question Tokens', 'Average Code Length', 'Average Question Length')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Tokens & Words Length Graph")
    plt.show()

def predictQuestion(logits, tokenizer):
    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = ''
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


def generateQuestion():
    text.delete('1.0', END)
    global enc_dec_model, code_text_tokenizer, max_code_len, question_text_tokenizer
    codeData = tf1.get()
    if len(codeData) > 0:
        testCode_tokenize = code_text_tokenizer.texts_to_sequences([codeData])
        testCode_tokenize = pad_sequences(testCode_tokenize, max_code_len, padding = "post")
        predict_question = predictQuestion(enc_dec_model.predict(testCode_tokenize)[0], question_text_tokenizer)
        print(testCode_tokenize)
        text.insert(END,"Generated question ====> "+str(predict_question))


font = ('times', 16, 'bold')
title = Label(main, text='Generating Question Titles for Stack Overflow from Mined Code Snippets')
title.config(bg='LightGoldenrod1', fg='medium orchid')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Code Question Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

preButton = Button(main, text="Dataset Preprocessing", command=Preprocessing)
preButton.place(x=250,y=100)
preButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Encoding-Decoding Algorithm", command=trainLSTM)
lstmButton.place(x=450,y=100)
lstmButton.config(font=font1)

graphButton = Button(main, text="LSTM Accuracy-Loss Graph", command=graph)
graphButton.place(x=780,y=100)
graphButton.config(font=font1)

tokensButton = Button(main, text="Code & Question Tokens Graph", command=tokenGraph)
tokensButton.place(x=1030,y=100)
tokensButton.config(font=font1)

l1 = Label(main, text='Input Code:')
l1.config(font=font1)
l1.place(x=50,y=150)

tf1 = Entry(main,width=90)
tf1.config(font=font1)
tf1.place(x=180,y=150)

generateButton = Button(main, text="Generate Question from Mined Code", command=generateQuestion)
generateButton.place(x=920,y=150)
generateButton.config(font=font1)




main.config(bg='OliveDrab2')
main.mainloop()
