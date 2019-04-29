# helper functions

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

import tensorflow.keras as keras
from tensorflow.keras import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import re

from sklearn.model_selection import train_test_split

def load_texts(PATH_PREFIX="./data/"):
    all_texts_paths = [[PATH_PREFIX+"Murakami_Wood.txt", 11],
                       [PATH_PREFIX+"Murakami_Colorless.txt", 19],
                       [PATH_PREFIX+"Murakami_Wonderland.txt", 40],
                       [PATH_PREFIX+"Abe_Sakura.txt", 25],
                       [PATH_PREFIX+"Abe_Woman.txt", 31],
                       [PATH_PREFIX+"Kafka_Metamorphosis.txt", 3],
                       [PATH_PREFIX+"Kafka_Trial.txt", 10],
                       [PATH_PREFIX+"Soseki_Botchan.txt", 11],
                       [PATH_PREFIX+"Soseki_Cat.txt", 3],
                       [PATH_PREFIX+"Soseki_Kokoro.txt", 11],
                       [PATH_PREFIX+"Yoshimoto_Kitchen.txt", 3],
                       [PATH_PREFIX+"Yoshimoto_Lake.txt", 3]]

    print("Number of Texts:", len(all_texts_paths))

    author, text_len = [ ], [ ]

    for book in all_texts_paths:
        book_path, chapter_count = book[0], book[1]
        book_raw = [line.rstrip("\n").strip() for line in open(book_path)]
        text = ""
        for line in book_raw:
            text = text + line
        text_len.append(len(text))
        book_name = book_path.split("/")[-1]
        author_name = book_name.split("_")[0]
        author.append(author_name)
        
    print("Authors:", set(author))
    
    return all_texts_paths
    
def export_text_sections(path_to_text, sentences_per_section):
    """
    Convert a book from plain text into sections
    """
    
    book_path = str(path_to_text)
    book_raw = [line.rstrip("\n").strip() for line in open(book_path)]

    book_sentences = ""

    for section in book_raw:
        book_sentences = book_sentences + " " + section.strip()
        
    book_sentences = book_sentences.replace("\'", " ' ").replace('“', ' " ')
    book_sentences = book_sentences.replace(",", " , ").replace(";", " ; ")
    book_sentences = book_sentences.replace("’", " ' ").replace('”', ' " ')
    book_sentences = book_sentences.replace(".", " . ").replace('"', ' " ')
    book_sentences = book_sentences.replace("!", " ! ").replace("?", " ? ")
    book_sentences = book_sentences.replace("-", " - ").replace("—", " - ")
    book_sentences = book_sentences.replace(":", " : ").replace("  ", " ")
    book_sentences = book_sentences.replace("  ", " ").replace("  ", " ")
    
    book_sentences = book_sentences.lower()
    
    name_list = ["naoko", "nakano", "watanabe", "toru", "kizuki", "midori",
                 "reiko", "tsukuru", "tazaki", "jumpei", "niki", "karl",
                 "gregor", "samsa", "charwoman", "josef", "frau", "grubach",
                 "botchan", "kiyo", "uranari", "yama", "arashi", "nodaiko",
                 "tanuki", "k", "sneaze", "waverhouse", "goldfield",
                 "sensei", "ojosan", "okusan", "mikage", "sakurai",
                 "yuichi", "tanabe", "eriko", "sotaro", "okuno", "nori",
                 "kuri", "chika", "chihiro", "nakajima", "mino", "chii"]
    
    for name in name_list:
        book_sentences = book_sentences.replace(" "+name+" ", " <NAME> ")
    
    book_sentences = re.sub("\d", " <NUM> ", book_sentences)
    
    book_sentences = book_sentences.replace("  ", " ").split(" . ")

    book_section = [ ]

    count = 0

    for sentence in book_sentences:
        sentence = sentence.strip()
        if count==0:
            chapter = ""
            chapter = chapter + " . " + sentence
            count += 1
        elif count==sentences_per_section-1:
            chapter = chapter + " " + sentence
            book_section.append(chapter)
            chapter = [ ]
            count = 0
        else:
            chapter = chapter + " . " + sentence
            count += 1

    return book_section

def plot_distribution(data, labels):
    # distribution of text lengths
    lengths = np.array([len(row.strip().split(" ")) for row in data])
    summary = "mean: "+str(int(np.mean(lengths)))+" , min/max: "+str(np.min(lengths))+"/"+str(np.max(lengths))+" (95%: "+ str(round(np.percentile(lengths, 95), 2)) + ")"
    plt.figure(1, figsize=(10,4))
    plt.hist(lengths, bins='auto')
    plt.title("Distribution of text lengths")
    plt.xlabel("Text Length: " + summary); plt.ylabel("Examples")
    plt.axvline(np.mean(lengths), ls="-", color="k")
    plt.axvline(np.percentile(lengths, 95), ls="--", color="k")
    plt.xlim(0, int(np.percentile(lengths, 99)))
    plt.show()
    
    # distribution of label counts
    labels = [str(label) for label in labels]
    plt.figure(2, figsize=(10,4))
    plt.hist(labels, bins='auto')
    plt.title("Distribution of labels")
    plt.xlabel("Labels"); plt.ylabel("Examples")
    plt.show()
    
def sequence_to_text(list_of_indices, reverse_word_map):
    """
    Takes a tokenized sentence and returns the words
    """
    list_of_indices = list_of_indices.tolist()
    result = [reverse_word_map.get(letter) for letter in list_of_indices]
    return result
    
def export_html(result, max_activation, model_correct):
    output = "<p>"
    
    for line in result:
        word, activation = line
        if word==None:
            pass
        else:
            if model_correct==False:
                activation=0

            if activation>0:
                activation = activation/(max_activation+1e-7)
                colour = str(int(255 - activation*255))
                tag_open = "<span style='background-color: rgb(255,"+colour+","+colour+");'>"

            else:
                activation = -1 * activation/(max_activation+1e-7)
                colour = str(int(255 - activation*255))
                tag_open = "<span style='background-color: rgb("+colour+","+colour+",255);'>"

            tag_close = "</span>"
            tag = " ".join([str(tag_open), str(word), str(tag_close)])

            output = output + tag
        
    output = output + "</p>"
    
    return output

def test_and_export_html(intermediate_layer_model, model, reverse_word_map, test_data_og, test_label):
    test_data = np.asarray(test_data_og)
    test_data = np.reshape(test_data, (1, test_data.shape[0]))
    intermediate_output = intermediate_layer_model.predict(test_data)
    y_pred = model.predict(test_data)[0]
    y_pred = np.argmax(y_pred,axis=0)
    
    y_truth = np.argmax(test_label,axis=0)
    
    if y_truth==y_pred:
        model_correct = True
    else:
        model_correct = False
    
    activations_list = intermediate_output.tolist()[0]
    max_activation = max(activations_list)

    result = zip(sequence_to_text(test_data_og, reverse_word_map), activations_list)
    output = export_html(result, max_activation, model_correct)
    
    return output