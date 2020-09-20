#D'abord tu prend ton ensemble conjunto_etiquetado_balanced et tu change les categories


import pandas as pd
import csv
import numpy as np
import re
import random

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

data = pd.read_csv("conjunto_etiquetado_balanced.csv")
df_data_SW=pd.DataFrame(data)


df_data_SW['categorie'] = df_data_SW['categorie'].str.replace('alegria', '1')
df_data_SW['categorie'] = df_data_SW['categorie'].str.replace('apoyo', '1')
df_data_SW['categorie'] = df_data_SW['categorie'].str.replace('tristeza', '0')
df_data_SW['categorie'] = df_data_SW['categorie'].str.replace('enojo', '0')

#ON FABRIQUE conjunto_etiquetado_balanced pour SWN
df_data_SW.to_csv('conjunto_etiquetado_balanced_train_SWN.csv', index=False) #conjunto_etiquetado_balanced

data_train = pd.read_csv("conjunto_etiquetado_balanced_train_SWN.csv", encoding='utf-8')

data_test = pd.read_csv("mix_sans_etiquette.csv", encoding='utf-8')
total_tweet = 2999000
data_test = data_test[:total_tweet]
dfres=pd.DataFrame(data_test)
#
data_test["text"]=data_test
data_test['categorie']=0
data_test = data_test[['categorie', 'text']]

# dftest=dftest[1:10]
(nb_tweet,col)=data_train.shape
# print(data_train.shape)
# print(data_train["text"][0])
# print(data_train["categorie"][0])

# data_train= list(data_train)

################################ SWN method #####################################################

sentiment_data = list(zip(data_train["text"], data_train["categorie"]))
random.shuffle(sentiment_data)

# 80% for training
nb_training=int(0.67*nb_tweet)
train_X, train_y = zip(*sentiment_data[:nb_training])

# Keep 20% for testing
nb_test=int(0.33*nb_tweet)
test_X, test_y = zip(*sentiment_data[:nb_test])

# print(train_X[0])

lemmatizer = WordNetLemmatizer()


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def clean_text(text):
    text = text.replace("<br />", " ")
    # text = text.decode("utf-8")

    return text


def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """

    sentiment = 0.0
    tokens_count = 0

    # text = clean_text(text)


    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))

        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue

            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())

            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1

    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 0

    # sum greater than 0 => positive sentiment
    if sentiment >= 0:
        return 1

    # negative sentiment
    return 0


# Since we're shuffling, you'll get diffrent results
# print(swn_polarity(train_X[0]), swn_polarity(str(train_y[0])))
# print(swn_polarity(train_X[1]), swn_polarity(str(train_y[1])))
# print(swn_polarity(train_X[2]), swn_polarity(str(train_y[2])))
# print(swn_polarity(train_X[3]), swn_polarity(str(train_y[3])))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred_y = [swn_polarity(text) for text in test_X]

print(confusion_matrix(test_y,pred_y))
print(classification_report(test_y,pred_y))
print('precision=',accuracy_score(test_y, pred_y)) # 0.54

######################################################################################################

(nb_tweet_test,col)=data_test.shape
print('nombre de tweet testé =',nb_tweet_test)

nb_tweet_test=total_tweet

sentiment_data_test = list(zip(data_test["text"]))
test = zip(*sentiment_data_test[:nb_tweet_test])
tab = list(test)
out = str([item for t in tab for item in t])

tweet_test = [x.replace('\n', '') for x in out]


# print(tweet_test[0])
# print(tweet_test[1])
#
# dfres=dftest
# # print(dfres.head)
#
# # df_text=dfres['text']
# df_cat=dfres['categorie']
dfres=data_test
df_text=dfres['text']
df_cat=dfres['categorie']

for i in range(nb_tweet_test):

    # if (swn_polarity(tweet_test[i])==1):
    df_cat[i]=swn_polarity(tweet_test[i])
    # else:
    #     df_cat[i]='negative'
    # print(swn_polarity(tweet_test[i]) , tweet_test[i])

dfres['text']=df_text
dfres['categorie']=df_cat
dfres.to_csv('resSWN.csv', index=False, encoding='utf-8') #TRI DANS UN CSV

##### AFFICHAGE DU TRI ##############################################

stats = dfres['categorie'].value_counts()
stats_percent = dfres['categorie'].value_counts(normalize=True)

stats_nom=dfres['categorie'].value_counts().index.tolist()
print(stats)
print(stats_percent)

#
objects = (stats_nom[1],stats_nom[0])
y_pos = np.arange(len(objects))
performance = [stats[0],stats[1]]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Nombre de tweet')
plt.title('Nombre pos/neg')
plt.show()

# objects = (stats_nom[1],stats_nom[0])
# y_pos = np.arange(len(objects))
# performance = [stats_percent[0],stats_percent[1]]
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Nombre de tweet')
# plt.title('Percentage pos/neg')
# plt.show()

#percent
# print('tweets positive = ')
# print('tweets negatives = ')
# print(swn_polarity(tweet_test[0]) , tweet_test[0])
# print(swn_polarity(tweet_test[1])
# tweet_test[1])
