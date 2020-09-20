import pandas as pd
import csv
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
#load_files function divides dataset into data and target sets
nltk.download('stopwords')
nltk.download('wordnet')
import pickle
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


#########################################################   ON PREPARE LE CLASSIFICATEUR ################################################
data = pd.read_csv("conjunto_etiquetado_balanced.csv")
df=pd.DataFrame(data)

# fig = plt.figure(figsize=(8,6))
# ax1 = fig.add_subplot(111)
# ax1.set_ylabel('cantidad de tweet')
# ax1.set_xlabel('tweet')
# ax1.set_title('conjunto de entrenamiento')
# df.groupby('categorie').text.count().plot.bar(ylim=0)
# # colors = {'1':'blue', '2':'red', '3':'green', '4':'black'}
# # c = df['categorie'].apply(lambda x: colors[x])
# # ax = plt.subplot(111) #specify a subplot
# plt.legend()
# plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.text).toarray()
labels = df.categorie
taille=features.shape
#taille= (552, 223) = les 552 tweet are represented by 223 features, representing the tf-idf score for different unigrams and bigrams.

print(taille)


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['categorie'],test_size=0.33, random_state = 0)
count_vect = CountVectorizer()

#fit le train_set
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

classifier = MultinomialNB().fit(X_train_tfidf, y_train) # CLASSIFICATEUR PRET

# ON VERIFIE LA PRECISION
# print(classifier.predict(count_vect.transform([" I am happy, very happy ."])))
# print(classifier.predict(count_vect.transform([" I am angry, very angry ."])))
# print(classifier.predict(count_vect.transform([" I hope, I am very hopefully ."])))
# print(classifier.predict(count_vect.transform([" I am sad, very sad ."])))
y_pred = classifier.predict(count_vect.transform(X_test)) #verif la qualité de ton classificateur

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print('precision=',accuracy_score(y_test, y_pred))

######################################################################################################################

############################# ON CLASSIFIE L'ENSEMBLE PAS CLASSIFIER #################################################

data_test = pd.read_csv("mix_sans_etiquette.csv") #ON OUVRE L'ENSEMBLE MIX ET ON VA AFFICHER LA CATEGORIE A COTE DE CHACUN DES TWEET
dftest=pd.DataFrame(data_test)
dftest['text'] = dftest
dftest['categorie']=0
dftest = dftest[['categorie', 'text']]
#
dftest=dftest[1:3000000] #on essaye sur petite partie du mix
# print(dftest.head)
(nb_tweet,cat)=dftest.shape #ON COMPTE LE NOMBRE DE TWEET
print('nombre de tweet testé =',nb_tweet)
dfres=dftest
# print(dfres.head)

df_text=dfres['text']
df_cat=dfres['categorie']

for i in range(nb_tweet+1): #ON CLASSIFIE CHAQUE TWEET LIGNE PAR LIGNE
    tab_str=[]
    tab_str=df_text[i:i+1].to_string()
    classe=classifier.predict(count_vect.transform([tab_str]))
    # classe=classe.to_string()
    # classe = classe.str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
    df_cat[i]=classe


dfres['text']=df_text
dfres['categorie']=df_cat

##### AFFICHAGE DU TRI ##############################################

stats = dfres['categorie'].value_counts()
stats_percent = dfres['categorie'].value_counts(normalize=True)

stats_nom=dfres['categorie'].value_counts().index.tolist()
print(stats)
print(stats_percent)

objects = (stats_nom[0], stats_nom[1], stats_nom[2], stats_nom[3])
y_pos = np.arange(len(objects))
performance = [stats[0],stats[1],stats[2],stats[3]]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Nombre de tweet')
plt.title('Catégorie')
plt.show()

print('nb tweet neg =', stats[0]+stats[1])
print('nb tweet pos =', stats[2]+stats[3],'\n')
print('% tweet neg =', stats_percent[0]+stats_percent[1])
print('% tweet pos =', stats_percent[2]+stats_percent[3])
# objects = (stats_nom[1],stats_nom[0])
# y_pos = np.arange(len(objects))
# performance = [stats_percent[0],stats_percent[1],stats_percent[2],stats_percent[3]]
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Nombre de tweet')
# plt.title('Percentage categorie')
# plt.show()
################################################################

dfres.to_csv('res.csv', index=False, encoding='utf-8') #TRI DANS UN CSV
