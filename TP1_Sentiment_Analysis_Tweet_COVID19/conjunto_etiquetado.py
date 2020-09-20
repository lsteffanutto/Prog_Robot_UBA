import pandas as pd
import csv
import numpy as np
import re
# import nltk
# from sklearn.datasets import load_files
# #load_files function divides dataset into data and target sets
# nltk.download('stopwords')
# nltk.download('wordnet')
# import pickle
# from nltk.corpus import stopwords

header_list=['text', 'categorie']
#LECTURE DES DATAS POUR DEAD CA AVEC LA LIB PANDA
data_nrv=pd.read_csv('nrv.csv',names=header_list,encoding='utf-8')
df_nrv=pd.DataFrame(data_nrv)
df_nrv['categorie']='enojo'
df_nrv = df_nrv[['categorie', 'text']]

data_joie=pd.read_csv('joie.csv',names=header_list,encoding='utf-8')
df_joie=pd.DataFrame(data_joie)
df_joie['categorie']='alegria'
df_joie = df_joie[['categorie', 'text']]

data_aide=pd.read_csv('challa.csv',names=header_list,encoding='utf-8')
df_aide=pd.DataFrame(data_aide)
df_aide['categorie']='apoyo'
df_aide = df_aide[['categorie', 'text']]

data_triste=pd.read_csv('tristesse.csv',names=header_list,encoding='utf-8')
df_triste=pd.DataFrame(data_triste)
df_triste['categorie']='tristeza'
df_triste = df_triste[['categorie', 'text']]

# On doit prendre meme nombre de chaque categorie, donc on prend nb ligne de la catégorie ou y'en a le moins
(ligne_nrv,col) = df_nrv.shape
(ligne_joie,col) = df_joie.shape
(ligne_aide,col) = df_aide.shape
(ligne_triste,col) = df_triste.shape
print('tweets enojos =',ligne_nrv)
print('tweets alegria =',ligne_joie)
print('tweets apoyo =',ligne_aide)
print('tweets tristeza =',ligne_triste)

tab_ligne_categories = [ligne_nrv,ligne_joie  ,ligne_triste]
nb_ligne_a_conserver = min(tab_ligne_categories)
#print(tab_ligne_categories)
#print(nb_ligne_a_conserver)

df_nrv=df_nrv[:nb_ligne_a_conserver]
df_joie=df_joie[:nb_ligne_a_conserver]
df_aide=df_aide[:nb_ligne_a_conserver]
df_triste=df_triste[:nb_ligne_a_conserver]

# data_nrv=pd.read_csv('nrv.csv',encoding='utf-8')
# df_nrv=pd.DataFrame(data)
# data_nrv=pd.read_csv('nrv.csv',encoding='utf-8')
# df_nrv=pd.DataFrame(data)

frames = [df_joie,df_nrv,df_aide,df_triste]
df_train_set_balanced = pd.concat(frames)
(nb_tweet_conjunto_etiquado_1cat,cat)=df_joie.shape
(nb_tweet_conjunto_etiquado,cat)=df_train_set_balanced.shape
print('nb tweet 1 categorie =',nb_tweet_conjunto_etiquado_1cat)
print('nb tweet total classifies pour entrainement =',nb_tweet_conjunto_etiquado)
#on dégage les emojis
df_train_set_balanced['text'] = df_train_set_balanced['text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)

# df=df.drop(df.(5).index,inplace=True) # drop first n rows

# df.to_csv('nrv_bis.csv', index=False,header=False, encoding='utf-8')

# df_nrv.to_csv('nrv_bis.csv', index=False,header=False, encoding='utf-8')
# df_joie.to_csv('joie_bis.csv', index=False,header=False, encoding='utf-8')
# df_aide.to_csv('challa_bis.csv', index=False, encoding='utf-8')
# df_triste.to_csv('tristesse_bis.csv', index=False, encoding='utf-8')
df_train_set_balanced.to_csv('conjunto_etiquetado_balanced.csv', index=False, encoding='utf-8')

#une colonnne categorie, une colonne tweet
