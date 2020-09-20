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

#LECTURE DES DATAS POUR DEAD CA AVEC LA LIB PANDA
data=pd.read_csv('coronavirus-covid19-tweets-late-april\data_16.csv',encoding='utf-8')
df=pd.DataFrame(data)

data1=pd.read_csv('coronavirus-covid19-tweets-late-april\data_20.csv',encoding='utf-8')
df1=pd.DataFrame(data1)

data2=pd.read_csv('coronavirus-covid19-tweets-late-april\data_24.csv',encoding='utf-8')
df2=pd.DataFrame(data2)

data3=pd.read_csv('coronavirus-covid19-tweets-late-april\data_27.csv',encoding='utf-8')
df3=pd.DataFrame(data3)

data4=pd.read_csv('coronavirus-covid19-tweets-late-april\data_30.csv',encoding='utf-8')
df4=pd.DataFrame(data4)

#####

data5=pd.read_csv('coronavirus-covid19-tweets-late-april\data_17.csv',encoding='utf-8')
df5=pd.DataFrame(data5)

data6=pd.read_csv('coronavirus-covid19-tweets-late-april\data_21.csv',encoding='utf-8')
df6=pd.DataFrame(data6)

data7=pd.read_csv('coronavirus-covid19-tweets-late-april\data_25.csv',encoding='utf-8')
df7=pd.DataFrame(data7)

data8=pd.read_csv('coronavirus-covid19-tweets-late-april\data_28.csv',encoding='utf-8')
df8=pd.DataFrame(data8)

data9=pd.read_csv('coronavirus-covid19-tweets-late-april\data_29.csv',encoding='utf-8')
df9=pd.DataFrame(data9)

#####

data10=pd.read_csv('coronavirus-covid19-tweets-late-april\data_18.csv',encoding='utf-8')
df10=pd.DataFrame(data10)

data11=pd.read_csv('coronavirus-covid19-tweets-late-april\data_22.csv',encoding='utf-8')
df11=pd.DataFrame(data11)

data12=pd.read_csv('coronavirus-covid19-tweets-late-april\data_26.csv',encoding='utf-8')
df12=pd.DataFrame(data12)

data13=pd.read_csv('coronavirus-covid19-tweets-late-april\data_19.csv',encoding='utf-8')
df13=pd.DataFrame(data13)

data14=pd.read_csv('coronavirus-covid19-tweets-late-april\data_23.csv',encoding='utf-8')
df14=pd.DataFrame(data14)

#####


# data1=pd.read_csv('coronavirus-covid19-tweets-late-april\data_21.csv',encoding='utf-8')
# df1=pd.DataFrame(data1)

# df = pd.concat([df,df1,df2,df3,df4])
df = pd.concat([df,df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14])
########################   NETTOYAGE #################################################

#ON GARDE QUE LES DATA EN ANGLAIS OU LANG=EN
df = df.query('lang == ["en"]')

# ON DEGAGE LES "#COVID" DE KIKOO
# ( dataset realisé avec les # suivant: #coronavirus, #coronavirusoutbreak, #coronavirusPandemic, #covid19, #covid_19, #epitwitter, #ihavecorona, #StayHomeStaySafe, #TestTraceIsolate)
hashtag = ["#COVID19",'#Covid_19',"#Covid19","#COVID","#Covid","#covid","#covid19",
              "#COVID19","#COVID-19","#coronavirus","#Coronavirus","#CoronaVirus","#Corona","#corona","#coronavirusoutbreak","#epitwitter",
              "#ihavecorona","#StayHomeStaySafe","#TestTraceIsolate"]

for kikoo in hashtag:
    df['text'] = df['text'].str.replace(kikoo, '')

# ON DEGAGE LES NOMBRES
df['text'] = df['text'].str.replace('\d+', '')

# ON DEGAGE LES JOURS ET MOIS
jours_mois = ["January",'February',"March","April","May","June","July",
              "August","Septembre","October","November","December","Monday","Tuesday","Wednesday","Thursday",
              "Friday","Saturday","Sunday"]

for date in jours_mois:
    df['text'] = df['text'].str.replace(date, '')

# ON DEGAGE LES URL MA BITE
df['text'] = df['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

# ON DEGAGE LES CARACTERES SPECIAUX SAUF LE HASTAG QUI PEUT SERVIR IL A DIT
#Le @ ça designe une personne mais ça dégage, on laisse le nom de la personne
spec_chars = ["!",'"',"%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]

for char in spec_chars:
    df['text'] = df['text'].str.replace(char, '')

df['text'] = df['text'].str.replace('  ', ' ') #double espace to un espace

# ON GARDE QUE LA COLONNE 'TEXT'
df=df[['text']]

#########################################################################
########################  TRI #################################################

#####TRI PAR EMOJI######### 10 / 13 / 7 / 7
emoji_alegria = u"\U0001F601|\U0001F602|\U0001F603|\U0001F604|\U0001F606|\U0001F609|\U0001F60A|\U0001F60D|\U0001F600|\U0001263A" #ajout emoji yeux en coeur, smiley additionnal 600, 263A
emoji_tristeza = u"\U0001F613|\U0001F616|\U0001F622|\U0001F62D|\U0001F630|\U0001F631|\U0001F64D|\U0001F64E|\U0001F614|\U0001F61E|\U0001F625|\U0001F63F|\U0001F61F"  #ajout emoji yeux vers le bas 614 et 61E deception, 625 pleures, 63F chat qui pleure, 61F
emoji_enojo = u"\U0001F620|\U0001F621|\U0001F624|\U0001F63E|\U0001F47F|\U0001F4A2|\U0001F4A5" #angry cat 63E, , 47F satan, anger symbol 4A2, collision symbol 4A5
emoji_apoyo = u"\U0001F637|\U0001F64B|\U0001F64C|\U0001F64F|\U0001270A|\U0001F4AA|\U0001F44A" #raised fist 270A, flexed biceps 4AA,fist 44A

#####TEXTE CONTENANT UNE CATEGORIE EMOJI#########
contenir_emoji_alegria=df['text'].str.contains(emoji_alegria)
contenir_emoji_tristeza=df['text'].str.contains(emoji_tristeza)
contenir_emoji_enojo=df['text'].str.contains(emoji_enojo)
contenir_emoji_apoyo=df['text'].str.contains(emoji_apoyo)

#####ON TRIE CHQ TWEET EN LE CLASSANT DANS UNE CATEGORIE SEULEMENT SI IL CONTIENT UN SEUL TYPE D'EMOJI D'UNE CATEGORIE#########

#valeur logique d'une categorie
full_alegria = (contenir_emoji_alegria) & (~(contenir_emoji_enojo)) & (~(contenir_emoji_tristeza)) & (~(contenir_emoji_apoyo))
full_tristeza = (contenir_emoji_tristeza) & (~(contenir_emoji_enojo)) & (~(contenir_emoji_alegria)) & (~(contenir_emoji_apoyo))
full_enojo = (contenir_emoji_enojo) & (~(contenir_emoji_alegria)) & (~(contenir_emoji_tristeza)) & (~(contenir_emoji_apoyo))
full_apoyo = (contenir_emoji_apoyo) & (~(contenir_emoji_enojo)) & (~(contenir_emoji_alegria)) & (~(contenir_emoji_tristeza))
#tri par categorie
df_alegria=df[ full_alegria ]
df_tristeza=df[ full_tristeza ]
df_enojo=df[ full_enojo ]
# df_enojo['categorie']= 2
df_apoyo=df[ full_apoyo ]

#sans etiquette = pas trié
df_mix_sans_etiquette=df[ (~(full_alegria)) & (~(full_tristeza)) & (~(full_enojo)) & (~(full_apoyo)) ]
df_mix_sans_etiquette = df_mix_sans_etiquette.sample(frac=1)
(nb_tweet_mix_sans_etiquette,cat)=df_mix_sans_etiquette.shape
print(nb_tweet_mix_sans_etiquette)
#avec etiquette = trié par catégorie
frames = [df_alegria,df_tristeza,df_enojo,df_apoyo]
df_etiquette = pd.concat(frames)

#########################################################################
#################GO TO CSV###############################################
df_alegria.to_csv('joie.csv', index=False,header=False, encoding='utf-8')
df_tristeza.to_csv('tristesse.csv', index=False,header=False, encoding='utf-8')
df_enojo.to_csv('nrv.csv', index=False,header=False, encoding='utf-8')
df_apoyo.to_csv('challa.csv', index=False,header=False, encoding='utf-8')

# df_etiquette.to_csv('etiquette.csv', index=False,header=False, encoding='utf-8')
df_mix_sans_etiquette.to_csv('mix_sans_etiquette.csv', index=False,header=False, encoding='utf-8') #text total sans les catégorie = TWEET SIN CLASSIFICAR
# df.to_csv('total.csv', index=False,header=False, encoding='utf-8') #text total qu'on garde
#########################################################################


# oui=data.head() #lire les 5 première ligne d'un csv
# print(oui)
#
# # print le nom des columns python
# with open('coronavirus-covid19-tweets-late-april\data_16.csv',encoding='Latin1') as fp:
#     reader = csv.reader(fp)
#     header = next(reader)
#     print(f"header: {header}")

# res = pd.read_csv('test.csv')
# bite=res.head() #lire les 5 première ligne d'un csv
# print(bite)0


# res = pd.read_csv('output.csv')
# bite=res.head() #lire les 5 première ligne d'un csv
# print(bite)


print("Hello World")
