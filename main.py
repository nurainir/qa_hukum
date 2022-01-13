import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.utils
import re
import string
import pickle


@st.cache(allow_output_mutation=True)
def get_data():
    return pd.read_csv("dataset.csv" ,sep='\t', quotechar="'")


def cleandata(kalimat):
    stop_words = {'begitu','maupun','hanya','adalah','yakni','bagi','yang','dengan','itu','ada','dan','atau', 'ini', 
                  'itu','di','ke','ini','jika','of','dalam','pada','yaitu','saja','untuk','dapat','adalah','dari','karena','tidak','juga','dan/atau',
                  'bahwa','harus','kita','apabila','oleh','apakah','mengapa','bagaimana','bagaimanakah','apa','dimaksud'
                 ,'sajakah','pada','maka','bolehkah'}
    kalimat = kalimat.replace("-", " ")
    kalimat = kalimat.replace("berapa", "jumlah")
    kalimat = kalimat.replace("mengapa", "alasan penyebab")
    kalimat = kalimat.replace("apakah yang dimaksud", "definisi")
    kalimat = kalimat.replace("apa yang dimaksud", "definisi")
    kalimatbaru=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', kalimat)
    kalimatbaru=re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", kalimatbaru)
    word_list=kalimatbaru.split()
    output = [w for w in word_list if not w in stop_words]
  
    kalimatbaru = " ".join(output)
    return kalimatbaru

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

lawdata=get_data()
jawab=np.array(lawdata.Response)
lawdata['gabung']=lawdata.Context.str.lower()+ ' ' +lawdata.Keywords.str.lower()
lawdata['gabung']=lawdata['gabung'].apply(lambda x :cleandata(x))
lawdata['gabung2']=lawdata.Response.str.lower()+ ' ' +lawdata.Keywords.str.lower()
lawdata['gabung2']=lawdata['gabung2'].apply(lambda x :cleandata(x))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(np.array(lawdata.gabung))
df=pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names())

vectorizer2 = TfidfVectorizer()
X2 = vectorizer2.fit_transform(np.array(lawdata.gabung2))
df2=pd.DataFrame(X2.T.toarray(), index=vectorizer2.get_feature_names())

def get_response(q):
	data=np.array(lawdata.gabung)
	similarity=np.zeros((len(data)))
	for i,line in enumerate(data):
		if (type(line)==str):
			qlist=cleandata(q).lower().split()
			linelist=cleandata(line).lower().split()
			similarity[i]=jaccard_similarity(qlist,linelist)
	idx=(-similarity).argsort()[:5]
	dataku = []
	i=1
	for y in idx:
    		dataku.append([i,jawab[y],similarity[y]])
    		i = i +1
	return(dataku)  

def get_response2(q):
	data=np.array(lawdata.gabung2)
	similarity=np.zeros((len(data)))
	for i,line in enumerate(data):
		if (type(line)==str):
			qlist=cleandata(q).lower().split()
			linelist=cleandata(line).lower().split()
			similarity[i]=jaccard_similarity(qlist,linelist)
	idx=(-similarity).argsort()[:5]
	dataku = []
	i=1
	for y in idx:
    		dataku.append([i,jawab[y],similarity[y]])
    		i = i +1
	return(dataku)  

def get_similar_articles(q,df):
    data=np.array(lawdata.gabung)
    q=cleandata(q)
    q = [q]
    q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
    
    sim = {}
    for i in range(len(data)):
        sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
  
    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
    dataku = []
    i=1
    for k, v in sim_sorted:
        if i == 6:
            break
        if v != 0.0:
            dataku.append([i,jawab[k],v])
        i = i +1
    return(dataku)

def get_similar_articles2(q,df2):
    data=np.array(lawdata.gabung2)
    q=cleandata(q)
    q = [q]
    q_vec = vectorizer2.transform(q).toarray().reshape(df2.shape[0],)
    
    sim = {}
    for i in range(len(data)):
        sim[i] = np.dot(df2.loc[:, i].values, q_vec) / np.linalg.norm(df2.loc[:, i]) * np.linalg.norm(q_vec)
  
    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
    dataku = []
    i=1
    for k, v in sim_sorted:
        if i == 6:
            break
        if v != 0.0:
            dataku.append([i,jawab[k],v])
        i = i +1
    return(dataku)
	
st.title("Assalamu 'alaikum di Open data konstitusi :)")

cari = st.text_input("Masukkan pertanyaan", "Apa tugas dari lembaga negara?")
  
# display the name when the submit button is clicked
# .title() is used to get the input text string 
if(st.button('Submit')):
	result = cari.title().lower()
	if(result):
		st.header("Hasil Pencarian Algoritma 1 "+result)
		mydata=get_response(result)
		hasil = pd.DataFrame(mydata, columns=['Rangking','Hasil', 'Nilai Similarity'])
		st.table(hasil)
		
		
		st.header("Hasil Pencarian Algoritma 2 "+result)
		mydata2=get_response2(result)
		hasil2 = pd.DataFrame(mydata2, columns=['Rangking','Hasil', 'Nilai Similarity'])
		st.table(hasil2)
		
		st.header("Hasil Pencarian Algoritma 3 "+result)
		mydata3=get_similar_articles(result, df)
		hasil3 = pd.DataFrame(mydata3, columns=['Rangking','Hasil', 'Nilai Similarity'])
		st.table(hasil3)
		
		
		st.header("Hasil Pencarian Algoritma 4 "+result)
		mydata4=get_similar_articles2(result, df2)
		hasil4 = pd.DataFrame(mydata4, columns=['Rangking','Hasil', 'Nilai Similarity'])
		st.table(hasil4)
		
	else:
		st.error("Masukkan input ya")
    

