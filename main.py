import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim.utils
import re
import string

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()


@st.cache
def get_data():
    return pd.read_csv("dataset.csv" ,sep='\t', quotechar="'")

def cleandata(kalimat):
    kalimatbaru=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', kalimat)
    kalimatbaru=stopword.remove(kalimatbaru)
    return kalimatbaru

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


  
lawdata=get_data()
tanya = np.array(lawdata.Context)
jawab=np.array(lawdata.Response)
data=np.array(lawdata.Context+ ' ' +lawdata.Keywords)


def get_response(q):
	highsim=0.0
	highnum=0
	for i,line in enumerate(data):
		if (type(line)==str):
			qlist=cleandata(q).lower().split()
			linelist=cleandata(line).lower().split()
			simnow=jaccard_similarity(qlist,linelist)
			if(simnow>highsim):
				highnum=i
				highsim=simnow
	return(highnum)  
	
st.title("Assalamu 'alaikum di Open data konstitusi :)")

cari = st.text_input("Masukkan pertanyaan", "Apa tugas dari lembaga negara?")
  
# display the name when the submit button is clicked
# .title() is used to get the input text string 
if(st.button('Submit')):
	result = cari.title().lower()
	if(result):
		has=get_response(result)
		st.markdown(jawab[has])
	else:
		st.error("Masukkan input ya")
    

