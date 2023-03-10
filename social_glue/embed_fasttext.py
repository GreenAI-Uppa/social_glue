import numpy as np

#analyzing word embeddings
from sklearn.neighbors import KDTree

#NLP
import string
from nltk import word_tokenize
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop

punctuations = list(string.punctuation) # liste de caractère de ponctuations


## On transforme une phrase en vecteur grâce au model
punctuations = list(string.punctuation)
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
stop_words_list = list(fr_stop)

def preproc_(sentence):
    sentence = [i for i in word_tokenize(sentence) if i not in punctuations and i not in stop_words_list ]
    """
    sentence = ''
    for word in sen:
        if word not in punctuations:
            sentence += word
    sentence_bis = ''
    for word in sentence.split():
        if word.lower() not in stop_words_list:
            sentence_bis += word+' '
    """
    return sentence

def get_sentence_vector(sentence : string, model):
    sentence = " ".join(preproc_(sentence))
    return model.get_sentence_vector(sentence)

def embedding(sentence, model):
    # NOTE: removing punctuation seems already be done in voisins_sentence. Remove this condition in the upper function ?
    sentence = preproc_(sentence)
    sentence_vector = []
    for word in sentence:
        sentence_vector.append(model[word])
    if len(sentence_vector) == 0:
        print('WARNING zero length vector to mean in embedding()')
    res = np.mean(np.array(sentence_vector), axis=0)
    return res

def voisins_sentence(contenu, contents, model, n_voisin):
    ### on pre-traite le contenu
    punctuations = list(string.punctuation)
    stop_words_list = list(fr_stop)
    contenu_not_punctuate = ''
    for word in contenu:
        if word not in punctuations:
            contenu_not_punctuate += word
    contenu_tokenize = ''
    for word in contenu_not_punctuate.split():
        if word.lower() not in stop_words_list:
            contenu_tokenize += word+' '
    zone = embedding(contenu_tokenize, model)
    #print('On recherche les voisins de :\n')
    #print(contenu,'\n')
    X = np.zeros((len(contents), zone.shape[0]))
    for k,content in enumerate(contents):
        ### on va enlever la ponctuation et les stops words ###
        sentence = ''
        for word in content[0]:
            if word not in punctuations:
                sentence += word
        sentence_bis = ''
        for word in sentence.split():
            if word.lower() not in stop_words_list:
                sentence_bis += word+' '
        ###
        X[k,:] = embedding(sentence_bis, model)
    print(zone[:10])
    zone = np.array(zone).reshape(1,zone.shape[0])
    tree = KDTree(X,leaf_size=40)
    dist, ind = tree.query(zone, n_voisin)
    tab_res = []
    for k,indice in enumerate(ind[0]):
        if k>=0:
            tab_res.append(indice)
            #print('Voisin',k, ':',indice,contents[indice]['text'])
            #print(contents[indice]['source'], contents[indice]['tag'], '\n')
    return tab_res, dist
