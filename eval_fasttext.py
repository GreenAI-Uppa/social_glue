import scipy
import csv
import embed_fasttext
import json
import eval_retrieval

import fasttext

model = fasttext.load_model('/home/paul/data/social_computing/derniere/model2.bin')
#model = fasttext.load_model('/home/paul/data/social_computing/twitter/atelier_5_1/fasttext24K.bin')
#model = fasttext.load_model('/home/paul/data/social_computing/twitter/69.bin')
contents = json.load(open('all_contents.json'))
queries = json.load(open('queries.json'))
queries_and_gt = json.load(open('queries_and_gt.json'))

mode = 'sentence' #'average'

embed_cont = {}
for i, con in contents.items():
    if mode == 'average':
      embed_cont[i] = embed_fasttext.embedding(con, model)
    else:
      embed_cont[i] = embed_fasttext.get_sentence_vector(con, model)

split = 'val'
predictions = {}
for q in queries[split]:
    if mode == 'average':
      embed_q = embed_fasttext.embedding(q["query"], model)
    else:
      embed_q = embed_fasttext.get_sentence_vector(q["query"], model)
    preds = sorted([ (scipy.spatial.distance.cosine(embed_q, embed_c), i) for (i,embed_c) in embed_cont.items()])[:10]
    #predictions[q['id']] = preds
    predictions[q['id']] = [ int(i) for (sc,i) in preds]

Y = dict([(q['id'],q['gt_ids']) for q in queries_and_gt[split]])

print(eval_retrieval.get_score(predictions,Y,tops=[1,2,5,10]))


#for sc, pred in predictions[10]: #.items()
#    print(sc, contents[pred])
