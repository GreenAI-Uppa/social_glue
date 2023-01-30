from tqdm import tqdm 
from scipy import spatial
from social_glue import embed_trans
import json, os
from social_glue import eval_retrieval

cache_folder='/data/pgay/hugging_face_cache'

"""
from transformers import BloomModel, BloomTokenizerFast
model_name = "bigscience/bloom-560m"
model = BloomModel.from_pretrained(model_name, cache_folder='/data/pgay/hugging_face_cache')
tokenizer = BloomTokenizerFast.from_pretrained(model_name, cache_folder='/data/pgay/hugging_face_cache')
model = model.eval().to('cuda')
"""

#"""
from transformers import FlaubertModel, FlaubertTokenizer
tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_large_cased",cache_folder=cache_folder)
model = FlaubertModel.from_pretrained("flaubert/flaubert_large_cased",cache_dir=cache_folder)
model = model.eval().to('cuda')  # disable dropout (or leave in train mode to finetune)
#"""

"""
from transformers import CamembertModel, CamembertTokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-large",cache_folder=cache_folder)
model = CamembertModel.from_pretrained("camembert/camembert-large",cache_dir=cache_folder)
model = model.eval().to('cuda')  # disable dropout (or leave in train mode to finetune)
"""

glue_social_dir = '/data/pgay/social_computing/glue_social/'
contents = json.load(open(os.path.join(glue_social_dir,'all_contents.json')))
queries = json.load(open(os.path.join(glue_social_dir,'queries.json')))
queries_and_gt = json.load(open(os.path.join(glue_social_dir,'queries_and_gt.json')))

# check bloom, flaubert, camembert
# check with oracle, find the best possible layer
#  ie get all layers output
#  then in the query loop
#     you compute the distance for all the layers and you take the min
#        NB : you take the distance between query layer x and content layer x

# then you can try average, average over the last four, or results for each layer individually

embed_cont = {}
for i, con in tqdm(contents.items()):
    embed_cont[i] = embed_trans.embedding_all_l(con, model, tokenizer)


# oracle validation
# we take the best layer for the prediction
split = 'val'
predictions = {}
best_layers = []
for q in queries[split]:
    embed_q = embed_trans.embedding_all_l(q['query'], model, tokenizer)
    # get the id of the good answer
    gtid = None
    qid = q['id']
    for gt in queries_and_gt[split]:
        if gt['id'] == qid:
            gtid = gt['gt_ids']
            gt_content = gt['gt_contents']
    best_pred = 20 
    best_layer = -1
    for l in embed_q.keys(): # for each layer
        # get the ranking of the content
        preds = sorted([ (spatial.distance.cosine(embed_q[l], embed_c[l]), i) for (i,embed_c) in embed_cont.items()])[:10]
        preds = [ int(i) for (sc,i) in preds]
        # if the good answer is in the predictions
        if gtid in preds:
            prop = preds.index(gtid)
            # and if it is best rank than before
            if prop < best_pred:
                prop = best_pred
                predictions[q['id']] = preds 
                best_layer = l
    best_layers.append((best_layer, q['query'],gt_content))
    if q['id'] not in predictions:
        predictions[q['id']] = list(range(10))

Y = dict([(q['id'],q['gt_ids']) for q in queries_and_gt[split]])
print('best layers')
for best_layer, q, c in best_layers:
    print(best_layer, q, c)
print(eval_retrieval.get_score(predictions,Y,tops=[1,2,5,10]))


#for sc, pred in predictions[10]: #.items()
#    print(sc, contents[pred])
