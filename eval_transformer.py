from tqdm import tqdm 
from scipy import spatial
from social_glue import embed_trans
import json, os
from social_glue import eval_retrieval

from sentence_transformers import util

cache_folder='/data/pgay/hugging_face_cache'

"""
from transformers import BloomModel, BloomTokenizerFast
model_name = "bigscience/bloom-560m"
model = BloomModel.from_pretrained(model_name, cache_folder='/data/pgay/hugging_face_cache')
tokenizer = BloomTokenizerFast.from_pretrained(model_name, cache_folder='/data/pgay/hugging_face_cache')
model = model.eval().to('cuda')
"""

"""
from transformers import CamembertModel, CamembertTokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-large",cache_folder=cache_folder)
model = CamembertModel.from_pretrained("camembert/camembert-large",cache_dir=cache_folder)
model = model.eval().to('cuda')  # disable dropout (or leave in train mode to finetune)
"""

"""
from transformers import FlaubertModel, FlaubertTokenizer 
tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_large_cased",cache_folder=cache_folder)
model = FlaubertModel.from_pretrained("flaubert/flaubert_large_cased",cache_dir=cache_folder)
model = model.eval().to('cuda')  # disable dropout (or leave in train mode to finetune)
"""

"""
from sentence_transformers import SentenceTransformer
model =  SentenceTransformer("dangvantuan/sentence-camembert-large", cache_folder='/data/pgay/hugging_face_cache')
model = model.to('cuda')
"""

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('inokufu/flaubert-base-uncased-xnli-sts')



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
    #embed_cont[i] = embed_trans.embedding(con, model, tokenizer)
    embed_cont[i] = embed_trans.embed_sentence(con, model)
    #embed_cont[i] = embed_trans.embedding_last_n(con, model, tokenizer,n=16)

split = 'val'
predictions = {}
for q in queries[split]:
    embed_q = embed_trans.embed_sentence(con, model)
    preds = sorted([ (util.cos_sim(embed_q, embed_c), i) for (i,embed_c) in embed_cont.items()], reverse=True)[:10]
    #embed_q = embed_trans.embedding(q["query"], model, tokenizer)
    #embed_q = embed_trans.embedding_last_n(q["query"], model, tokenizer,n=16)
    #preds = sorted([ (spatial.distance.cosine(embed_q, embed_c), i) for (i,embed_c) in embed_cont.items()])[:10]
    #predictions[q['id']] = preds
    predictions[q['id']] = [ int(i) for (sc,i) in preds]

Y = dict([(q['id'],q['gt_ids']) for q in queries_and_gt[split]])

print(eval_retrieval.get_score(predictions,Y,tops=[1,2,5,10]))


#for sc, pred in predictions[10]: #.items()
#    print(sc, contents[pred])
