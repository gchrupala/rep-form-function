import json
import csv
import logging
import imaginet.data_provider as dp
import imaginet.defn.lm_visual as D
moe
from scipy.spatial.distance import cosine


root = '/home/gchrupal/reimaginet/'
data_path = '/home/gchrupal/cl-resubmit/data/'
model_path = root + "/run-lm_visual-1/model.r.e6.zip"
def main():
    logging.getLogger().setLevel('INFO')
    logging.info("Loading data")
    prov = dp.getDataProvider(dataset='coco', root=root, audio_kind=None)
    sent = list(prov.iterSentences(split='val'))
    sent_id = [ sent_i['sentid'] for sent_i in sent]
    sent_tok = [ sent_i['tokens'] for sent_i in sent]
    logging.info("Loading imaginet model")
    model = D.load(model_path)
    writer = csv.writer(open(data_path + '/omission_coco_val.csv',"w"))
    writer.writerow(["sentid", "position", "word", "omission_v","omission_t"])
    for i in  range(len(sent)):
        logging.info("Processing: {}".format(sent_id[i]))
        O_v = omission(model, sent_tok[i], task=model.visual)
        O_t = omission(model, sent_tok[i], task=model.lm)
        for j in range(len(sent_tok[i])):
            writer.writerow([sent_id[i],
                            j,
                            sent_tok[i][j],
                            O_v[j],
                            O_t[j]])

def omission(model, toks, task=None):
    if task is None:
        task = model.visual
    orig = task.states(model, [toks], task=model.visual)[0]
    omit = task.states(model,
                      [ toks[:i] + toks[i+1:] for i in range(len(toks))],
                      task=model.visual)
    return [ cosine(orig[-1], omit_i[-1]) for omit_i in omit]



if __name__ == '__main__':
    main()
