import json
import csv
import logging
import imaginet.data_provider as dp
import imaginet.defn.lm_visual_vanilla as D
from scipy.spatial.distance import cosine
import imaginet.task

root = '/home/gchrupal/reimaginet/'
data_path = '/home/gchrupal/cl-resubmit/data/'
model_im_path = root + "/run-lm_visual_vanilla-1/model.r.e7.zip"
model_lm_path = root + "/run-lm-0/model.r.e6.zip"

def main():
    logging.getLogger().setLevel('INFO')
    logging.info("Loading data")
    prov = dp.getDataProvider(dataset='coco', root=root, audio_kind=None)
    sent = list(prov.iterSentences(split='val'))
    sent_id = [ sent_i['sentid'] for sent_i in sent]
    sent_tok = [ sent_i['tokens'] for sent_i in sent]
    logging.info("Loading imaginet model")
    model_im = D.load(model_im_path)
    logging.info("Loading plain LM model")
    model_lm = imaginet.task.load(model_lm_path)
    writer = csv.writer(open(data_path + '/omission_coco_val.csv',"w"))
    writer.writerow(["sentid", "position", "word", "omission_v","omission_t","omission_lm"])
    for i in  range(len(sent)):
        logging.info("Processing: {}".format(sent_id[i]))
        O_v = omission(model_im, sent_tok[i], task=model_im.visual)
        O_t = omission(model_im, sent_tok[i], task=model_im.lm)
        O_lm = omission(model_lm, sent_tok[i], task=model_lm.task)
        for j in range(len(sent_tok[i])):
            writer.writerow([sent_id[i],
                            j,
                            sent_tok[i][j],
                            O_v[j],
                            O_t[j],
                            O_lm[j]])

def omission(model, toks, task=None):
    if task is None:
        task = model.visual
    orig = imaginet.task.states(model, [toks], task=task)[0]
    omit = imaginet.task.states(model,
                      [ toks[:i] + toks[i+1:] for i in range(len(toks))],
                      task=task)
    return [ cosine(orig[-1], omit_i[-1]) for omit_i in omit]



if __name__ == '__main__':
    main()
