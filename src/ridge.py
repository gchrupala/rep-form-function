from sklearn.linear_model import Ridge
import pandas as pd
import numpy
import sys
import logging

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

def choose(ds, keys):
        return  ( dict([(key,d[key]) for key in keys ]) for d in ds )

def dataset(data, score='score'):
    dicts = data.T.to_dict().values()
    encoder = dict(word=DictVectorizer(), dep=DictVectorizer(), situation=DictVectorizer(), full=DictVectorizer())
    X_word   = encoder['word'].fit_transform(choose(dicts, ['word']))
    X_dep = encoder['dep'].fit_transform(choose(dicts, ['word', 'dep', 'word:dep']))
    X_situation = encoder['situation'].fit_transform(choose(dicts, ['word','situation','word:situation']))
    X_full = encoder['full'].fit_transform(choose(dicts, ['word','dep','word:dep','situation','word:situation']))
    return dict(Y=numpy.array(data[score]), encoder=encoder, word=X_word, dep=X_dep, situation=X_situation, full=X_full)

def ridge_alphas(D, ix=100000):
    Y = D['Y']
    scores = dict(word=[], dep=[], situation=[], full=[])
    alphas = [ 2**n for n in range(-2,6) ]

    def score(alpha, modeltyp):
        model = Ridge(alpha=alpha)
        model.fit(D[modeltyp][:ix,:], Y[:ix])
        scores[modeltyp].append(model.score(D[modeltyp][ix:,:], Y[ix:]))

    for alpha in alphas:
        for modeltyp in ['word','dep','situation','full']:
            score(alpha, modeltyp)
    return (alphas, scores)

def predscore(kind, scores, out):
        for key in scores.keys():
            out.write("{}\t{}\t{}\n".format(kind, key, round(max(scores[key]),3)))

def situation(row):
    if row['position'] == 0:
        return "first"
    elif row['position'] == 1:
        return "second"
    elif row['position'] == 2:
        return "third"
    elif row['position'] + 1 == row['length']:
        return "last"
    elif row['position'] + 1 == row['length'] - 1:
        return "penult"
    elif row['position'] + 1 == row['length'] - 2:
        return "antepenult"
    else:
        return "middle"

# Read and preprocess data
def main():
    logging.getLogger().setLevel('INFO')
    logging.info("Reading and preprocessing data")
    depparse = pd.read_csv("../data/depparse_coco_val.csv")
    omit = pd.read_csv("../data/omission_coco_val.csv")
    data = pd.merge(depparse, omit)

    length = data.groupby("sentid")["position"].max()+1
    data = data.merge(pd.DataFrame(dict(sentid=length.index, length=numpy.array(length))))
    data["situation"] = data.apply(situation, axis=1)
    data["word:dep"] = data["word"]+":"+data["dep"]
    data["word:situation"] = data["word"]+":"+data["situation"]
    data.to_csv("../data/ridge_data.csv", index=False)
    data_v = dataset(data, score='omission_v')
    data_t = dataset(data, score='omission_t')
    data_lm = dataset(data, score='omission_lm')
    data_sum = dataset(data, score='omission_sum')
    logging.info("Training Ridge models on visual")
    alphas, scores_v = ridge_alphas(data_v)
    logging.info("Training Ridge models on textual")
    alphas, scores_t = ridge_alphas(data_t)
    logging.info("Training Ridge models on LM")
    alpahs, scores_lm = ridge_alphas(data_lm)
    logging.info("Training Ridge models on vectorsum")
    alphas, scores_sum =ridge_alphas(data_sum)
    with open("../data/ridge_scores.txt","w") as out:
        out.write("model\tpredictors\tR2\n")
        predscore("visual", scores_v, out)
        predscore("textual", scores_t, out)
        predscore("LM", scores_lm, out)
        predscore("sum", scores_sum, out)

if __name__ == '__main__':
    main()
