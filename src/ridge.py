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

def tune(D, ix=100000):
    Y = D['Y']
    scores = dict(word=[], dep=[], situation=[], full=[])
    alphas = [ 2**n for n in range(-2,6) ]
    predict = dict(word=[], dep=[], situation=[], full=[])
    models = dict(word=[], dep=[], situation=[], full=[])
    def score(alpha, modeltyp):
        model = Ridge(alpha=alpha)
        model.fit(D[modeltyp][:ix,:], Y[:ix])
        scores[modeltyp].append(model.score(D[modeltyp][ix:,:], Y[ix:]))
        predict[modeltyp].append(model.predict(D[modeltyp][ix:,:]))
        models[modeltyp].append(model)
    for alpha in alphas:
        for modeltyp in ['word','dep','situation','full']:
            score(alpha, modeltyp)
    return {'alphas':alphas, 'scores':scores, 'predict': predict, 'models': models }

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

def coef(D, model):
    coef_full = D['encoder']['full'].inverse_transform(model.coef_)[0]
    for which in ['first','second','third','middle','antepenult','penult','last']:
        yield (which, round(coef_full['situation={}'.format(which)],3))


def topwords(valid):
    data_freq = valid.groupby("word").filter(lambda x: len(x) > 100)
    def er(row):
        hi = abs(row["omission_v_pred_word"]-row["omission_v"])
        lo = abs(row["omission_v_pred_dep"]-row["omission_v"])
        return hi-lo
    data_freq["er"] = data_freq.apply(er, axis=1)
    top7_er = data_freq.groupby("word")["er"].mean().sort_values()[-7:].index
    return data_freq[data_freq["word"].isin(top7_er)][["word","dep","omission_v"]]

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
    IX = 100000
    logging.info("Training Ridge models on visual")
    result_v = tune(data_v, ix=IX)
    logging.info("Training Ridge models on textual")
    result_t = tune(data_t, ix=IX)
    logging.info("Training Ridge models on LM")
    result_lm = tune(data_lm, ix=IX)
    logging.info("Training Ridge models on vectorsum")
    result_sum = tune(data_sum, ix=IX)
    with open("../data/ridge_scores.txt","w") as out:
        out.write("model\tpredictors\tR2\n")
        predscore("visual", result_v['scores'], out)
        predscore("textual", result_t['scores'], out)
        predscore("LM", result_lm['scores'], out)
        predscore("sum", result_sum['scores'], out)
    # Predictions
    logging.info("Dumping predictions")
    valid = data.loc[IX:,:]
    best_word = numpy.argmax(result_v['scores']['word'])
    best_dep  = numpy.argmax(result_v['scores']['dep'])
    valid['omission_v_pred_word'] = result_v['predict']['word'][best_word]
    valid['omission_v_pred_dep'] = result_v['predict']['dep'][best_dep]
    valid.to_csv("../data/ridge_predict.csv", index=False)
    #valid = pd.read_csv("../data/ridge_predict.csv")
    logging.info("Saving ridge coefficients")
    with open("../data/position_coef.txt","w") as out:
        out.write("model coef value\n")
        for item in zip(["visual","textual","LM","sum"],
                        [data_v, data_t, data_lm, data_sum],
                        [result_v, result_t, result_lm, result_sum]):
            name, dat, res = item
            best = res['models']['full'][numpy.argmax(res['scores']['full'])]
            for rec in coef(dat, best):
                out.write("{} {} {}\n".format(name, rec[0], rec[1]))
    logging.info("Computing top dependency-sensitive words")
    topwords(valid).to_csv("../data/top7_words_er.csv", index=False)


if __name__ == '__main__':
    main()
