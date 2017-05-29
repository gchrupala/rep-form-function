import imaginet.defn.lm_visual_vanilla as D
import imaginet.data_provider as dp
import numpy
import imaginet.evaluate as evaluate
import logging
import urllib
import json

root = '/home/gchrupal/reimaginet/'

def main():
    logging.getLogger().setLevel('INFO')
    logging.info("Loading imaginet model")
    model_im_path = root + "/run-lm_visual_vanilla-1/model.r.e7.zip"
    model_im = D.load(model_im_path)

    #sent = list(prov.iterSentences(split='val'))
    logging.info("Building ranker")
    ranker = Ranker(model_im)

    ex1a = "a baby sits on a bed laughing with a laptop computer open".split()
    ex1b = "a sits on a bed laughing with a laptop computer open".split()
    hits = ranker.top1([ex1a, ex1b])
    dest = "/home/gchrupal/cl-resubmit/doc/"
    logging.info("Writing images")
    for hit in hits:
        logging.info("Writing {}/{}.jpg".format(dest, hit))
        urllib.urlretrieve("http://mscoco.org/images/{}".format(hit), "{}/{}.jpg".format(dest, hit))

class Ranker:
    def __init__(self, model):
        self.model = model
        prov = dp.getDataProvider(dataset='coco', root=root, audio_kind=None)
        images = list(prov.iterImages(split='val'))
        self.img_fs = model.scaler.transform(numpy.array([ img['feat'] for img in images ], dtype='float32'))
        data = json.load(open("/home/gchrupal/reimaginet/data/coco/dataset.json"))
        COCOID = {}
        for img in data['images']:
            COCOID[img['imgid']] = img['cocoid']
        self.IDS = [COCOID[img['imgid']] for img in images]

    def top1(self, sents):
        pred = D.encode_sentences(self.model, sents)
        distances = evaluate.Cdist(batch_size=2**13)(pred, self.img_fs)
        result = []
        for row in distances:
            result.append(self.IDS[numpy.argsort(row)[0]])
        return result

if __name__ == '__main__':
    main()
