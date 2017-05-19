import json
import csv
import spacy
import en_core_web_md
import logging

coco_path = '/home/gchrupal/reimaginet/data/coco'
data_path = '/home/gchrupal/cl-resubmit/data/'

def main():
    logging.getLogger().setLevel('INFO')
    nlp = en_core_web_md.load()
    split = 'val'
    sent = sentences(json.load(open(coco_path + '/dataset.json')), split=split)
    writer = csv.writer(open(data_path + '/depparse_coco_val.csv',"w"))
    writer.writerow(["sentid", "position", "word", "postag", "dep", "head"])
    for sent_i in sent:
        logging.info("Parsing: {} {}".format(sent_i['sentid'], ' '.join(sent_i['tokens'])))
        postag, label, head = parse(nlp, sent_i['tokens'])
        for i in range(len(sent_i['tokens'])):
            writer.writerow([sent_i['sentid'],
                            i,
                            sent_i['tokens'][i],
                            postag[i],
                            label[i],
                            head[i]])


def parse(nlp, tokens):
    doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
    nlp.tagger(doc)
    nlp.parser(doc)
    head =  [word.head.i for word in doc]
    label = [word.dep_ for word in doc]
    postag = [word.tag_ for word in doc]
    return postag, label, head

def sentences(data, split='val'):
    for image in data['images']:
        if image['split'] == split:
            for sentence in image['sentences']:
                yield sentence

if __name__ == '__main__':
    main()
