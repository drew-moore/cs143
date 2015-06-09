import csv
from collections import defaultdict
from nltk.corpus import wordnet as wn

nouns = None
verbs = None

def load_wordnet_ids(filename):
    file = open(filename, 'r')
    if "noun" in filename: type = "noun"
    else: type = "verb"
    csvreader = csv.DictReader(file, delimiter=",", quotechar='"')
    word_ids = defaultdict()
    for line in csvreader:
        word_ids[line['synset_id']] = {'synset_offset': line['synset_offset'], 'story_'+type: line['story_'+type], 'stories': line['stories']}
    return word_ids

def initialize():
    global nouns
    global verbs
    noun_ids = load_wordnet_ids("Wordnet_nouns.csv")
    verb_ids = load_wordnet_ids("Wordnet_verbs.csv")
    nouns = {noun_ids[d]['story_noun'] : (d, noun_ids[d]) for d in noun_ids}
    verbs = {verb_ids[v]['story_verb'] : (v, verb_ids[v]) for v in verb_ids}

def get_noun_synset(word):
    global nouns
    if word in nouns:
        sense = nouns[word][0]
        ss = wn.synset(sense)
        return ss
    else:
        return None

def get_verb_synset(word):
    global verbs
    if word in verbs:
        sense = verbs[word][0]
        ss = wn.synset(sense)
        return ss
    else:
        return None


def get_noun_hypernym(word):
    ss = get_noun_synset(word)
    if ss and ss.hypernyms():
        if ss.hypernyms():
            return ss.hypernyms()[0].name()
        else:
            return None

def get_noun_hyponyms(word):
    ss = get_noun_synset(word)
    if ss:
        i = 3

def get_verb_hypernym(word):
    ss = get_verb_synset(word)
    if ss and ss.hypernyms():
        if ss.hypernyms():
            return ss.hypernyms()[0].name()
        else:
            return None



def get_noun_synonyms(word):
    ss = get_noun_synset(word)
    ret = [lemma.name() for lemma in ss.lemmas()]
    return ret

def get_hypernym(word):
    ret = get_noun_hypernym(word)
    if not ret:
        ret = get_verb_hypernym(word)
    return ret

def disambiguate(sent):
    ret = []
    for tup in sent:
        if tup[1].startswith('V'):
            curr = get_verb_hypernym(tup[0])
        else:
            curr = get_noun_hypernym(tup[0])
        if curr:
            ret.append((curr, tup[1]))
        else:
            ret.append((tup[0], tup[1]))
    return ret

if __name__ == "__main__":

    ## You can use either the .csv files or the .dict files.
    ## If you use the .dict files, you MUST use "rb"!


    # {synset_id : {synset_offset: X, noun/verb: Y, stories: set(Z)}}, ...}
    # e.g. {help.v.01: {synset_offset: 2547586, noun: aid, stories: set(Z)}}, ...
    #noun_ids = pickle.load(open("Wordnet_nouns.dict", "rb"))
    #verb_ids = pickle.load(open("Wordnet_verbs.dict", "rb"))

    # iterate through dictionary
    for synset_id, items in noun_ids.items():
        noun = items['story_noun']
        stories = items['stories']
        # print(noun, stories)
        # get lemmas, hyponyms, hypernyms

    for synset_id, items in verb_ids.items():
        verb = items['story_verb']
        stories = items['stories']
        # print(verb, stories)
        # get lemmas, hyponyms, hypernyms


    # 'Rodent' is a hypernym of 'mouse',
    # so we look at hyponyms of 'rodent' to find 'mouse'
    #
    # Question: Where did the rodent run into?
    # Answer: the face of the lion
    # Sch: The lion awaked because a mouse ran into the face of the lion.
    rodent_synsets = wn.synsets("rodent")
    print("'Rodent' synsets: %s" % rodent_synsets)

    print("'Rodent' hyponyms")
    for rodent_synset in rodent_synsets:
        rodent_hypo = rodent_synset.hyponyms()
        print("%s: %s" % (rodent_synset, rodent_hypo))

        for hypo in rodent_hypo:
            print(hypo.name()[0:hypo.name().index(".")])
            print("is hypo_synset in Wordnet_nouns/verbs.dict?")
            # match on "mouse.n.01"


    # 'Know' is a hyponym of 'recognize' (know.v.09),
    # so we look at hypernyms of 'know' to find 'recognize'
    #
    # Question: What did the mouse know?
    # Answer: the voice of the lion
    # Sch: The mouse recognized the voice of the lion.
    know_synsets = wn.synsets("know")
    print("\n'Know' synsets: %s" % know_synsets)

    print("'Know' hypernyms")
    for know_synset in know_synsets:
        know_hyper = know_synset.hypernyms()
        print("%s: %s" % (know_synset, know_hyper))

    # 'Express mirth' is a lemma of 'laugh'
    # so we look at lemmas of 'express mirth' to find 'laugh'
    #
    # Question: Who expressed mirth?
    # Answer: the lion
    # Sch: The lion laughed aloud because he thought that the mouse is extremely not able to help him.
    mirth_synsets = wn.synsets("express_mirth")
    print("\n'Express Mirth' synsets: %s" % mirth_synsets)

    print("'Express mirth' lemmas")
    for mirth_synset in mirth_synsets:
        print(mirth_synset)

        # look up in dictionary
        print("\n'%s' is in our dictionary: %s" % (mirth_synset.name(), (mirth_synset.name() in verb_ids)))


