
# Uses dependency trees to

import re, sys, nltk, operator
from nltk.parse import DependencyGraph
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from DepGraphADT import DepGraph
import disambiguator
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

stopwords = set(nltk.corpus.stopwords.words("english"))

disambiguator.initialize()
lmtzr = WordNetLemmatizer()
# Read the lines of an individual dependency parse

def lemmatization(word, tag):
    if tag in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):
        pos = 'v'
    elif tag in ('JJ', 'JJR', 'JJS'):
        pos = 'a'
    elif tag in ('RB','RBR', 'RBS'):
        pos = 'r'
    else:
        pos = 'n'
    return WordNetLemmatizer().lemmatize(word, pos)

def get_bow(tagged_tokens, stopwords):
    bow = []
    for token in tagged_tokens:
        token_lower = token[0].lower()
        #print(token_lower)
        if token_lower not in stopwords:
            tag = token[1]
            #print(tag)
            token_stem = PorterStemmer().stem(token_lower)
            token_lemma = lemmatization(token_stem, tag)
            bow.append(token_lemma)

    return set(bow)

def baseline(qbow, sentences, stopwords):
    # Collect all the candidate answers
    answers = []
    for sent in sentences:
        sbow = get_bow(sent, stopwords)

        overlap = len(qbow & sbow)

        answers.append((overlap, sent))

    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    answers_overlap = [num[0] for num in answers]
    max_overlap = max(answers_overlap)
    sentence_list = [val[1] for index, val in enumerate(answers) if val[0] == max_overlap]
    best_answer = [sent for sentence in sentence_list for sent in sentence]

    if len(sentence_list) == 1:
        return best_answer
    else:
        return ""



def lemmatize_node(node):
    if node and 'tag' in node and node['tag'] != 'TOP':
        if node['tag'].startswith("V"):
            node['lemma'] = lmtzr.lemmatize(word=node['word'], pos='v')
            return node
        else:
            node['lemma'] = lmtzr.lemmatize(word=node['word'], pos='n')
            return node
    return None

def baseline_disambig(qgraph, sgraphs):
    global stopwords
    orig_order = {sgraphs[i]:i for i in range(0, len(sgraphs))}

    qbow = get_bow(qgraph.get_sentence(), stopwords)
    answers = []
    for sgraph in sgraphs:
        sbow = get_bow(sgraph.get_sentence(), stopwords)
        overlap = len(qbow & sbow)
        answers.append((overlap, sgraph))

    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    answers_overlap = [num[0] for num in answers]
    max_overlap = max(answers_overlap)
    results = [val[1] for index, val in enumerate(answers) if val[0] == max_overlap]

    if len(results) == 1:
        return [orig_order[results[0]]]
    else:
        qbow = get_bow(disambiguator.disambiguate(qgraph.get_sentence()), stopwords)
        d_answers = []
        for sgraph in sgraphs:
            # A list of all the word tokens in the sentence
            sbow = get_bow(disambiguator.disambiguate(sgraph.get_sentence()), stopwords)

            overlap = len(qbow & sbow)

            d_answers.append((overlap, sgraph))

        d_answers = sorted(d_answers, key=operator.itemgetter(0), reverse=True)

        d_answers_overlap = [num[0] for num in d_answers]
        d_max_overlap = max(d_answers_overlap)
        d_results = [val[1] for index, val in enumerate(answers) if val[0] == d_max_overlap]

        if len(d_results) == 0:
            return []
        elif len(d_results) == 1:
            return [orig_order[d_results[0]]]
       # elif len(d_results) > 1:
        else:
            return []




i = 3


def compare_baseline(qgraph, sgraphs):
    scores = {sgraph : 0 for sgraph in sgraphs}
    qWords = [x['word'].lower() for x in qgraph.nodes.values() if x['word'] and x['word'] not in stopwords]
    qLen = len(qWords)
    for sgraph in sgraphs:
        sWords = [x['word'].lower() for x in qgraph.nodes.values() if x['word'] and x['word'] not in stopwords]
        shared = [word for word in sWords if word in qWords]
        scores[sgraph] = len(shared)
        #     print ('SHARED: {0}'.format(' '.join([n['lemma'] for n in shared])))

    ret= sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return scores

def compare_clause_lemmas(qgraph, sgraphs, lemmas=False):
    scores = {sgraph : 0 for sgraph in sgraphs}
    qClause = qgraph.get_root_clause()
    if lemmas:
        for node in qClause:
            lemmatize_node(node)
        qWords= [x['lemma'].lower() for x in qClause if x['lemma']]
    else:
        qWords= [x['word'].lower() for x in qClause if x['word']]

    for sgraph in sgraphs:
        sClause = sgraph.get_root_clause()
        if lemmas:
            for node in sClause:
                lemmatize_node(node)
            sWords = [x['lemma'].lower() for x in sClause if x['lemma']]
        else:
            sWords = [x['word'].lower() for x in sClause if x['word']]
        shared = [word for word in sWords if word in qWords]
        scores[sgraph] = len(shared)

    ret= sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return scores

def subj_pred_disamb(qgraph, sgraphs):
    q_subj = qgraph.get_subject()
    q_pred = qgraph.get_predicate()
    if q_subj is not None:
        q_subj_hyper = disambiguator.get_noun_hypernym(q_subj['word'])
        q_pred_hyper = disambiguator.get_verb_hypernym(q_pred['word'])
        if q_subj_hyper and q_pred_hyper:
            i = 3
    i = 3

def compare_subj_pred(qgraph, sgraphs, disambig=False):
    orig_order = {sgraphs[i]:i for i in range(0, len(sgraphs))}

    scores = []
    q_subj = qgraph.get_subject()
    q_pred = qgraph.get_predicate()
    if disambig:
        if q_subj:
            q_subj = disambiguator.get_noun_hypernym(q_subj)
        if q_pred:
            q_pred = disambiguator.get_verb_hypernym(q_pred)
    lemmatize_node(q_subj)
    lemmatize_node(q_pred)
    snode = None
    for sgraph in sgraphs:
        s_subj = sgraph.get_subject()
        s_pred = sgraph.get_predicate()
        score = 0
        if q_subj and s_subj:
            if q_subj['word'] == s_subj['word']:
                if q_subj['rel'] == s_subj['rel']:
                    score = 3
                else:
                    score = 2
            else:
                lemmatize_node(s_subj)
                if q_subj['lemma'] == s_subj['lemma']:
                    if q_subj['rel'] == s_subj['rel']:
                       score = 2
                    else:
                        score = 1
        if q_pred and s_pred:
            if q_pred['word'] == s_pred['word']:
                if q_pred['rel'] == s_pred['rel']:
                    score += 3
                else:
                    score += 2
            else:
                lemmatize_node(s_pred)
                if q_pred['lemma'] == s_pred['lemma']:
                    if q_pred['rel'] == s_pred['rel']:
                        score += 2
                    else:
                        score += 1
        scores.append((score, sgraph))


    hi = max([entry[0] for entry in scores])
    answers = [orig_order[entry[1]] for entry in scores if entry[0] == hi]

    return answers

def compare_subj_pred_disambig(qgraph, sgraphs):
    orig_order = {sgraphs[i]:i for i in range(0, len(sgraphs))}

    scores = []
    q_subj = qgraph.get_subject()
    q_pred = qgraph.get_predicate()
    q_subj_hyper = disambiguator.get_noun_hypernym(q_subj['word']) if q_subj else None
    q_pred_hyper = disambiguator.get_verb_hypernym(q_pred['word']) if q_pred else None

    for sgraph in sgraphs:
        s_subj = sgraph.get_subject()
        s_subj_hyper = disambiguator.get_noun_hypernym(s_subj['word']) if s_subj else None
        s_pred = sgraph.get_predicate()
        s_pred_hyper = disambiguator.get_verb_hypernym(s_pred['word']) if s_pred else None

        score = 0
        if q_subj_hyper and s_subj_hyper:
            if s_subj_hyper == q_subj['word'] or s_subj['word'] == q_subj['word']:
                score += 3
            elif s_subj_hyper == q_subj_hyper:
                score += 1

        if q_pred_hyper and s_pred_hyper:
            if s_pred_hyper == q_pred['word'] or s_pred['word'] == q_pred_hyper:
                score += 3
            elif s_subj_hyper == q_subj_hyper:
                score += 1

        scores.append((score, sgraph))


    hi = max([entry[0] for entry in scores])
    answers = [orig_order[entry[1]] for entry in scores if entry[0] == hi]

    return answers

def find_answer(qgraph, sgraphs):

    qgraph = DepGraph(qgraph)
    sgraphs = [DepGraph(x) for x in sgraphs]
#    subj_pred_disamb(qgraph, sgraphs)


    answer = baseline_disambig(qgraph, sgraphs)
    if len(answer) == 1:
        return answer[0]
    else:
        answer2 = compare_subj_pred(qgraph, sgraphs)
        if len(answer2) == 1 and answer2[0] in answer:
            return answer2[0]
        elif len(answer2) > 1 and qgraph.print_sentence().split(' ')[0] == 'Why':
            sgs = [(idx, sgraphs[idx]) for idx in answer2]
            has_because = [sgraph for sgraph in sgs if re.search('because', sgraph[1].print_sentence())]
            if len(has_because) == 1:
                return has_because[0][0]
        else:
            answer3 = compare_subj_pred_disambig(qgraph, sgraphs)
            if len(answer3) == 1 and answer3[0] in answer2 and answer3[0] in answer:
                return answer3[0]
            in_all = list((set(answer3) & set(answer2) & set(answer)))
            if len(in_all) == 1:
                return in_all[0]
            in_both = list((set(answer3) & set(answer2)))
            if len(in_both) == 1:
                return in_both[0]
            else:
                return None

