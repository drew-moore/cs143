
# Uses dependency trees to

import re, sys, nltk, operator
from nltk.parse import DependencyGraph
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from DepGraphADT import DepGraph

lmtzr = WordNetLemmatizer()
# Read the lines of an individual dependency parse

stopwords = set(nltk.corpus.stopwords.words("english"))


def lemmatize_node(node):
    if node and 'tag' in node and node['tag'] != 'TOP':
        if node['tag'].startswith("V"):
            node['lemma'] = lmtzr.lemmatize(word=node['word'], pos='v')
            return node
        else:
            node['lemma'] = lmtzr.lemmatize(word=node['word'], pos='n')
            return node
    return None


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


def compare_subj_pred(qgraph, sgraphs):
    scores = {sgraph : 0 for sgraph in sgraphs}

    q_subj = qgraph.get_subject()
    q_pred = qgraph.get_predicate()
    lemmatize_node(q_subj)
    lemmatize_node(q_pred)
    snode = None
    for sgraph in sgraphs:
        s_subj = sgraph.get_subject()
        s_pred = sgraph.get_predicate()

        if q_subj and s_subj:
            if q_subj['word'] == s_subj['word']:
                if q_subj['rel'] == s_subj['rel']:
                    scores[sgraph] += 5
                else:
                    scores[sgraph] += 4
            else:
                lemmatize_node(s_subj)
                if q_subj['lemma'] == s_subj['lemma']:
                    if q_subj['rel'] == s_subj['rel']:
                        scores[sgraph] += 5
                    else:
                        scores[sgraph] += 4
        if q_pred and s_pred:
            if q_pred['word'] == s_pred['word']:
                if q_pred['rel'] == s_pred['rel']:
                    scores[sgraph] += 5
                else:
                    scores[sgraph] += 4
            else:
                lemmatize_node(s_pred)
                if q_pred['lemma'] == s_pred['lemma']:
                    if q_pred['rel'] == s_pred['rel']:
                        scores[sgraph] += 5
                    else:
                        scores[sgraph] += 4

    ret= sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return scores


def find_answer(qgraph, sgraphs):
    qgraph = DepGraph(qgraph)
    sgraphs = [DepGraph(x) for x in sgraphs]
    orig_order = {sgraphs[i]:i for i in range(0, len(sgraphs))}

    subj_pred = compare_subj_pred(qgraph, sgraphs)

    claus_comp = compare_clause_lemmas(qgraph, sgraphs, True)

    result = {sgraph:(subj_pred[sgraph] + claus_comp[sgraph]) for sgraph in sgraphs}

    ret = sorted(result.items(), key=lambda entry:entry[1], reverse=True)



    if ret[0][1] == ret[1][1] or ret[0][1] < 2:
        fallback = compare_baseline(qgraph, sgraphs)
        ret = {sgraph: (result[sgraph] + fallback[sgraph]) for sgraph in result}
        ret = sorted(ret.items(), key=lambda entry:entry[1], reverse=True)
        if ret[0][1] < 2 or ret[0][1] == ret[1][1] == ret[2][1]:
            return None
    answer = ret[0][0]
    #return index of resulting sgraph in the array passed in
    return orig_order[answer]
