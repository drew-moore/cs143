
# Uses dependency trees to

import re, sys, nltk, operator
from nltk.parse import DependencyGraph
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from DepGraphADT import DepGraph
import wordnet_demo as wnd
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

stopwords = set(nltk.corpus.stopwords.words("english"))

wnd.initialize()
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
        # A list of all the word tokens in the sentence
        sbow = get_bow(sent, stopwords)

        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)

        answers.append((overlap, sent))

    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
    #print(answers)
    '''
    # if no sentence has overlapped words, return the entire story
    if answers[0][0] == 0:
       best_answer = [sent for sentence in sentences for sent in sentence]
    # Return the best answer
    else:
       best_answer = (answers[0])[1]
    '''
    # return all tied overlapped sentences, including 0 overlapped case
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
        # A list of all the word tokens in the sentence
        sbow = get_bow(sgraph.get_sentence(), stopwords)

        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)

        answers.append((overlap, sgraph))

    d_answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
    #print(answers)
    '''
    # if no sentence has overlapped words, return the entire story
    if answers[0][0] == 0:
       best_answer = [sent for sentence in sentences for sent in sentence]
    # Return the best answer
    else:
       best_answer = (answers[0])[1]
    '''
    # return all tied overlapped sentences, including 0 overlapped case
    answers_overlap = [num[0] for num in answers]
    max_overlap = max(answers_overlap)
    results = [val[1] for index, val in enumerate(answers) if val[0] == max_overlap]

    if len(results) == 1:
        return orig_order[results[0]]
    else:
        qbow = get_bow(wnd.disambiguate(qgraph.get_sentence()), stopwords)
        d_answers = []
        for sgraph in sgraphs:
            # A list of all the word tokens in the sentence
            sbow = get_bow(wnd.disambiguate(sgraph.get_sentence()), stopwords)

            # Count the # of overlapping words between the Q and the A
            # & is the set intersection operator
            overlap = len(qbow & sbow)

            d_answers.append((overlap, sgraph))

        d_answers = sorted(d_answers, key=operator.itemgetter(0), reverse=True)
        #print(answers)
        '''
        # if no sentence has overlapped words, return the entire story
        if answers[0][0] == 0:
           best_answer = [sent for sentence in sentences for sent in sentence]
        # Return the best answer
        else:
           best_answer = (answers[0])[1]
        '''
        # return all tied overlapped sentences, including 0 overlapped case
        d_answers_overlap = [num[0] for num in answers]
        d_max_overlap = max(answers_overlap)
        d_results = [val[1] for index, val in enumerate(answers) if val[0] == max_overlap]

        if len(d_results) == 1:
            return orig_order[results[0]]
       # elif len(d_results) > 1:
        else:
            return None




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
        q_subj_hyper = wnd.get_noun_hypernym(q_subj['word'])
        q_pred_hyper = wnd.get_verb_hypernym(q_pred['word'])
        if q_subj_hyper and q_pred_hyper:
            i = 3
    i = 3

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
#    subj_pred_disamb(qgraph, sgraphs)


    ans = baseline_disambig(qgraph, sgraphs)


    # answer_idx = find_answer(question[1]['dep_parse'], q_deps)
    #
    # answer = baseline(qbow, q_sentences, stopwords)
    #
    # subj_pred = compare_subj_pred(qgraph, sgraphs)
    #
    # claus_comp = compare_clause_lemmas(qgraph, sgraphs, True)
    #
    # result = {sgraph:(subj_pred[sgraph] + claus_comp[sgraph]) for sgraph in sgraphs}
    #
    # ret = sorted(result.items(), key=lambda entry:entry[1], reverse=True)



    #return index of resulting sgraph in the array passed in
    return ans
