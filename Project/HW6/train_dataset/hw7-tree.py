#!/usr/bin/env python

import zipfile, argparse, os, nltk, operator, sys, re
from collections import defaultdict
import collections
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tree import Tree

###############################################################################
## Utility Functions ##########################################################
###############################################################################

def read_file(filename):
    fh = open(filename, 'r')
    text = fh.read()
    fh.close()

    return text

# This method takes as input the file extension of the set of files you want to open
# and processes the data accordingly
# Assumption: this python program is in the same directory as the training files
def getData(file_extension, given_filename):
    dataset_dict = {}

    # iterate through all the files in the current directory
    for filename in os.listdir("."):
        if filename.endswith(file_extension) and filename.startswith(given_filename):

            # get stories and cumulatively add them to the dataset_dict
            if file_extension == ".story" or file_extension == ".sch":
                dataset_dict[filename[0:len(filename)-len(file_extension)]] = open(filename, 'rU').read()

            # question and answer files and cumulatively add them to the dataset_dict
            elif file_extension == ".answers" or file_extension == ".questions":
                getQA(open(filename, 'rU', encoding="latin1"), dataset_dict)
                #getQA(open(filename, 'rU'), dataset_dict)

    return dataset_dict

# returns a dictionary where the question numbers are the key
# and its items are another dict of difficulty, question, type, and answer
# e.g. story_dict = {'fables-01-1': {'Difficulty': x, 'Question': y, 'Type':}, 'fables-01-2': {...}, ...}
def getQA(content, dataset_dict):
    qid = ""
    for line in content:
        if "QuestionID: " in line:
            qid = line[len("QuestionID: "):len(line)-1]
            # dataset_dict[qid] = defaultdict()
            dataset_dict[qid] = {}
        elif "Question: " in line: dataset_dict[qid]['Question'] = line[len("Question: "):len(line)-1]
        elif "Answer: " in line: dataset_dict[qid]['Answer'] = line[len("Answer:")+1:len(line)-1]
        elif "Difficulty: " in line: dataset_dict[qid]['Difficulty'] = line[len("Difficult: ")+1:len(line)-1]
        elif "Type: " in line: dataset_dict[qid]['Type'] = line[len("Type:")+1:len(line)-1]
    return dataset_dict

###############################################################################
## Question Answering Functions ###############################################
###############################################################################

def create_filename(parseCurrQID):
    currFileName = parseCurrQID[0] + "-" + parseCurrQID[1]

    currType = question[1]["Type"]

    if currType == "Story":
        currFileName = currFileName + ".story"
    else:
        currFileName = currFileName + ".sch"

    return  currFileName

def questionCasePicker(filename):
    trees = read_con_parses(filename)
    #print(trees)
    questionTypes = []
    for tree in trees:
        questionTypes.append(tree.leaves()[0])
    return questionTypes

###############################################################################
## Constituency Tree Functions ###############################################
###############################################################################

# Read the constituency parse from the line and construct the Tree
def read_con_parses(parfile):
    fh = open(parfile, 'r')
    lines = fh.readlines()
    fh.close()

    if parfile.endswith(".questions.par"):
       Q_index = [int(re.split('-', line)[2].strip()) for line in lines if line[0] == 'Q']
       Q_tree =  [Tree.fromstring(line) for line in lines if line[0] == '(']
       Q_dict = {}
       for i,j in zip(Q_index, Q_tree):
           Q_dict.update({i: j})
       Q_ordered = collections.OrderedDict(sorted(Q_dict.items()))
       return [val for key,val in Q_ordered.items()]
 
    else:
       return [Tree.fromstring(line) for line in lines if line[0] == '(']      

# See if our pattern matches the current root of the tree
def matches(pattern, root):
    # Base cases to exit our recursion
    # If both nodes are null we've matched everything so far
    if root is None and pattern is None:
        return root

    # We've matched everything in the pattern we're supposed to (we can ignore the extra
    # nodes in the main tree for now)
    elif pattern is None:
        return root

    # We still have something in our pattern, but there's nothing to match in the tree
    elif root is None:
        return None

    # A node in a tree can either be a string (if it is a leaf) or node
    plabel = pattern if isinstance(pattern, str) else pattern.label()
    rlabel = root if isinstance(root, str) else root.label()

    # If our pattern label is the * then match no matter what
    if plabel == "*":
        return root
    # Otherwise they labels need to match
    elif plabel == rlabel:
        # If there is a match we need to check that all the children match
        # Minor bug (what happens if the pattern has more children than the tree)
        for pchild, rchild in zip(pattern, root):
            match = matches(pchild, rchild)
            if match is None:
                return None

        return root

    return None

def pattern_matcher(pattern, tree):

    t = []
    for subtree in tree.subtrees():
        node = matches(pattern, subtree)

        if node is not None:
            t.append(node)
    return t
    return None

def responseTree(par_file, sentenceNum, questionCase):
    trees = read_con_parses(par_file)

    tree = trees[sentenceNum]
    
    #TODO add in cases patterns here
    '''
    if questionCase == "Who":
        #nltk.ParentedTree.fromstring("(VP (*) (PP))")
        return("".join(tree.leaves()))
    elif questionCase == "Where":
        pattern = nltk.ParentedTree.fromstring("(VP (*) (PP))")
        subtree = pattern_matcher(pattern, tree)
        pattern = nltk.ParentedTree.fromstring("(PP)")
        #subtree2 = pattern_matcher(pattern, subtree)
        #print(subtree)
        answer = ""
        for leaf in subtree:
            answer = " ".join(leaf.leaves())
        return answer
    elif questionCase == "What":
        #nltk.ParentedTree.fromstring("(VP (*) (PP))")
        return("".join(tree.leaves()))   
    elif questionCase == "Why":
        #nltk.ParentedTree.fromstring("(VP (*) (PP))")
        return("".join(tree.leaves()))
    elif questionCase == "How":
        #nltk.ParentedTree.fromstring("(VP (*) (PP))")
        return("".join(tree.leaves()))
    '''
    return (" ".join(tree.leaves()))


###############################################################################
## Added Baseline Functions ###################################################
###############################################################################


# The standard NLTK pipeline for POS tagging a document
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]

    sentences = [nltk.pos_tag(sent) for sent in sentences]

    return sentences

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

# ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'

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

    #return set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])

def find_phrase(tagged_tokens, qbow):
    for i in range(len(tagged_tokens) - 1, 0, -1):
        word = (tagged_tokens[i])[0]
        if word in qbow:
            return tagged_tokens[i+1:]

# qtokens: is a list of pos tagged question tokens with SW removed
# sentences: is a list of pos tagged story sentences
# stopwords is a set of stopwords
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

def sentMacher(answer, text):
    token_answer = nltk.word_tokenize(answer)

    sentences = nltk.sent_tokenize(text)
    token_text = []
    index = 0
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        overlap = len(set(token_answer) & set(words))
        token_text.append((overlap, sent, index))
        index += 1

    match = sorted(token_text, key=operator.itemgetter(0), reverse=True)
    
    sent = match[0][1]
    index = match[0][2]
    #print(sent)
    #print(index)
    return sent, index


# wrote this in a way that it should generalize to be usable with the next two assignments, assuming
# they'll continue to want the results sorted numerically by story_id then by question_id. We just have to make sure
# to pass the arrays in in the order they want, as the order they want those appears random so far
def write_results(dicts, file):
    for dic in dicts:
        arrs = [entry.split('-') for entry in dic]

        for story_type in set([arr[0] for arr in arrs]):
            for story_id in sorted(set([arr[1] for arr in arrs if arr[0] == story_type])):
                questions = sorted([arr for arr in arrs if arr[0] == story_type and arr[1] == story_id], key=lambda arr:int(arr[2]))
                questions = ['-'.join(arr) for arr in questions]
                responses = [(question, dic[question]) for question in questions]
                file.write('\n\n'.join(['\n'.join(['QuestionID: {0}'.format(response[0]), 'Answer:{0}'.format(response[1])])for response in responses]))
                file.write('\n\n')

###############################################################################
## Program Entry Point ########################################################
###############################################################################
if __name__ == '__main__':
    # optional functions for opening and organizing some of the data
    # if you do not understand how the data is being returned,
    # you can write your own methods; these are to help you get started

    filename = sys.argv[1]
    file = open("train_my_answers.txt", 'w', encoding="utf-8").close()
    file = open("train_my_answers.txt", 'a', encoding="utf-8")

    filesToParse = read_file(filename)
    filesList = filesToParse.split('\n')

    for fileItem in filesList:
        stories = getData(".story", fileItem) # returns a list of stories
        sch = getData(".sch", fileItem) # returns a list of scheherazade realizations
        questions = getData(".questions", fileItem) # returns a dict of questionIds
        answers = getData(".answers", fileItem) # returns a dict of questionIds
        questionTypes = questionCasePicker(fileItem + ".questions.par")
        #print(questionTypes)

        stopwords = set(nltk.corpus.stopwords.words("english"))

        outputDictFables = {}
        outputDictBlogs = {}

        index = 0
        for question in questions.items():
            parseCurrQID = question[0].split("-")
            currFileName = create_filename(parseCurrQID)
            currQ = question[1]["Question"]
            text = read_file(currFileName)

            qbow = get_bow(get_sentences(currQ)[0], stopwords)

            sentences = get_sentences(text)
            answer = baseline(qbow, sentences, stopwords)
            #print(answer)
            if answer == "":
               finalAnswer = ""
            else:
               finalAnswer = " ".join(t[0] for t in answer)
               sent, index_sent = sentMacher(finalAnswer, text)
               #print(sent)              
               #finalAnswer = sent
               finalAnswer = responseTree(currFileName+".par", index_sent, questionTypes[index])
            #print(finalAnswer)
            index = index + 1
            
            
            if parseCurrQID[0] == "fables":
                outputDictFables.update({question[0]:finalAnswer})
            else:
                outputDictBlogs.update({question[0]:finalAnswer})

    # read in other data, ".story.par", "story.dep", ".sch.par", ".sch.dep", ".questions.par", ".questions.dep"

        write_results([outputDictFables, outputDictBlogs], file)

    file.close()


    # create methods to perform information extraction and question and answering
