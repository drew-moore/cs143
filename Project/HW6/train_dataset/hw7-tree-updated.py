
import zipfile, argparse, os, nltk, operator, sys, re
from collections import defaultdict
import collections
from dep_analyzer import find_answer

from nltk.parse import DependencyGraph
from nltk.tree import *

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
    print(currFileName) 
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
    #return [Tree.fromstring(line) for line in lines if line[0] == '(']    
    ### change it to ParentedTree ###  
    return [ParentedTree.fromstring(line) for line in lines if line[0] == '(']

def read_q_dep(fh):
    dep_lines = []
    id = None
    for line in fh:
        if re.match(r"^QuestionId:\s+(.*)$", line):
            # You would want to get the question id here and store it with the parse
            id = line.split(': ')[1].split('\n')[0]
            break
    for line in fh:
        line = line.strip()
        if len(line) == 0:
            return (id,update_inconsistent_tags("\n".join(dep_lines)))
        elif re.match(r"^QuestionId:\s+(.*)$", line):
            continue
        dep_lines.append(line)

    if len(dep_lines) > 0:
        return (id,update_inconsistent_tags("\n".join(dep_lines)))
    else:
        return None

# Read the lines of an individual dependency parse
def read_dep(fh):
    dep_lines = []
    for line in fh:
        line = line.strip()
        if len(line) == 0:
            return update_inconsistent_tags("\n".join(dep_lines))
        elif re.match(r"^QuestionId:\s+(.*)$", line):
            # You would want to get the question id here and store it with the parse
            continue
        dep_lines.append(line)
    if len(dep_lines) > 0:
        return update_inconsistent_tags("\n".join(dep_lines))
    else:
        return None

# Note: the dependency tags return by Stanford Parser are slightly different than
# what NLTK expects. We tried to change all of them, but in case we missed any, this
# method should correct them for you.
def update_inconsistent_tags(old):
    return old.replace("root", "ROOT")

# Read the dependency parses from a file
def read_dep_parses(depfile, q=False):
    fh = open(depfile, 'r')

    # list to store the results
    if q:
        graphs = {}
    else:
        graphs = []

    # Read the lines containing the first parse.
    if q:
        dep = read_q_dep(fh)
    else:
        dep = read_dep(fh)

    # While there are more lines:
    # 1) create the DependencyGraph
    # 2) add it to our list
    # 3) try again until we're done
    while dep is not None:

        if q:
            graph = DependencyGraph(dep[1])
            graphs[dep[0]] = graph
            dep = read_q_dep(fh)
        else:
            graph = DependencyGraph(dep)
            graphs.append(graph)
            dep = read_dep(fh)
    fh.close()

    return graphs

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
'''
def pattern_matcher(pattern, tree):
    t = []
    for subtree in tree.subtrees():
        node = matches(pattern, subtree)

        if node is not None:
            t.append(node)
    return t
    return None

'''
def pattern_matcher(pattern, tree):
    for subtree in tree.subtrees():
        node = matches(pattern, subtree)
        if node is not None:
            return node
    return None

def who_question(tree, question_par):
    pattern = ParentedTree.fromstring('(SQ)')
    subtree_qn = pattern_matcher(pattern, question_par)
    subtree_qn_leaves = subtree_qn.leaves() 

    subtree1 = None
    while subtree1 == None:
        for subtree in tree.subtrees():
            if len(subtree.leaves()) >= len(subtree_qn_leaves):
               if subtree.leaves()[:len(subtree_qn_leaves)] == subtree_qn_leaves:
                  subtree1 = subtree
                  break
        subtree_qn_leaves = subtree_qn_leaves[1:] 
    #print(subtree1)
    #print(subtree1.left_sibling())
    pattern = ParentedTree.fromstring('(NP)')
    
    subling = pattern_matcher(pattern, subtree1.left_sibling())
    if subling is not None:
       final = subling
    else:    
       subtree2 = subtree1.parent()
       subling2 = pattern_matcher(pattern, subtree2.left_sibling())
       final = subling2
    return(" ".join(final.leaves()))


def where_question(tree, question_par):
    #pattern = nltk.ParentedTree.fromstring("(VP (*) (PP))")
    #subtree = pattern_matcher(pattern, tree)
    pattern = nltk.ParentedTree.fromstring("(PP)")
    #subtree2 = pattern_matcher(pattern, subtree)
    subtree2 = pattern_matcher(pattern, tree)
    return(" ".join(subtree2.leaves()))
     

def responseTree(par_file, sentenceNum, questionCase, question_par):
    trees = read_con_parses(par_file)
    #print(sentenceNum)
    #print(questionCase)
    #print(question_par)

    tree = trees[sentenceNum]
    #print(tree)
    #TODO add in cases patterns here
    
    if questionCase == "Who":
        try:
            return who_question(tree, question_par)
        except AttributeError:   
            return(" ".join(tree.leaves())) 

    elif questionCase == "Where":
        ### need to find the right sentence, and pattern ### 
        try:
            return where_question(tree, question_par)
        except AttributeError:      
            return(" ".join(tree.leaves()))   

    elif questionCase == "What":
        ### different kinds of what questions ### 
        #nltk.ParentedTree.fromstring("(VP (*) (PP))")
        return(" ".join(tree.leaves()))   
    elif questionCase == "Why":
        ### because & in order for, see the parser for more details ###

        #nltk.ParentedTree.fromstring("(VP (*) (PP))")
        return(" ".join(tree.leaves()))
    elif questionCase == "How":
        ### only one question: fables-01-13 ###
        #nltk.ParentedTree.fromstring("(VP (*) (PP))")
        return(" ".join(tree.leaves()))
    else:
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


# ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'

    #return set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])

def find_phrase(tagged_tokens, qbow):
    for i in range(len(tagged_tokens) - 1, 0, -1):
        word = (tagged_tokens[i])[0]
        if word in qbow:
            return tagged_tokens[i+1:]

# qtokens: is a list of pos tagged question tokens with SW removed
# sentences: is a list of pos tagged story sentences
# stopwords is a set of stopwords

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
    file = open("answers-bl.txt", 'w', encoding="utf-8").close()
    file = open("answers-bl.txt", 'a', encoding="utf-8")

    filesToParse = read_file(filename)
    filesList = filesToParse.split('\n')
    outputDictFables = {}
    outputDictBlogs = {}
    for fileItem in filesList:

        stories = getData(".story", fileItem) # returns a list of stories
        sch = getData(".sch", fileItem) # returns a list of scheherazade realizations

        sch_text = read_file(fileItem + '.sch')
        story_text = read_file(fileItem + '.story')

        sch_sentences = get_sentences(sch_text)
        sch_consts = read_con_parses(fileItem + '.sch.par')
        sch_deps = read_dep_parses(fileItem + '.sch.dep')
        sch_parses = [{'sentence':sch_sentences[i], 'const':sch_consts[i], 'dep':sch_deps[i] } for i in range(0, len(sch_sentences))]

        story_sentences = get_sentences(story_text)
        story_consts = read_con_parses(fileItem + '.story.par')
        story_deps = read_dep_parses(fileItem + '.story.dep')
        try:
            story_parses = [{'sentence':story_sentences[i], 'const':story_consts[i], 'dep':story_deps[i]} for i in range(0, len(story_sentences))]
        except:
            story_parses = [{'sentence':story_sentences[i], 'const':story_consts[i-1], 'dep':story_deps[i-1]} for i in range(0, len(story_sentences))]
            pass
        questions = getData(".questions", fileItem) # returns a dict of questionIds
        questions = collections.OrderedDict(sorted(questions.items()))

        answers = getData(".answers", fileItem) # returns a dict of questionIds

        questionTypes = questionCasePicker(fileItem + ".questions.par")
        question_par = read_con_parses(fileItem + ".questions.par")
        #print(questionTypes)

        stopwords = set(nltk.corpus.stopwords.words("english"))



        index = 0

        q_deps = read_dep_parses(fileItem + '.questions.dep', q=True)
        for question in questions:
            questions[question]['dep_parse'] = q_deps[question]

        for question in questions.items():
            #print(question)
            #print(question_par[index]) 

            if question[1]['Type'] == 'Sch':
                q_sentences = sch_sentences
                q_deps = sch_deps
                q_text = sch_text
            else:
                q_sentences = story_sentences
                q_deps = story_deps
                q_text = story_text

            parseCurrQID = question[0].split("-")
            currFileName = create_filename(parseCurrQID)
            print(parseCurrQID)
            currQ = question[1]["Question"]

            answer_idx = find_answer(question[1]['dep_parse'], q_deps)

            print(currQ)
            #print(answer)
            if answer_idx is None:
               finalAnswer = ""
               print('no answer')
            else:
                answer = q_sentences[answer_idx]
                finalAnswer = " ".join(t[0] for t in answer)
                print(finalAnswer)
                sent, index_sent = sentMacher(finalAnswer, q_text)
                #print(sent)
                #finalAnswer = sent
                finalAnswer = responseTree(currFileName+".par", index_sent, questionTypes[index], question_par[index])
            #print(finalAnswer)
            index = index + 1
            
            if parseCurrQID[1] == '5':
                i = 43
            if parseCurrQID[0] == "fables":
                outputDictFables.update({question[0]:finalAnswer})
            else:
                outputDictBlogs.update({question[0]:finalAnswer})

    # read in other data, ".story.par", "story.dep", ".sch.par", ".sch.dep", ".questions.par", ".questions.dep"

    write_results([outputDictBlogs, outputDictFables], file)

    file.close()


    # create methods to perform information extraction and question and answering
