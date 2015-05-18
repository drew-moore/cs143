#!/usr/bin/env python

import zipfile, argparse, os, nltk, operator
from collections import defaultdict

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
def getData(file_extension):
    dataset_dict = {}

    # iterate through all the files in the current directory
    for filename in os.listdir("."):
        if filename.endswith(file_extension):

            # get stories and cumulatively add them to the dataset_dict
            if file_extension == ".story" or file_extension == ".sch":
                dataset_dict[filename[0:len(filename)-len(file_extension)]] = open(filename, 'rU').read()

            # question and answer files and cumulatively add them to the dataset_dict
            elif file_extension == ".answers" or file_extension == ".questions":
                getQA(open(filename, 'rU'), dataset_dict)

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

###############################################################################
## Added Baseline Functions ###################################################
###############################################################################


# The standard NLTK pipeline for POS tagging a document
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    return sentences

def get_bow(tagged_tokens, stopwords):
	return set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])

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

    # Return the best answer
    best_answer = (answers[0])[1]
    return best_answer



###############################################################################
## Program Entry Point ########################################################
###############################################################################
if __name__ == '__main__':
    # optional functions for opening and organizing some of the data
    # if you do not understand how the data is being returned,
    # you can write your own methods; these are to help you get started
    stories = getData(".story") # returns a list of stories
    sch = getData(".sch") # returns a list of scheherazade realizations
    questions = getData(".questions") # returns a dict of questionIds
    answers = getData(".answers") # returns a dict of questionIds

    file = open("response_file.txt", 'w', encoding="utf-8")

    stopwords = set(nltk.corpus.stopwords.words("english"))

    outputDictFables = {}
    outputDictBlogs = {}

    for question in questions.items():

        parseCurrQID = question[0].split("-")
        currFileName = create_filename(parseCurrQID)
        currQ = question[1]["Question"]
        text = read_file(currFileName)

        qbow = get_bow(get_sentences(currQ)[0], stopwords)
        sentences = get_sentences(text)

        answer = baseline(qbow, sentences, stopwords)
        finalAnswer = " ".join(t[0] for t in answer if t not in stopwords)

        if parseCurrQID[0] == "fables":
            outputDictFables.update({question[0]:finalAnswer})
        else:
            outputDictBlogs.update({question[0]:finalAnswer})

    # read in other data, ".story.par", "story.dep", ".sch.par", ".sch.dep", ".questions.par", ".questions.dep"

    # outputDict = sorted(outputDictFables.items(),key=lambda
    #          item: item[0].split("-")[1])

    outputDict = sorted(outputDictFables.items())
    for response in outputDict:
        file.write("QuestionID: " + response[0] + "\n"
                    "Answer: " + response[1] + "\n\n")

    outputDict = sorted(outputDictBlogs.items())
    for response in outputDict:
        file.write("QuestionID: " + response[0] + "\n"
                    "Answer: " + response[1] + "\n\n")

    # create methods to perform information extraction and question and answering
