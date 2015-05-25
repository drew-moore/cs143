#!/usr/bin/env python
'''
Created on May 14, 2014

@author: reid
'''

import re, sys, nltk, operator
from nltk.parse import DependencyGraph, ProjectiveDependencyParser as pdp
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.parse import stanford
lmtzr = WordNetLemmatizer()
# Read the lines of an individual dependency parse
def read_dep(fh):
    dep_lines = []
    for line in fh:
        line = line.strip()
        if len(line) == 0:
            return "\n".join(dep_lines)
        elif re.match(r"^QuestionId:\s+(.*)$", line):
            # You would want to get the question id here and store it with the parse
            continue
        dep_lines.append(line)

    return "\n".join(dep_lines) if len(dep_lines) > 0 else None


# Read the dependency parses from a file
def read_dep_parses(depfile):
    fh = open(depfile, 'r')

    # list to store the results
    graphs = []

    # Read the lines containing the first parse.
    dep = read_dep(fh)
#     print(dep)
#     graph = DependencyGraph("""There   EX      3       expl
# once    RB      3       advmod
# was     VBD     0       ROOT
# a       DT      5       det
# crow    NN      3       nsubj""")
#     graphs.append(graph)

    # While there are more lines:
    # 1) create the DependencyGraph
    # 2) add it to our list
    # 3) try again until we're done
    while dep is not None:
        graph = DependencyGraph(dep)
        graphs.append(graph)

        dep = read_dep(fh)
    fh.close()

    return graphs

def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'ROOT':
            return node
    return None

def find_node(word, graph):
    for node in graph.nodes.values():
        if node["word"] == word:
            return node
    return None



def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph)

    return results


def find_answer(qgraph, sgraph):
    qmain = find_main(qgraph)
    qword = qmain["word"]

    snode = find_node(qword, sgraph)

    for node in sgraph.nodes.values():
        #print("node in nodelist:", node)
        if node.get('head', None) == snode["address"]:
            print("Our parent is:", snode)
            print("Our relation is:", node['rel'])
            if node['rel'] == "prep":
                deps = get_dependents(node, sgraph)
                deps = sorted(deps, key=operator.itemgetter("address"))

                return " ".join(dep["word"] for dep in deps)

# def build_clauses(graph):
#     root_word = graph.root['word']
#     root_add = graph.root['address']
#     ret = []
#     curr_head = root_add
#     for node in [graph.nodes[node] for node in graph.nodes if graph.nodes[node]['tag'] != 'TOP']:
#         if node['head'] == root_add or node['head'] == 0:
#             ret.append(node['word'])
#             curr_head = node['address']
#     for dep in graph.nodes[curr_head]['deps']:
#         if dep in ['pobj']: #TODO
#             idx = graph.nodes[curr_head]['deps'][dep][0]
#             ret.append(graph.nodes[idx]['word'])
#             i = 9
#     return ret
#
# def build_key_clause(graph):
#
#     root_add = graph.root['address']
#     ret = []
#     deps = [(dep, graph.root['deps'][dep][0]) for dep in graph.root['deps'] if dep in ['pobj', 'dobj', 'psubj', 'nsubj', 'advmod', 'prep']]
#     srt = sorted(deps, key=lambda dep:dep[1])
#     for dep in srt:
#         if dep[1] > root_add:
#             ret.append((graph.root['word'],graph.root['ctag']))
#         ret.append((graph.nodes[dep[1]]['word'], graph.nodes[dep[1]]['ctag']))
#         if dep[0] == 'prep':
#             med = graph.nodes[dep[1]]
#             if 'pobj' in med['deps']:
#                 target_idx = med['deps']['pobj'][0]
#             elif 'psubj' in med['deps']:
#                 target_idx = med['deps']['psubj'][0]
#             if target_idx:
#                 ret.append((graph.nodes[target_idx]['word'], graph.nodes[target_idx]['ctag']))
#     if len(deps) > 0 and root_add > max([dep[1] for dep in deps]):
#         ret.append((graph.root['word'],graph.root['ctag']))
#     return ret
#
#
# def get_key_nodes(graph):
#     deps = get_dependents(graph.root, graph)
#     i = 3
#
# def try_root_match(qgraph, sgraphs):
#     matches = []
#     q_root = qgraph.root['word']
#     q_tag = qgraph.root['tag']
#     for sgraph in sgraphs:
#         if sgraph.root['word'] == q_root :
#             matches.append(sgraph)
#         # TODO - handle cases where multiple sentencor note graphs share the same root
#     if len(matches) > 0:
#         return matches
#         #prefer matches without lemmatizing. Only if we don't find one do we lemmatize
#     if len(matches) == 0:
#         q_lemma = lmtzr.lemmatize(q_root, 'v' if q_tag.startswith('v') else 'n')
#         for sgraph in sgraphs:
#             if sgraph.root['word'] == q_lemma:
#                 matches.append(sgraph)
#     if len(matches) > 0:
#         return matches
#
#     for sgraph in sgraphs:
#         s_lemma =  lmtzr.lemmatize(sgraph.root['word'], 'v' if sgraph.root['tag'].startswith('v') else 'n')
#         if s_lemma == q_lemma:
#             matches.append(sgraph)
#     return matches
#
# def try_clause(qgraph, sgraphs):
#     q_clause = build_key_clause(qgraph)
#     for sg in sgraphs:
#         s_clause = build_key_clause(sg)
#         i = 3
#
# def find_sentence(qgraph, sgraphs):
#     root_matches = try_root_match(qgraph, sgraphs)
#     if len(root_matches) == 1:
#         return root_matches[0]
#         #occams razor - prefer the simplest solution: if there's exactly one sentence that has the root word in it, return that
#         clause_matches = try_clause(qgraph, sgraphs)
#         i = 4
def get_node_at_address(address, nodes):
    if type(nodes) is list:
        for node in nodes:
            if node['address'] == address:
                return node
        return None

def get_root_node(nodes):
    for node in nodes:
        if node['rel'] == 'ROOT':
            return node
    return None
def get_subj_pred(graph):
    if graph.root['tag'].startswith('V'):
        match_v = graph.root['lemma']
        if 'nsubj' in graph.root['deps']:
            match_n = get_node_at_address(graph.root['deps']['nsubj'][0], q)
            match_v = graph.root
            return {'subj':match_n, 'pred':match_v}
            #TODO handle noun roots

def get_subject(graph):
    if graph.root['tag'].startswith('V'):
        for node in graph.nodes.values():
            if node['rel'] == 'nsubj' and node['head'] == graph.root['address']:
                return node
    else:
        return graph.root

def get_predicate(graph):
    if graph.root['tag'].startswith('V'):
        return graph.root
    else:
        #TODO
        i = 4
        #root is noun
        #for node in nodes, if nsubj in node.deps == address of root, return it
        #if 'nsubj' in graph.root['deps']:
        return None

def lemmatize_node(node):
    if node:
        if node['tag'].startswith("V"):
            return lmtzr.lemmatize(word=node['word'], pos='v')
        else:
            return lmtzr.lemmatize(word=node['word'], pos='n')
    return None

def get_core_clause(graph):
    root = graph.root
    result = []
    root['lemma'] = lemmatize_node(root)
    deps = get_dependents(root, graph)
    result.append(root)
    for dep in deps:
        if dep['address'] not in [x['address'] for x in result] and dep['tag'] != 'DT':
            dep['lemma'] = lemmatize_node(dep)
            result.append(dep)
            if dep['rel'] == 'prep':
                if 'pobj' in dep['deps']:
                    node = get_node_at_address(dep['deps']['pobj'][0], graph.nodes.values())
                    if node:
                        node['lemma'] = lemmatize_node(node)
                        result.append(node)
    result = sorted(result, key=operator.itemgetter("address"))
    return result




#Intuition here is to get the main clause, lemmatize it, and compare the lemmatized subjects and predicates:
# If the root is a verb, look at the POS tag - if a sentence contains
# Crow have in beak matches was in beak of crow
def get_sentence(qgraph, sgraphs):
    q = get_core_clause(qgraph)
    q_subj = get_subject(qgraph)
    q_pred = get_predicate(qgraph)
    priorities = {i: 0 for i in range (0, len(sgraphs))}
    for i in range (0, len(sgraphs)):
        priority = 0
        s = get_core_clause(sgraphs[i])
        s_subj = get_subject(sgraphs[i])
        s_pred = get_predicate(sgraphs[i])
        if not s_subj and s_pred:
            continue

        if s_subj['word'] == q_subj['word']:
            priority += 4
        if s_subj['lemma'] == q_subj['lemma']:
            priority += 1
        if s_pred['word'] == q_pred['word']:
            priority += 4
        if s_pred['lemma'] == q_pred['lemma']:
            priority += 2
        priorities[i] = priority
        i  = 4
    results = sorted(priorities.items(), key=lambda item:item[1])
    idx = results[0][0]
    return sgraphs[idx]

if __name__ == '__main__':
    text_file = "fables-01.sch"
    dep_file = "fables-01.sch.dep"
    q_file = "fables-01.questions.dep"

    # Read the dependency graphs into a list
    sgraphs = read_dep_parses(dep_file)
    qgraphs = read_dep_parses(q_file)
    # for graph in qgraphs:
    #     g = build_key_clause(graph)
    #     print(' '.join(['{0},{1}'.format(x[0], x[1]) for x in build_key_clause(graph)]))



    for qgraph in qgraphs:

        sgraph = get_sentence(qgraph, sgraphs)
        answer = find_answer(qgraph, sgraph)
        o = 0
        #
        # for node in sgraph.nodes.values():
        #     tag = node["tag"]
        #     word = node["word"]
        #     if word is not None:
        #         if tag.startswith("V"):
        #             print(lmtzr.lemmatize(word, 'v'))
        #         else:
        #             print(lmtzr.lemmatize(word, 'n'))
        # print()

   #     answer = find_answer(qgraph, sgraph)
    #    print(answer)
