import re, sys, nltk, operator

class DepGraph:
    stopwords = set(nltk.corpus.stopwords.words("english"))
    def __init__(self, graph):
        self.graph = graph
        self.root = graph.root
        self.nodes = graph.nodes
        self.root_type = 'V' if graph.root['tag'].startswith('V') else 'N'
        self.root_clause = self.get_root_clause()

    def print_sentence(self):
        ret = ' '.join([node['word'] for node in self.nodes.values() if 'word' in node and node['word']])
     #   print(ret)
        return ret
    def get_root_clause(self):
        ret = []

        subj = self.get_subject()
        pred = self.get_predicate()

        if subj:
            self.add_to_result(ret, self.get_dependents(subj, recurse=False))
            self.add_to_result(ret, self.get_nodes_with_dep_to(subj, recurse=False))
        if pred:
            self.add_to_result(ret, self.get_dependents(pred, recurse=False))
            self.add_to_result(ret, self.get_nodes_with_dep_to(pred, recurse=False))




            #       ret.append(self.root)
        # sent = self.print_sentence()
        # print(sent)
        # subroots = self.get_nodes_with_head(self.root['address'])

        ret = [node for node in ret if node['tag'] != 'DT']
        ret = sorted(ret, key=operator.itemgetter("address"))

        return ret


    def print_node_edges(self, node):
        print('{0}'.format(node['word']))
        for deptype in node['deps']:
            for idx in node['deps'][deptype]:
                rel = self.node_at_idx(idx)
                print('  ->{0} : {1}'.format(deptype, rel['word']))
        for n in self.nodes.values():
            for deptype in n['deps']:
                if node['address'] in n['deps'][deptype]:
                    print(' <-{0} : {1}'.format(deptype, n['word']))

    def add_to_result(self, ret, nodes):
        if not nodes:
            return
        if not isinstance(nodes, (list)):
            nodes = [nodes]
        for node in nodes:
            add = True
            for x in ret:
                if x['address'] == node['address']:
                    add = False
                    break
            if add:
                ret.append(node)

    def get_nouns_verbs(self):
        ret = [node for node in self.nodes.values() if 'tag' in node and node['tag'].startswith('N') or node['tag'].startswith('V')]
        return ret

    def get_subject(self):
        if self.root['tag'].startswith('V'):
            for node in self.nodes.values():
                if 'rel' in node and node['rel'] == 'nsubj' and node['head'] == self.root['address']:
                    return node
                if 'rel' in node and node['rel'] == 'nsubjpass' and node['head'] == self.root['address']:
                    return node
        else:
            return self.root

    def get_predicate(self):
        if self.root['tag'].startswith('V'):
            return self.root
        else:
            #TODO
            i = 4
            #root is noun
            #for node in nodes, if nsubj in node.deps == address of root, return it
            #if 'nsubj' in graph.root['deps']:
            return None

    def get_dependents(self, node, recurse=True):
        results = []
        for deptype in node["deps"]:
            for idx in node['deps'][deptype]:
                dep = self.node_at_idx(idx)
                results.append(dep)
                if recurse:
                    results += self.get_dependents(dep)
        return results

    def get_nodes_with_dep_to(self, node, recurse=True):
        ret = []
        idx = node['address']
        for n in self.nodes.values():
            for deptype in n['deps']:
                if idx in n['deps'][deptype]:
                    ret.append(n)
                    if recurse:
                        ret.append(self.get_nodes_with_dep_to(n))
        return ret

    def get_clause(self, node=None):
        ret = []
        if node is None:
            node = self.root

        ret.append(node)

        rels = self.get_key_rels(node)
        if rels:
            for reltype, rel in rels:
                # if reltype == 'case':
                #     i = 3
                #     if rel['address'] in rels:
                #         ret.append(rel)
                #
                #     if rel['head'] in rels:
                #         ret.append(self.node_at_idx(rel['head']))
                #         curr = self.node_at_idx(rel['head'])
                #         ho = self.get_heads_of(curr)
                #         for n in ho:
                #             if self.rel_between(curr, n):
                #                 case = self.node_has_case(n)
                #                 if case:
                #                     ret.append(case)
                #                 ret.append(n)
                #         return ret
                if reltype == 'cc':
                    heads = self.get_nodes_with_head(rel['address'])
                    head = self.node_at_idx(rel['head'])
                    x = 3
                elif reltype == 'ccomp':
                    x = 3
                else:
                    ret.append(rel)
                if reltype in ['VB', 'VBD', 'case', 'IN', 'TO', 'conj']:
                    ret.append(self.get_clause(rel))
        return ret

    def get_key_rels(self, node=None):
        if node is None:
            node = self.root
        ret = []
        for type in ['nsubj', 'dobj', 'nobj', 'nmod', 'acl', 'compound', 'advmod', 'IN', 'mark']:
            if type in node['deps']:
                for idx in node['deps'][type]:
                    ret.append((type, self.node_at_idx(idx)))

        rels_to = self.get_nodes_with_head(node)
        for type in ['acl', 'compound']:
                p = 3

        return ret


    def rel_between(self, node0, node1, directed=True):
        if node1['address'] in self.rels_to(node0):
            for type in node0['deps']:
                for idx in node0['deps'][type]:
                    if idx == node1['address']:
                        return type
        return False

    def get_nodes_with_head(self, idx):
        ret = []
        for node in self.nodes.values():
            if 'head' in node and node['head'] == idx:
                ret.append(node)
        return ret



    def get_root_clause_text(self):
        if self.root_clause:
            return " ".join(node['word'] for node in self.root_clause)


    def node_has_case(self, node):
        if 'case' in self.dep_types(node):
            return self.node_at_idx(node['deps']['case'][0])

    def get_heads_of(self, node, subsequent=False):
        ret = [x for x in self.graph.nodes.values() if 'head' in x and x['head'] == node['address']]
        if len(ret) > 0:
            if subsequent:
                return [x for x in ret if x['address'] > node['address']]
            else:
                return ret
        else:
            return False

    def graph_has(self, type):
        ret = []
        for node in self.graph.nodes.values():
            if 'rel' in node and node['rel'] == type:
                ret.append(node)
        if len(ret) == 0:
            return False
        else:
            return ret

    def rels_to(self, node):
        return [rel for type in node['deps'] for rel in node['deps'][type]]
    def dep_types(self, node):
        return [rel for rel in node['deps']]

    def node_at_idx(self, idx):
        for node in self.graph.nodes.values():
            if node['address'] == idx:
                return node
        return False