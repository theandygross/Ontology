__author__ = 'agross'

import pandas as pd


class Node(object):
    """
    Represents an internal node in a ontology.

    Defines a series of helper methods for returning
    features of the ontology.
    """
    def __init__(self, node_id, name, attributes={}):
        self.id = node_id
        self.name = name
        self.attributes = attributes
        self.ontology = None
        self.parents = None
        self.children = None
        self.descendents = None
        self.genes = None

    def get_parents(self):
        if self.get_parents is None:
            self.parents = self.ontology.get_parents(self)
        return self.parents

    def get_children(self):
        if self.children is None:
            self.children = self.ontology.get_children(self)
        return self.children

    def get_descendents(self):
        if self.descendents is None:
            children = self.get_children()
            if len(children) == 0:
                return pd.Series({self.id: self})
            else:
                descendents = pd.concat([c.get_descendents() for c
                                         in children])
                descendents[self.id] = self
            self.descendents = descendents
        return self.descendents

    def get_genes(self):
        if self.genes is None:
            self.genes = self.ontology.get_genes(self)
        return self.genes

    def get_enrichment(self, hit_vec):
        return self.ontology.get_enrichment(self, hit_vec)

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.id == other.id


class Edge(object):
    """
    Represents a parent-child relationship between terms in an ontology.
    """
    def __init__(self, parent, child):
        self.parent = parent
        self.child = child

    def __repr__(self):
        s = '''{} -->  {}'''.format(self.parent.name, self.child.name)
        return s

    def __eq__(self, other):
        return (self.parent == other.parent) & (self.child == other.child)


class Ontology(object):
    """
    Class for storing meta-information about an ontology.
    """
    def __init__(self, nodes, edges, gene_map):
        self.nodes = nodes
        self.edges = edges
        self.gene_map = gene_map
        self.background = gene_map.Gene.unique()

        for node in nodes:
            node.ontology = self

    def get_parents(self, node):
        if node.id in self.edges.index.get_level_values(1):
            return pd.Series({e.parent.id: e.parent for e in self.edges[:,node.id]})
        else:
            return pd.Series()

    def get_children(self, node):
        if node.id in self.edges.index.get_level_values(0):
            return pd.Series({e.child.id: e.child for e in self.edges[node.id]})
        else:
            return pd.Series()

    def get_genes(self, node):
        descendents = node.get_descendents()
        genes = self.gene_map[self.gene_map.GO.isin(descendents.index)].Gene.unique()
        return genes

    def get_enrichment(self, node, hit_vec):
        background = hit_vec.index.intersection(self.background)
        hit_vec = hit_vec.ix[background]

        genes = node.get_genes()
        node_vec = pd.Series(1, index=genes).ix[background].fillna(0)

        stats = pd.Series({'N': len(background), 'B': len(genes), 'n': sum(hit_vec),
                           'b': sum((hit_vec > 0) & (node_vec > 0))})

        test = chi2_cont_test(node_vec, hit_vec)
        r = pd.concat([stats, test], keys=['stats','test'])
        return r