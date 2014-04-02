__author__ = 'agross'

"""
This module is for convenience functions that compute statistics on an
ontology. General purpose statistics should be left for other modules,
rather here the idea is to wrap these functions in code that uses the
structure and/or attributes of the ontology.
"""

import pandas as pd
from Scipy import fisher_exact_test, chi2_cont_test, anova


def node_enrichment(node, data_vec, test='default', context='global',
                   parent='All'):
    """
    Get enrichment of a Node in an Ontology for a given signal.

    Args:
        test: FET (Fischer's Exact Test), chi-square, anova
        context: global (background is all genes in Ontology) local (background
            is parent genes)
        parent: for local enrichment specify which parent to test, defaults to
            all parents of node

    Returns:
        For local enrichment --> A DataFrame with a column for each parent
        For global enrichment --> A Series
        Both are indexed by a MultiIndex with:
            name: name attribute for the node(s) represented,
            stats: a brief data summary
            test: details and p-value of the statistical test
    """
    if test is 'default':
        if len(data_vec.unique()) == 2:
            test = 'FET'
        else:
            test = 'anova'

    if context is 'global':
        name = pd.Series({'node': node.name})
        background = data_vec.index.intersection(node.ontology.background)
    if context is 'local':
        if parent is None or len(node.parents) == 0:
            return pd.DataFrame()
        elif parent is 'All':
            return pd.concat({p.id: node_enrichment(node, data_vec, test,
                                                   context, p)
                              for p in node.parents}, axis=1)
        else:
            name = pd.Series({'parent': parent.name, 'child': node.name})
            background = data_vec.index.intersection(parent.genes)

    data_vec = data_vec.ix[background]
    node_vec = pd.Series(1, index=node.genes).ix[background].fillna(0)

    stats = pd.Series({'N': len(background), 'B': int(sum(node_vec))})

    if test in ['FET', 'chi-square']:
        stats['n'] = sum(data_vec)
        stats['b'] = sum((data_vec > 0) & (node_vec > 0))

    if test == 'FET':
        res = fisher_exact_test(node_vec, data_vec)
    elif test == 'chi-square':
        res = chi2_cont_test(node_vec, data_vec)
    elif test == 'anova':
        res = anova(node_vec, data_vec)

    r = pd.concat([name, stats, res], keys=['name', 'stats', 'test'])
    return r
