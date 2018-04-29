import networkx as nx

def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined in bonus.md.
    Args:
      graph....a networkx graph
      node.....a node to score potential new edges for.
    Returns:
      A list of ((node, ni), score) tuples, representing the 
                score assigned to edge (node, ni)
                (note the edge order)
    """
    #code for jaccard
    denom1 = 0
    
    neighbors = set(graph.neighbors(node))
    scores = []
    #calculating first term in denominator
    for node_val in neighbors:
        #print("The neighbors of node D", node_val)
        denom1 += (graph.degree(node_val))
    #iterating through each neighbor
    for n in graph.nodes():
        denom2 = 0
        numerator_jaccard = 0
        if (node,n) not in graph.edges() and node != n:
            #print("D is compared with", n  )
            neighbors2 = set(graph.neighbors(n))
            #calculating second term in denominator
            for node_val in neighbors2:
                denom2 += (graph.degree(node_val))
                #print("The neighbor is " , node_val)
            #calculating the numerator
            for node_val in (neighbors & neighbors2) :
                numerator_jaccard += (1 / graph.degree(node_val))
                #print("The common neighbor is" , node_val)
                #print("The numerator is ", numerator_jaccard)
            scores.append(((node, n), numerator_jaccard / ((1/denom1) + (1/denom2))))
                    
    return (sorted(scores, key=lambda x: x[1], reverse=True))
