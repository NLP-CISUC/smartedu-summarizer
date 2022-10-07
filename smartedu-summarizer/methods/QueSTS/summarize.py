import logging
import nltk

from nltk.corpus import stopwords
import yake
from nltk.tokenize import word_tokenize
import networkx as nx
from nltk.stem import PorterStemmer
from math import sqrt
import string

logger = logging.getLogger('smartedu-summarizer')

def sentence_split(paragraph):
    return nltk.sent_tokenize(paragraph)


def casefolding(txt):
    return txt.lower()


def stemming(wordsList):
    ps = PorterStemmer()
    stemmed = []
    for w in wordsList:
        stemmed.append(ps.stem(w))
    return stemmed


def sim(sentence, otherSent, language):
    # sw contains the list of stopwords
    sw = stopwords.words(language) 

    s1_list = word_tokenize(sentence) 
    s1_set = {w for w in s1_list if not w in sw} 
    l1 =[]

    s2_list = word_tokenize(otherSent)
    s2_set = {w for w in s2_list if not w in sw}
    l2 =[]

    # form a set containing keywords of both strings 
    rvector = s1_set.union(s2_set) 
    for w in rvector:
        if w in s1_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in s2_set: l2.append(1)
        else: l2.append(0)
    c = 0
    
    # cosine formula 
    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    div = float((sum(l1) * sum(l2)) ** 0.5)
    if div != 0:
      cosine = c / div
    else:
      cosine = 0

    return cosine
    

def integrated_graph(sentence_list, minSimilarity, language):

    g = nx.Graph()
    
    for sentence in range(len(sentence_list)-1):
        
        #adjacent sentences are connected
        cosine = sim(sentence_list[sentence+1], sentence_list[sentence], language)
        g.add_edge(sentence+1, sentence, sim=cosine)
        
        #minSimilarity < cosine similarity between two sentences
        for otherSent in range(sentence+2, len(sentence_list)):
            
            cosine = sim(sentence_list[sentence], sentence_list[otherSent], language)
            if cosine > minSimilarity:
                g.add_edge(sentence, otherSent, sim=cosine)
    return g


#considers neighbours’ node weights
def neighbours_weights(G, language, max_iter=100, tol=1.0e-4):

    sum_neighbors = dict([(n,1.0/len(G)) for n in G])

    nnodes = G.number_of_nodes()

    # make up to max_iter iterations        
    for i in range(max_iter):
        xlast = sum_neighbors
        sum_neighbors = dict.fromkeys(list(xlast.keys()),0) #{2: 0, 1: 0, 3: 0, 4: 0} keys are nodes

        #n: keys/nodes in x
        for s in sum_neighbors: 
            #v: values in n
            for v in G[s]:
                simV = 0
                for u in G[v]:
                    simV += G[v][u].get('sim')
                if simV > 0:
                    sum_neighbors[s] += (G[s][v].get('sim') / simV) * xlast[v]
                else:
                    sum_neighbors[s] += 0

        # check convergence            
        err = sum([abs(sum_neighbors[n]-xlast[n]) for n in sum_neighbors])
        if err < nnodes * tol:
            return sum_neighbors

    raise nx.NetworkXError("""eigenvector_centrality(): 
power iteration failed to converge in %d iterations."%(i+1))""")


def node_weight(ig, sentence, query, sentence_list, neighbours, language):

    #d gives trade-off between the two parts of the formula
    d = 0.85

    #relevancy of node to the query
    sum_nodes = 0
    for i in range(len(sentence_list)):
        sum_nodes += sim(sentence_list[i], query, language)

    #final formula
    if sum_nodes == 0:
        w = (1-d) * neighbours[sentence]
    else:
        w = d * (sim(sentence_list[sentence], query, language) /sum_nodes) + (1-d) * neighbours[sentence]

    return w


def calculate_alpha(ig, node, weights, CTree, root, q, sentence_list, ngb, b, language):

    # m is average of top three node weights among the neighbours of node excluding parent of node 
    mList = []

    # n is maximum edge weight among nodes incident on node
    n = -1
    
    for neighbour in ig[node]:
        if node!= root:
            parents = [pred for pred in CTree.predecessors(node)]
            if ig[node] not in parents:
                if weights[neighbour] != -1:
                  mList.append(weights[neighbour])
                else:
                  weights[neighbour] = node_weight(ig, neighbour, q, sentence_list, ngb, language)
                  mList.append(weights[neighbour])
        else:
            if weights[neighbour] != -1:
              mList.append(weights[neighbour])
            else:
              weights[neighbour] = node_weight(ig, neighbour, q, sentence_list, ngb, language)
              mList.append(weights[neighbour])
        
        #calculate n
        if ig[node][neighbour].get('sim') > n:
            n = ig[node][neighbour].get('sim')

    if n == 0:    
        alpha = 0
    else: 
      #calculate m
      mList.sort(reverse=True)
      sumWeights = 0

      if len(mList) >= b:
          iter = b
      else:
          iter = len(mList)

      for i in range(iter):
          sumWeights += mList[i]
      m = sumWeights / b

      alpha = m/n

    return alpha, weights


def calculate_h(weights, ig, node, neighbour, q, sentence_list, ngb, alpha, beta, language):
    if weights[neighbour] != -1:
      h = alpha * ig[node][neighbour].get('sim') + beta * weights[neighbour]
    else:
      weights[neighbour] = node_weight(ig, neighbour, q, sentence_list, ngb, language)
      h = alpha * ig[node][neighbour].get('sim') + beta * weights[neighbour]

    return weights, h


def contextual_tree(ig, q, root, b, d, sentence_list, weights, ngb, language):
    openList = [root]
    closedList = [] 
    expandedArea = []  
    CTree = nx.DiGraph()
    level = 0

    #if root has query then the tree only has root
    if q in sentence_list[root]:
        #for i in range((len(expandedArea))):
            #CTree.add_edge(root, expandedArea[i])
        CTree.add_node(root)
        #print("nivel -1")
        beta = 1
        CTreeScore = beta * node_weight(ig, root, q, sentence_list, ngb, language)
        return CTree, CTreeScore, weights
    else:

        #Visit IG in BFS order starting at r
        while level < d: 
            expandedArea_q = 0

            #iterate through all previous parents (in openList)
            for node in openList:
                if node not in closedList:

                    alpha, weights = calculate_alpha(ig, node, weights, CTree, root, q, sentence_list, ngb, b, language)
                    #alpha = 0.07  # 0.05 - 0.088
                    beta = 1
                    listValues = {}

                    #choose children of node (max b)
                    for neighbour in ig[node]:
                      #print("n:", neighbour)
                      weights, h = calculate_h(weights, ig, node, neighbour, q, sentence_list, ngb, alpha, beta, language)
                      listValues[neighbour] = h

                    listValues = sorted(listValues.items(), key=lambda x: x[1], reverse=True)
                    if len(listValues) >= b:
                        iter = b
                    else:
                        iter = len(listValues)

                    for i in range(iter):
                        expandedArea.append(listValues[i][0])
                        CTree.add_node(listValues[i][0])
                        CTree.add_edge(node, listValues[i][0], level=level)
                        
                        if q in sentence_list[listValues[i][0]]:
                            expandedArea_q = 1
                        
                    #node operations are closed
                    closedList.append(node)
                    
            #verify if children have query term, if not continue the search
            if expandedArea_q:
                #print("nivel", level)
                beta = 1
                if weights[root] != -1:
                  CTreeScore = beta * weights[root]
                else:
                  weights[root] = node_weight(ig, root, q, sentence_list, ngb, language)
                  CTreeScore = beta * weights[root]

                for node,neighbour in CTree.edges:
                  alpha, weights = calculate_alpha(ig, node, weights, CTree, root, q, sentence_list, ngb, b, language)
                  weights, h = calculate_h(weights, ig, node, neighbour, q, sentence_list, ngb, alpha, beta, language)
                  lvl = CTree[node][neighbour].get('level') + 1
                  CTreeScore += h / sqrt(lvl)
                 
                return CTree, CTreeScore, weights

            #make parents = child and child null
            openList = expandedArea
            expandedArea = []
            level += 1
            

    beta = 1
    if weights[root] != -1:
       CTreeScore = beta * weights[root]
    else:
       weights[root] = node_weight(ig, root, q, sentence_list, ngb, language)
       CTreeScore = beta * weights[root]

    for node,neighbour in CTree.edges:
        alpha, weights = calculate_alpha(ig, node, weights, CTree, root, q, sentence_list, ngb, b, language)
        weights, h = calculate_h(weights, ig, node, neighbour, q, sentence_list, ngb, alpha, beta, language)
        lvl = CTree[node][neighbour].get('level') + 1
        CTreeScore += h / sqrt(lvl)

    return CTree, CTreeScore, weights

#combines all query trees
def S_graph(ig, q, root, sentence_list, weights, ngb, b, language):
    SGraph = nx.DiGraph()
    query_words = q.split(" ")
    d = 15
    S_edges = []
    CTreesScores = 0
    solo_nodes = []
    num_nodes = 0
    ps = PorterStemmer()

    for word in query_words:
        #print(word)
        CTree, CTreeScore, weights = contextual_tree(ig, ps.stem(word), root, b, d, sentence_list, weights, ngb, language)
        if CTree.number_of_nodes() == 1:
            solo_nodes.append(list(CTree.nodes)[0])
        else:    
            S_edges = list(set(S_edges).union(set(CTree.edges)))
        CTreesScores += CTreeScore
    
    SGraph.add_edges_from(S_edges)

    for i in solo_nodes:
        if i not in list(SGraph.nodes):
            num_nodes += 1
            SGraph.add_node(i)
    num_nodes += SGraph.number_of_nodes()

    SGraphScore = num_nodes * CTreesScores #mudar o tamanho para sumarios mais peq(1/sqrt)

    return SGraph, SGraphScore, weights


def QueSTS(txt, b, language):
    
    #-------------------------preprocessing-------------------------------------------
    #cleaning the text
    sentence_list_clean = sentence_split(txt)
    sentence_list = sentence_split(txt)
    #print(sentence_list[1:8])
    
    for i in range(len(sentence_list)):
        text = casefolding(sentence_list[i]) #lower case
        text = text.translate(str.maketrans('', '', string.punctuation)) #remove puntuaction
        wordsList = word_tokenize(text)
        filtered_text = (" ").join(stemming(wordsList)) #stemming
        sentence_list[i] = filtered_text

    #construct integrated graph
    minSimilarity = 0.001
    ig = integrated_graph(sentence_list, minSimilarity, language)
    ngb = neighbours_weights(ig, language, max_iter=100, tol=1.0e-4)
    
    #extract keywords from text
    kw_extractor = yake.KeywordExtractor()
    max_ngram_size = 3
    deduplication_threshold = 0.5
    numOfKeywords = 5
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(txt)
    #for kw in keywords:
        #print(casefolding(kw[0]))
    query = casefolding(keywords[1][0])
    #print("q:", casefolding(keywords[1][0]))


    #-------------------------------------summarization------------------------
    nSentences = ig.number_of_nodes()
    summ = []
    score = -1
    weights = [-1] * len(sentence_list)

    for i in range(nSentences):
        SGraph, SGraphScore, weights = S_graph(ig, query, i, sentence_list, weights, ngb, b, language)
        if SGraphScore > score:
            score = SGraphScore
            summ = list(SGraph.nodes())

    summ.sort()
    summary = ""
    for sentence in summ:
        summary += sentence_list_clean[sentence] + " "

    #print(summary)
    return summary


def summarize(args):
    logger.info(f'Loading file \'{args.file}\'')
    with args.file.open('rU', encoding='utf-8') as file_:
        text = file_.read()
    logger.info(f'File loaded')
    logger.debug(f'Loaded text\n\n{text}')

    logger.info('Summarizing')
    summary = QueSTS(text, 11, args.language)
    logger.info('Summarization done')
    logger.debug(f'Summary\n\n{summary}')

    logger.info('Saving summary')
    with args.output.open('w', encoding='utf-8') as file_:
        file_.write(summary)
    logger.info(f'Saved summary to \'{args.output}\'')