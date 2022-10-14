import logging
import nltk

from nltk.corpus import wordnet as wn
import numpy as np
import spacy
nltk.download('omw-1.4')

logger = logging.getLogger('smartedu-summarizer')


def sentence_split(paragraph):
    return nltk.sent_tokenize(paragraph)


def posTag(text, language):

    if language == 'english':
        nlp = spacy.load('en_core_web_sm')
    else:
        nlp = spacy.load('pt_core_news_sm')

    Noun_phrases = []
    words = []
    doc = nlp(text)
    for token in doc:
      words.append(token.text)
      if token.pos_ == 'NOUN':
        Noun_phrases.append(token.text)

    return Noun_phrases, words


def append_value(dict_obj, key, value):
    # Check if key exist in dict or not
    if key in dict_obj:
        # Key exist in dict.
        # Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            # If type is not list then make it list
            dict_obj[key] = [dict_obj[key]]
        # Append the value in list
        dict_obj[key].append(value)
    else:
        # As key is not in dict,
        # so, add key-value pair
        dict_obj[key] = value


def LexChain(Noun_phrases):
    nltk.download('omw')
    lexicalChain = {}

    for i in Noun_phrases:
      for j in wn.synsets(i, lang='por'):
        word = j.name().split('.')[0]
        if word not in lexicalChain.keys():
            append_value(lexicalChain, word, i)
    return lexicalChain


def lexicalChainScore(lexicalChain, words):
    distinct_occurences = {}
    length = {}
    score = {}

    for i in lexicalChain.keys():
        if i not in length:
            length[i] = 0

        #checks if is a list or scalar
        if isinstance(lexicalChain[i], list):

            #iterates through the list of values
            for j in range(0, len(lexicalChain[i])):
                count = lexicalChain[i].count(lexicalChain[i][j])
                subs = len(list(filter(lambda x: lexicalChain[i][j] in x, lexicalChain[i])))

                #checks if the word is unique in the list
                if count == 1 and subs == 1:
                    if i not in distinct_occurences:
                        distinct_occurences[i] = 1
                    else:
                        distinct_occurences[i] = distinct_occurences[i] + 1

                #adds the frequency of all the words in a lexical chain
                length[i] = length[i] + words.count(lexicalChain[i][j])
        else:
            distinct_occurences[i] = 1
            length[i] = length[i] + words.count(lexicalChain[i])

        #score of each lexical chain
        score[i] = length[i] * (1 - (distinct_occurences[i]/length[i]))
    return score


def strongChains(chains,lexicalChains):  #receives a dictionary with chain name and score and another the lexchains
  strong={}

  scores = list(chains.values())
  
  avg = np.average(scores)
  var = np.var(scores)

  for k in chains.keys():      #for each chain we compare its score to the threshold
    if(chains[k]> avg + 1 * var): 
      strong[k] = lexicalChains[k]  #if its relevant we copy the lexicalchain to the set of strong chains


  return strong


def sentenceScore(chains, text, Noun_phrases):
    scores = [] 
    text = sentence_split(text)

    for s in text:  #for every sentence
        words = s.split()  #gets words in each sentence
        length = len(words)
        score=0
        for w in words:
            if w in Noun_phrases:  # for every noun in the sentence
                for c in chains: 
                    if isinstance(chains[c], list): #checks how many words for each meaning (keys) there are, if its a list it needs to iterate on each word
                        for i in range (0, len(chains[c])):
                            if w in chains[c][i]:  #if the noun in the sentence is in the value of each chain (words with meaning c), add one to the score
                                score+=1

            if w in chains.values():  #if its not a list we can check directly, no need to iterate
                score+=1
        if(length!=0):
            scores.append(score/length)
        else:
            scores.append(0)

    return scores


def contains_word(s, w):
    return (' ' + w + ' ') in (' ' + s + ' ')


def sentences_selection(text, strong_chains, sentence_scores):
    words_list = sentence_split(text)

    sentences_selected = {}

    avg_sentencescore = np.average(sentence_scores)
    nsentence = 0

    # Extraction based on the Article Category
    first_sentence = words_list[0]

    for sentence in words_list:

        #Extraction using strong Lexical Chains
        for key in strong_chains:
            #first sentence that contains the first appearance of a representative chain member in the text
            # checks if is a list or scalar
            if isinstance(strong_chains[key], list):

                # iterates through the list of values
                for values in range(0, len(strong_chains[key])):
                    if contains_word(sentence, strong_chains[key][values]):
                        if sentence in sentences_selected:
                          sentences_selected[sentence] = sentences_selected[sentence] + 1
                        else:
                            sentences_selected[sentence] = 1
                        strong_chains[key] = -1
                        break

            elif strong_chains[key] != -1 and strong_chains[key] in sentence:
                if sentence in sentences_selected:
                  sentences_selected[sentence] = sentences_selected[sentence] + 1
                else:
                  sentences_selected[sentence] = 1
                strong_chains[key] = -1

        
        #Extraction using Sentence Score
        #if sentence_scores[nsentence] > avg_sentencescore and sentence not in sentences_selected:
        if sentence_scores[nsentence] > avg_sentencescore:
            if sentence in sentences_selected:
              sentences_selected[sentence] = sentences_selected[sentence] + 1
            else:
                sentences_selected[sentence] = 1

        nsentence += 1

    sentences_final = []
    for k, s in sentences_selected.items():
        sentences_final.append(k)

    if len(sentences_final) == 0:
      sentences_final.append(first_sentence)

    return sentences_final


def Lexical_Chains(text, language):

    lexicalChain = {}
    Noun_phrases, words = posTag(text, language)

    lexicalChain = LexChain(Noun_phrases)
    score = lexicalChainScore(lexicalChain, words)
    strong = strongChains(score, lexicalChain)
    
    sentenceScores = sentenceScore(strong, text, Noun_phrases)

    sentences_selected = sentences_selection(text, strong, sentenceScores)

    sep = " "
    s = sep.join(sentences_selected)

    return s



def summarize(args):
    logger.info(f'Loading file \'{args.file}\'')
    with args.file.open('rU', encoding='utf-8') as file_:
        text = file_.read()
    logger.info(f'File loaded')
    logger.debug(f'Loaded text\n\n{text}')

    logger.info('Summarizing')
    summary = Lexical_Chains(text, args.language)
    logger.info('Summarization done')
    logger.debug(f'Summary\n\n{summary}')

    logger.info('Saving summary')
    with args.output.open('w', encoding='utf-8') as file_:
        file_.write(summary)
    logger.info(f'Saved summary to \'{args.output}\'')