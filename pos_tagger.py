from multiprocessing import Pool
import numpy as np
import time
from sklearn.metrics import f1_score
from utils import *
from constants1 import *


""" Contains the part of speech tagger class. """


def sort_tuple(tup):
    # convert the list of tuples to a numpy array with data type (object, int)
    arr = np.array(tup, dtype=[('col1', object), ('col2', float)])
    # get the indices that would sort the array based on the second column
    indices = np.argsort(-arr['col2'])
    # use the resulting indices to sort the array
    sorted_arr = arr[indices]
    # convert the sorted numpy array back to a list of tuples
    sorted_tup = [(row['col1'], row['col2']) for row in sorted_arr]
    return sorted_tup

def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions,
    or you can use it as is. 
    
    As per the write-up, you may find it faster to use multiprocessing (code included). 
    
    """
    # for i in range(len(data[0])):
    #     sent = data[0][i]            
    #     if sent[1].lower() in model.low_vocub:
    #         sent[1] = sent[1].lower()
    #     data[0][i] = sent
    model.get_emissions_infer(data[0])
    processes = 6
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n//processes
    n_tokens = sum([len(d) for d in sentences])
    unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])
    predictions = {i:None for i in range(n)}
    gold_probabilities = {i:None for i in range(n)}
    probabilities = {i:None for i in range(n)}
         
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in tqdm(range(0, n, k)):
        res.append(pool.apply_async(infer_sentences, [model, sentences[i:i+k], i]))
    ans = [r.get(timeout=None) for r in tqdm(res)]
    predictions = dict()
    for a in tqdm(ans):
        predictions.update(a)
    print(f"Inference Runtime: {(time.time()-start)/60} minutes.")
    
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    predict_tags = list(predictions.values())
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i+k], predict_tags[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    probabilities = dict()
    for a in ans:
        probabilities.update(a)
    print(f"Probability Estimation Runtime: {(time.time()-start)/60} minutes.")

    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i+k], tags[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    gold_probabilities = dict()
    for a in ans:
        gold_probabilities.update(a)
    print(f"Probability Estimation Runtime: {(time.time()-start)/60} minutes.")
    
    gold_probabilities_list = list(gold_probabilities.values())
    probabilities_list = list(probabilities.values())
    prob_comp = [gold_probabilities_list[i] > probabilities_list[i] for i in range(n)]
    print("# of sub-optimality: ", sum(prob_comp))

    word_list = [sentences[i][j] for i in range(n) for j in range(len(sentences[i])) if sentences[i][j] != '<STOP>']
    pred_list = [predictions[i][j] for i in range(n) for j in range(len(sentences[i])) if sentences[i][j] != '<STOP>' ]
    tag_list = [tags[i][j] for i in range(n) for j in range(len(sentences[i])) if sentences[i][j] != '<STOP>' ]
    id_list = list(range(len(pred_list)))
    # df = pd.DataFrame({'id':id_list, 'word':word_list, 'pred':pred_list, 'tag':tag_list})
    # df.to_csv('pred.csv', index=False)
    df = pd.DataFrame({'id':id_list, 'tag':pred_list})
    df.to_csv('dev_pred.csv', index=False)



    token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j]]) / n_tokens
    unk_token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j] and sentences[i][j] not in model.word2idx.keys()]) / unk_n_tokens
    whole_sent_acc = 0
    num_whole_sent = 0
    for k in range(n):
        sent = sentences[k]
        eos_idxes = indices(sent, '.')
        start_idx = 1
        end_idx = eos_idxes[0]
        for i in range(1, len(eos_idxes)):
            whole_sent_acc += 1 if tags[k][start_idx:end_idx] == predictions[k][start_idx:end_idx] else 0
            num_whole_sent += 1
            start_idx = end_idx+1
            end_idx = eos_idxes[i]
    print("Whole sent acc: {}".format(whole_sent_acc/num_whole_sent))
    print("Mean Probabilities: {}".format(sum(probabilities.values())/n))
    print("Token acc: {}".format(token_acc))
    print("Unk token acc: {}".format(unk_token_acc))
    print("Mean F1 Score:", f1_score(
        tag_list,
        pred_list,
        average = "weighted"
    ))
    
    confusion_matrix(pos_tagger.tag2idx, pos_tagger.idx2tag, predictions.values(), tags, 'cm.png')

    model.clean_new_data()

    return whole_sent_acc/num_whole_sent, token_acc, sum(probabilities.values())/n


def predict(sentences, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions,
    or you can use it as is. 
    
    As per the write-up, you may find it faster to use multiprocessing (code included). 
    
    """
    model.get_emissions_infer(sentences)
    processes = 6
    n = len(sentences)
    k = n//processes
    predictions = {i:None for i in range(n)}
         
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in tqdm(range(0, n, k)):
        res.append(pool.apply_async(infer_sentences, [model, sentences[i:i+k], i]))
    ans = [r.get(timeout=None) for r in tqdm(res)]
    predictions = dict()
    for a in tqdm(ans):
        predictions.update(a)
    print(f"Inference Runtime: {(time.time()-start)/60} minutes.")
    
    pred_list = [predictions[i][j] for i in range(n) for j in range(len(sentences[i])) if sentences[i][j] != '<STOP>' ]
    id_list = list(range(len(pred_list)))
    df = pd.DataFrame({'id':id_list, 'tag':pred_list})
    df.to_csv('test_y.csv', index=False)    

    model.clean_new_data()

    return 0


class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        self.suffix_tag_probs = {}  # For storing suffix probabilities
        self.capitalized_tag_probs = {}  # For storing capitalized word tag probabilities
        self.infer_count = 0

    
    
    def get_unigrams(self):
        """
        Computes unigrams. 
        Tip. Map each tag to an integer and store the unigrams in a numpy array. 
        """
        self.unigrams = np.zeros(len(self.all_tags))
        for tags in self.data[1]:
            for tag in tags:
                self.unigrams[self.tag2idx[tag]] += 1
        if SMOOTHING == LAPLACE:
            len_token = np.sum(self.unigrams)
            for i in range(len(self.unigrams)):
                self.unigrams[i] = (self.unigrams[i] + LAPLACE_FACTOR) / (len_token + LAPLACE_FACTOR * len(self.all_tags))
        else:
            self.unigrams /= np.sum(self.unigrams)
        self.log_unigrams = np.log(self.unigrams)

    def get_bigrams(self):        
        """
        Computes bigrams. 
        Tip. Map each tag to an integer and store the bigrams in a numpy array
             such that bigrams[index[tag1], index[tag2]] = Prob(tag2|tag1). 
        """
        self.bigrams = np.zeros((len(self.all_tags), len(self.all_tags)))
        for tags in self.data[1]:
            for i in range(1, len(tags)):
                self.bigrams[self.tag2idx[tags[i-1]], self.tag2idx[tags[i]]] += 1
        if SMOOTHING == LAPLACE:
            for i in range(len(self.all_tags)):
                self.bigrams[i, ] = (self.bigrams[i, ] + LAPLACE_FACTOR) / (np.sum(self.bigrams[i, ]) + LAPLACE_FACTOR * len(self.all_tags))
        else:
            self.bigrams /= np.sum(self.bigrams, axis=1, keepdims=True)
        self.bigrams = np.nan_to_num(self.bigrams, 0)

        if (NGRAMM == 2) and (SMOOTHING == INTERPOLATION):
            for i in range(len(self.all_tags)):                
                self.bigrams[i, :] = LAMBDAS[0] * self.unigrams + LAMBDAS[1] * self.bigrams[i, :] 

        self.bigrams[self.bigrams == 0] = EPSILON 
        self.log_bigrams = np.log(self.bigrams)
    
    def get_trigrams(self):
        """
        Computes trigrams. 
        Tip. Similar logic to unigrams and bigrams. Store in numpy array. 
        """
        self.trigrams = np.zeros((len(self.all_tags), len(self.all_tags), len(self.all_tags)))
        for tags in self.data[1]:
            for i in range(2, len(tags)):
                self.trigrams[self.tag2idx[tags[i-2]], self.tag2idx[tags[i-1]], self.tag2idx[tags[i]]] += 1
        if SMOOTHING == LAPLACE:
            for i in range(len(self.all_tags)):
                for j in range(len(self.all_tags)):
                    self.trigrams[i, j, :] = (self.trigrams[i, j, :] + LAPLACE_FACTOR) / (np.sum(self.trigrams[i, j, :]) + LAPLACE_FACTOR * len(self.all_tags))
        else:
            for i in range(len(self.all_tags)):
                for j in range(len(self.all_tags)):
                    self.trigrams[i, j, :] = self.trigrams[i, j, :]/np.sum(self.trigrams[i, j, :])      
        self.trigrams = np.nan_to_num(self.trigrams, 0)

        if (NGRAMM == 3) and (SMOOTHING == INTERPOLATION):
            for i in range(len(self.all_tags)):
                for j in range(len(self.all_tags)):
                    self.trigrams[i, j, :] = LAMBDAS[0] * self.unigrams + LAMBDAS[1] * self.bigrams[j, :] + LAMBDAS[2] * self.trigrams[i, j, :]

        self.trigrams[self.trigrams == 0] = EPSILON
        self.log_trigrams = np.log(self.trigrams)  

    def get_fourgrams(self):
        """
        Computes trigrams. 
        Tip. Similar logic to unigrams and bigrams. Store in numpy array. 
        """
        self.fourgrams = np.zeros((len(self.all_tags), len(self.all_tags), len(self.all_tags), len(self.all_tags)))
        for tags in self.data[1]:
            for i in range(3, len(tags)):
                self.fourgrams[self.tag2idx[tags[i-3]], self.tag2idx[tags[i-2]], self.tag2idx[tags[i-1]], self.tag2idx[tags[i]]] += 1
        if SMOOTHING == LAPLACE:
            for i in range(len(self.all_tags)):
                for j in range(len(self.all_tags)):
                    for k in range(len(self.all_tags)):
                        self.fourgrams[i, j, :] = (self.fourgrams[i, j, k, :] + LAPLACE_FACTOR) / (np.sum(self.fourgrams[i, j, k, :]) + LAPLACE_FACTOR * len(self.all_tags))
        else:
            for i in range(len(self.all_tags)):
                for j in range(len(self.all_tags)):
                    for k in range(len(self.all_tags)):
                        self.fourgrams[i, j, k, :] = self.fourgrams[i, j, k, :]/np.sum(self.fourgrams[i, j, k, :])      
        self.fourgrams = np.nan_to_num(self.fourgrams, 0)

        if SMOOTHING == INTERPOLATION:
            for i in range(len(self.all_tags)):
                for j in range(len(self.all_tags)):
                    for k in range(len(self.all_tags)):
                        self.fourgrams[i, j, k, :] = LAMBDAS[0] * self.unigrams + LAMBDAS[1] * self.bigrams[j, :] + LAMBDAS[2] * self.trigrams[i, j, :] + LAMBDAS[3] * self.fourgrams[i, j, k, :]

        self.fourgrams[self.fourgrams == 0] = EPSILON
        self.log_fourgrams = np.log(self.fourgrams)        
    
    
    def get_emissions(self):
        """
        Computes emission probabilities. 
        Tip. Map each tag to an integer and each word in the vocabulary to an integer. 
             Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag) 
        """
        self.vocabulary = list(set([w for sent in self.data[0] for w in sent]))
        self.word2idx = {self.vocabulary[i]: i for i in range(len(self.vocabulary))}
        self.emissions = np.zeros((len(self.all_tags), len(self.vocabulary)))
        for sent, tags in zip(self.data[0], self.data[1]):
            for word, tag in zip(sent, tags):
                self.emissions[self.tag2idx[tag], self.word2idx[word]] += 1
        self.emissions /= np.sum(self.emissions, axis=1, keepdims=True)
        self.emissions[self.emissions == 0] = EPSILON
        self.log_emissions = np.log(self.emissions)

    def get_emissions_infer(self, sentences):
        """
        Computes emission probabilities. 
        Tip. Map each tag to an integer and each word in the vocabulary to an integer. 
             Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag) 
        """
        
        tmp_list = list(set([w for sent in sentences for w in sent]))
        self.unkonw_vocabulary = [w for w in tmp_list if w not in self.vocabulary]
        self.new_vocabulary = self.vocabulary.copy()
        self.new_vocabulary.extend(self.unkonw_vocabulary)
        self.unkonw_w_tags = []
        for w in self.unkonw_vocabulary:
            self.unkonw_w_tags.append(self.handle_unknown_word(w))
        
        self.new_word2idx = {self.new_vocabulary[i]: i for i in range(len(self.new_vocabulary))}
        self.log_new_emissions = np.zeros((len(self.all_tags), len(self.new_vocabulary)))
        self.log_new_emissions[self.log_new_emissions == 0] = np.log(EPSILON)
        self.log_new_emissions[:, :len(self.vocabulary)] = self.log_emissions

        for word in self.unkonw_vocabulary:                
            self.log_new_emissions[self.tag2idx[self.unkonw_w_tags[self.new_word2idx[word] - len(self.vocabulary)]], self.new_word2idx[word]] = 0

    def clean_new_data(self):
        """
        Computes emission probabilities. 
        Tip. Map each tag to an integer and each word in the vocabulary to an integer. 
             Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag) 
        """
        
        self.unkonw_vocabulary = []
        self.new_vocabulary = []       
        self.unkonw_w_tags = []        
        self.new_word2idx = []
        self.log_new_emissions = []       



    def handle_unknown_word(self, word):

        # return predict_tag(word, self.mlp_model, self.idx2tag, self.tag2idx)

        # If word is capitalized
        if word[0].isupper() and self.capitalized_tag_probs:
            return max(self.capitalized_tag_probs, key=self.capitalized_tag_probs.get)

        # If word has a known suffix
        suffix = word[-3:]
        if suffix in self.suffix_tag_probs:
            return max(self.suffix_tag_probs[suffix], key=self.suffix_tag_probs[suffix].get)

        if word[-3:] == "ing":
            return "VBG"  # Verb, gerund or present participle
        elif word[-2:] == "ly":
            return "RB"   # Adverb
        elif word[-2:] == "ed":
            return "VBD"  # Verb, past tense
        # Capitalization
        elif word[0].isupper():
            return "NNP"  # Proper noun, singular
        # Numeric
        elif any(char.isdigit() for char in word):
            return "CD"   # Cardinal number
        # Hyphenation
        elif "-" in word:
            return "JJ"   # Adjective
        # Default
        else:
            return self.idx2tag[np.argmax(self.unigrams)]


    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        
        """
        self.data = data
        self.all_tags = list(set([t for tag in data[1] for t in tag]))
        # self.len_tokens = sum([len(tag) for tag in data[1]])
        self.all_tags.sort()
        self.tag2idx = {self.all_tags[i]:i for i in range(len(self.all_tags))}
        self.idx2tag = {v:k for k,v in self.tag2idx.items()}

        # self.low_vocub = list(set([w for sent in data[0] for w in sent if w[0].islower()]))
        # for i in range(len(data[0])):
        #     sent = data[0][i]            
        #     if sent[1].lower() in self.low_vocub:
        #         sent[1] = sent[1].lower()
        #     data[0][i] = sent
        
        # Compute unigrams, bigrams, trigrams, and emissions
        self.get_unigrams()
        self.get_bigrams()
        if NGRAMM > 2:
            self.get_trigrams()
        if NGRAMM > 3:
            self.get_fourgrams()
        self.get_emissions()

        suffix_counts = {}
        for sentence, tags in zip(data[0], data[1]):
            for word, tag in zip(sentence, tags):
                # Consider the last 3 characters as suffix for simplicity
                suffix = word[-3:]
                if suffix not in suffix_counts:
                    suffix_counts[suffix] = {}
                if tag not in suffix_counts[suffix]:
                    suffix_counts[suffix][tag] = 0
                suffix_counts[suffix][tag] += 1

        for suffix, tag_counts in suffix_counts.items():
            total = sum(tag_counts.values())
            self.suffix_tag_probs[suffix] = {tag: count/total for tag, count in tag_counts.items()}

        # Compute capitalized word tag probabilities
        capitalized_counts = {}
        for sentence, tags in zip(data[0], data[1]):
            for word, tag in zip(sentence, tags):
                if word[0].isupper():
                    if tag not in capitalized_counts:
                        capitalized_counts[tag] = 0
                    capitalized_counts[tag] += 1

        total_capitalized = sum(capitalized_counts.values())
        self.capitalized_tag_probs = {tag: count/total_capitalized for tag, count in capitalized_counts.items()}



    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        log_prob = 0
        if NGRAMM == 2:
            for i in range(len(sequence)):
                word = sequence[i]
                tag = tags[i]
                log_emission_prob = self.log_new_emissions[self.tag2idx[tag], self.new_word2idx[word]]
                if i == 0:
                    log_transition_prob = self.log_unigrams[self.tag2idx[tag]]
                else:
                    log_transition_prob = self.log_bigrams[self.tag2idx[tags[i-1]], self.tag2idx[tag]]                
                log_prob += log_emission_prob + log_transition_prob
                
        if NGRAMM == 3:
            for i in range(len(sequence)):
                word = sequence[i]
                tag = tags[i]
                log_emission_prob = self.log_new_emissions[self.tag2idx[tag], self.new_word2idx[word]]
                if i == 0:
                    log_transition_prob = self.log_unigrams[self.tag2idx[tag]]
                elif i == 1:
                    log_transition_prob = self.log_bigrams[self.tag2idx[tags[i-1]], self.tag2idx[tag]]  
                else:
                    log_transition_prob = self.log_trigrams[self.tag2idx[tags[i-2]], self.tag2idx[tags[i-1]], self.tag2idx[tag]]
                log_prob += log_emission_prob + log_transition_prob
        
        if NGRAMM == 4:
            for i in range(len(sequence)):
                word = sequence[i]
                tag = tags[i]
                log_emission_prob = self.log_new_emissions[self.tag2idx[tag], self.new_word2idx[word]]
                if i == 0:
                    log_transition_prob = self.log_unigrams[self.tag2idx[tag]]
                elif i == 1:
                    log_transition_prob = self.log_bigrams[self.tag2idx[tags[i-1]], self.tag2idx[tag]]  
                elif i == 2:
                    log_transition_prob = self.log_trigrams[self.tag2idx[tags[i-2]], self.tag2idx[tags[i-1]], self.tag2idx[tag]]
                else:
                    log_transition_prob = self.log_fourgrams[self.tag2idx[tags[i-3]], self.tag2idx[tags[i-2]], self.tag2idx[tags[i-1]], self.tag2idx[tag]]
                log_prob += log_emission_prob + log_transition_prob
        return log_prob  
    
    def greedy_search(self, sequence):
        tags = []
        if NGRAMM == 2:
            for i, word in enumerate(sequence):                
                if i == 0:
                    log_tag_probs = self.log_new_emissions[:, self.new_word2idx[word]] + self.log_unigrams
                else:
                    log_tag_probs = self.log_new_emissions[:, self.new_word2idx[word]] + self.log_bigrams[self.tag2idx[tags[i-1]], :]
                best_tag_idx = np.argmax(log_tag_probs)
                tags.append(self.idx2tag[best_tag_idx])
        elif NGRAMM == 3:
            for i, word in enumerate(sequence):                
                if i == 0:
                    log_tag_probs = self.log_new_emissions[:, self.new_word2idx[word]] + self.log_unigrams                    
                elif i == 1:
                    log_tag_probs = self.log_new_emissions[:, self.new_word2idx[word]] + self.log_bigrams[self.tag2idx[tags[i-1]], :]
                else:
                    log_tag_probs = self.log_new_emissions[:, self.new_word2idx[word]] + self.log_trigrams[self.tag2idx[tags[i-2]], self.tag2idx[tags[i-1]], :]
                best_tag_idx = np.argmax(log_tag_probs)
                tags.append(self.idx2tag[best_tag_idx])                
        elif NGRAMM == 4:
            for i, word in enumerate(sequence):                
                if i == 0:
                    log_tag_probs = self.log_new_emissions[:, self.new_word2idx[word]] + self.log_unigrams                    
                elif i == 1:
                    log_tag_probs = self.log_new_emissions[:, self.new_word2idx[word]] + self.log_bigrams[self.tag2idx[tags[i-1]], :]
                elif i == 2:
                    log_tag_probs = self.log_new_emissions[:, self.new_word2idx[word]] + self.log_trigrams[self.tag2idx[tags[i-2]], self.tag2idx[tags[i-1]], :]
                else:
                    log_tag_probs = self.log_new_emissions[:, self.new_word2idx[word]] + self.log_fourgrams[self.tag2idx[tags[i-3]], self.tag2idx[tags[i-2]], self.tag2idx[tags[i-1]], :]
                best_tag_idx = np.argmax(log_tag_probs)
                tags.append(self.idx2tag[best_tag_idx])    

        return tags
    

    def beam_search(self, sequence):
        paths = [([], 0)]  # List of tuples (path, score)
        if NGRAMM == 2:
            for i, word in enumerate(sequence):                    
                new_paths = []            
                for path, path_score in paths:
                    for tag in self.all_tags:
                        if i == 0:                            
                            score = self.log_new_emissions[self.tag2idx[tag], self.new_word2idx[word]] + self.log_unigrams[self.tag2idx[tag]]                            
                        else:
                            score = self.log_new_emissions[self.tag2idx[tag], self.new_word2idx[word]] + self.log_bigrams[self.tag2idx[path[i-1]], self.tag2idx[tag]]                      
                        
                        new_path = path + [tag]
                        new_score = path_score + score
                        new_paths.append((new_path, new_score))
                # Sort by score and select top BEAM_K paths
                new_paths = sort_tuple(new_paths)
                paths = new_paths[:BEAM_K]    
        if NGRAMM == 3:
            for i, word in enumerate(sequence):                    
                new_paths = []            
                for path, path_score in paths:
                    for tag in self.all_tags:
                        if i == 0:                            
                            score = self.log_new_emissions[self.tag2idx[tag], self.new_word2idx[word]] + self.log_unigrams[self.tag2idx[tag]]                            
                        elif i == 1:
                            score = self.log_new_emissions[self.tag2idx[tag], self.new_word2idx[word]] + self.log_bigrams[self.tag2idx[path[i-1]], self.tag2idx[tag]]
                        else:
                            score = self.log_new_emissions[self.tag2idx[tag], self.new_word2idx[word]] + self.log_trigrams[self.tag2idx[path[i-2]], self.tag2idx[path[i-1]], self.tag2idx[tag]]
                        
                        new_path = path + [tag]
                        new_score = path_score + score
                        new_paths.append((new_path, new_score))
                # Sort by score and select top BEAM_K paths
                new_paths = sort_tuple(new_paths)
                paths = new_paths[:BEAM_K]   
        if NGRAMM == 4:
            for i, word in enumerate(sequence):                    
                new_paths = []            
                for path, path_score in paths:
                    for tag in self.all_tags:
                        if i == 0:                            
                            score = self.log_new_emissions[self.tag2idx[tag], self.new_word2idx[word]] + self.log_unigrams[self.tag2idx[tag]]                            
                        elif i == 1:
                            score = self.log_new_emissions[self.tag2idx[tag], self.new_word2idx[word]] + self.log_bigrams[self.tag2idx[path[i-1]], self.tag2idx[tag]]
                        elif i == 2:
                            score = self.log_new_emissions[self.tag2idx[tag], self.new_word2idx[word]] + self.log_trigrams[self.tag2idx[path[i-2]], self.tag2idx[path[i-1]], self.tag2idx[tag]]
                        else:
                            score = self.log_new_emissions[self.tag2idx[tag], self.new_word2idx[word]] + self.log_fourgrams[self.tag2idx[path[i-3]], self.tag2idx[path[i-2]], self.tag2idx[path[i-1]], self.tag2idx[tag]]
                        
                        new_path = path + [tag]
                        new_score = path_score + score
                        new_paths.append((new_path, new_score))
                # Sort by score and select top BEAM_K paths
                new_paths = sort_tuple(new_paths)
                paths = new_paths[:BEAM_K]           

        # Return the path with the highest score
        return paths[0][0]
    
    def viterbi(self, sequence):
        # Number of tags and words
        n_tags = len(self.all_tags)
        n_words = len(sequence)

        if NGRAMM == 2:
            # Initialize pi table and backpointer table
            pi = np.zeros((n_words, n_tags)) + float('-inf')
            backpointer = np.zeros((n_words, n_tags), dtype=int)

            # Base case initialization
            for u in self.all_tags:
                pi[0, self.tag2idx[u]] = self.log_unigrams[self.tag2idx[u]] + self.log_new_emissions[self.tag2idx[u], self.new_word2idx[sequence[0]]]

            # Viterbi recursion for bigram
            for k in range(1, n_words):
                for v in self.all_tags:
                    max_prob = float('-inf')
                    max_tag = None
                    for u in self.all_tags:
                        score = pi[k-1, self.tag2idx[u]] + self.log_bigrams[self.tag2idx[u], self.tag2idx[v]] + self.log_new_emissions[self.tag2idx[v], self.new_word2idx[sequence[k]]]
                        if score > max_prob:
                            max_prob = score
                            max_tag = u
                    pi[k, self.tag2idx[v]] = max_prob
                    backpointer[k, self.tag2idx[v]] = self.tag2idx[max_tag]

            # Backtrack to find the best path for bigram
            tags = []
            v_max = np.argmax(pi[n_words-1])
            tags.append(self.idx2tag[v_max])

            for k in range(n_words-1, 0, -1):
                u_max = backpointer[k, v_max]
                tags.append(self.idx2tag[u_max])
                v_max = u_max

        if NGRAMM == 3:
            # Initialize pi table and backpointer table
            pi = np.zeros((n_words, n_tags, n_tags)) + float('-inf')
            backpointer = np.zeros((n_words, n_tags, n_tags), dtype=int)

            # Base case initialization
            for t in self.all_tags:
                for u in self.all_tags:                
                    pi[0, self.tag2idx[t], self.tag2idx[u]] = self.log_unigrams[self.tag2idx[t]] + self.log_bigrams[self.tag2idx[t], self.tag2idx[u]] + self.log_new_emissions[self.tag2idx[u], self.new_word2idx[sequence[0]]]

            # Viterbi recursion
            for k in range(1, n_words):
                for u in self.all_tags:
                    for v in self.all_tags:
                        max_prob = float('-inf')
                        max_tag = None
                        for w in self.all_tags:
                            score = pi[k-1, self.tag2idx[w], self.tag2idx[u]] + self.log_trigrams[self.tag2idx[w], self.tag2idx[u], self.tag2idx[v]] + self.log_new_emissions[self.tag2idx[v], self.new_word2idx[sequence[k]]]
                            if score > max_prob:
                                max_prob = score
                                max_tag = w
                        pi[k, self.tag2idx[u], self.tag2idx[v]] = max_prob
                        backpointer[k, self.tag2idx[u], self.tag2idx[v]] = self.tag2idx[max_tag]

            # Backtrack to find the best path
            tags = []
            u_max, v_max = np.unravel_index(np.argmax(pi[n_words-1]), (n_tags, n_tags))
            tags.append(self.idx2tag[v_max])
            tags.append(self.idx2tag[u_max])

            for k in range(n_words-1, 1, -1):
                w_max = backpointer[k, u_max, v_max]
                tags.append(self.idx2tag[w_max])
                v_max, u_max = u_max, w_max

        if NGRAMM == 4:
            # Initialize pi table and backpointer table
            pi = np.zeros((n_words, n_tags, n_tags, n_tags)) + float('-inf')
            backpointer = np.zeros((n_words, n_tags, n_tags, n_tags), dtype=int)

            # Base case initialization
            for s in self.all_tags:
                for t in self.all_tags:
                    for u in self.all_tags:
                        pi[0, self.tag2idx[s], self.tag2idx[t], self.tag2idx[u]] = self.log_unigrams[self.tag2idx[s]] + self.log_bigrams[self.tag2idx[s], self.tag2idx[t]] + self.log_trigrams[self.tag2idx[s], self.tag2idx[t], self.tag2idx[u]] + self.log_new_emissions[self.tag2idx[u], self.new_word2idx[sequence[0]]]

            # Viterbi recursion for four-gram
            for k in range(1, n_words):
                for u in self.all_tags:
                    for v in self.all_tags:
                        for w in self.all_tags:
                            max_prob = float('-inf')
                            max_tag = None
                            for x in self.all_tags:
                                score = pi[k-1, self.tag2idx[x], self.tag2idx[u], self.tag2idx[v]] + self.log_fourgrams[self.tag2idx[x], self.tag2idx[u], self.tag2idx[v], self.tag2idx[w]] + self.log_new_emissions[self.tag2idx[w], self.new_word2idx[sequence[k]]]
                                if score > max_prob:
                                    max_prob = score
                                    max_tag = x
                            pi[k, self.tag2idx[u], self.tag2idx[v], self.tag2idx[w]] = max_prob
                            backpointer[k, self.tag2idx[u], self.tag2idx[v], self.tag2idx[w]] = self.tag2idx[max_tag]

            # Backtrack to find the best path for four-gram
            tags = []
            s_max, t_max, u_max = np.unravel_index(np.argmax(pi[n_words-1]), (n_tags, n_tags, n_tags))
            tags.append(self.idx2tag[u_max])
            tags.append(self.idx2tag[t_max])
            tags.append(self.idx2tag[s_max])

            for k in range(n_words-1, 2, -1):
                x_max = backpointer[k, s_max, t_max, u_max]
                tags.append(self.idx2tag[x_max])
                u_max, t_max, s_max = t_max, s_max, x_max

        return list(reversed(tags))


    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        # self.infer_count += 1
        # print(self.infer_count)
        if INFERENCE == GREEDY:
            return self.greedy_search(sequence)
        elif INFERENCE == BEAM:
            return self.beam_search(sequence)
        else:
            return self.viterbi(sequence)

        


if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data)

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    # evaluate((dev_data[0][:24], dev_data[1][:24]), pos_tagger)
    evaluate(dev_data, pos_tagger)

    # predict(test_data, pos_tagger)
    
    # Write them to a file to update the leaderboard
    # TODO
