import numpy as np
import multiprocessing as mp
import random
import functools
import ctypes

def glove_compute_and_update_grads(coo_list, Warr, Garr, shape, cost, count, x_max, alpha, initial_learning_rate):

    Wall = np.frombuffer(Warr)
    Wall = Wall.reshape(shape)
    W = Wall[:,:,:-1]
    B = Wall[:,:,-1]
    
    Gall = np.frombuffer(Garr)
    Gall = Gall.reshape(shape)
    Gw = Gall[:,:,:-1]
    Gb = Gall[:,:,-1]
    
    cost.value = 0.0
    count.value = 0
    
    for wid_pair, val in coo_list: # cr is coo record of ((target_wid, context_wid), val)
        wid_target, wid_context = wid_pair
        diff = np.dot(W[0][wid_target],W[1][wid_context])
        diff +=  B[0][wid_target] + B[1][wid_context] - np.log(val)
        fdiff = diff if val > x_max else np.power(val/x_max, alpha) * diff
        if True in [ np.isnan(d) or np.isinf(d) for d in (diff, fdiff)]:
            continue
        cost.value += 0.5 * fdiff * fdiff
        count.value += 1

        grad_w0 = np.clip(fdiff*W[1][wid_context],-100,100) * initial_learning_rate # initial gradient
        grad_w1 = np.clip(fdiff*W[0][wid_target],-100,100) * initial_learning_rate # initial gradient
        upd_w0 = grad_w0 / np.sqrt(Gw[0][wid_target]) # adagrad adjustment
        upd_w1 = grad_w1 / np.sqrt(Gw[1][wid_context]) # adagrad adjustment
        Gw[0][wid_target] += np.square(grad_w0)
        Gw[1][wid_context] += np.square(grad_w1)

        if True not in [np.isnan(upd_val) or np.isinf(upd_val) for upd_val in (np.sum(upd_w0), np.sum(upd_w1))]:
            W[0][wid_target] -= upd_w0
            W[1][wid_context] -= upd_w1

        upd_b0 = fdiff/np.sqrt(Gb[0][wid_target])
        upd_b1 = fdiff/np.sqrt(Gb[1][wid_context])
        Gb[0][wid_target] += np.square(fdiff)
        Gb[1][wid_context] += np.square(fdiff)

        if True not in [np.isnan(upd_val) or np.isinf(upd_val) for upd_val in (upd_b0, upd_b1)]:
            B[0][wid_target] -= upd_b0
            B[1][wid_context] -= upd_b1

class Glove(object):
    """
    Class for GloVe word embeddings implemented only on python
    """
    def __init__(self, sentences, num_component, min_count=1, max_vocab=0, window_size=15, distance_weighting=True, window_range=None):
        self.num_component = num_component
        self.check_sentences(sentences, window_range)
        self.build_vocabulary(sentences, min_count, max_vocab)
        if window_range is None: # cooccurrence counting based on window_size of integer, distance is 1 between adjacent words
            self.count_cooccurrence(sentences, window_size, distance_weighting)
        else:
            self.count_cooccurrence_range(sentences, window_range, distance_weighting)
        self.initialize_weights()
    
    def check_sentences(self, sentences, window_range):
        if window_range is None:
            return
        for sentence in sentences:
            for word in sentence:
                if type(word) is not tuple or len(word) != 2:
                    raise ValueError("word {0} is not acceptable to window range mode, need a pair of (str, float)".format(word))
    
    def build_vocabulary(self, sentences, min_count=1, max_vocab=0):
        word_dict = {}
        for sentence in sentences:
            for word in sentence:
                if type(word) is tuple:
                    word = word[0]
                if word_dict.get(word) is None:
                    word_dict[word] = 0
                word_dict[word] += 1
        self.word_count = [(w,c) for w,c in word_dict.items() if c >= min_count]
        self.word_count.sort(key=functools.cmp_to_key(lambda lhs,rhs : rhs[1] - lhs[1] if rhs[1] != lhs[1] else -1 if lhs[0] < rhs[0] else 1 if lhs[0] > rhs[0] else 0))
        if max_vocab > 0:
            self.word_count = self.word_count[0:max_vocab]
        
        self.word_to_wid = { wc[0]:i for i, wc in enumerate(self.word_count) }
        self.wid_to_word = { i:wc[0] for i, wc in enumerate(self.word_count) }
        self.shape = (2, len(self.word_count), self.num_component+1)
    
    def count_cooccurrence(self, sentences, window_size, distance_weighting):
        coo = {}
        for sentence in sentences:
            words = sentence
            for ti, word in enumerate(sentence):
                if self.word_to_wid.get(word) is None:
                    continue
                wid_target = self.word_to_wid[word]
                for ci in range(max(ti-window_size,0),ti): # for words left to target word within window
                    if self.word_to_wid.get(words[ci]) is None:
                        continue
                    wid_context = self.word_to_wid[words[ci]]
                    if wid_target == wid_context:
                        continue
                    key = (wid_target,wid_context)
                    if coo.get(key) is None:
                        coo[key] = 0.0
                    weight = 1.0/(ti-ci) if distance_weighting else 1.0
                    coo[key] += weight
                    rkey = (wid_context,wid_target)
                    if coo.get(rkey) is None:
                        coo[rkey] = 0.0
                    coo[rkey] += weight
        self.coo_records = list(coo.items())
        random.shuffle(self.coo_records)
    
    def count_cooccurrence_range(self, sentences, window_range, distance_weighting):
        coo = {}
        for sentence in sentences:
            words = sentence
            lb_ci = 0
            for ti, pair in enumerate(sentence):
                word_target, word_target_value = pair
                if self.word_to_wid.get(word_target) is None:
                    continue
                wid_target = self.word_to_wid[word_target]
                search_range = range(lb_ci,ti)
                for ci in search_range:
                    word_context, word_context_value = words[ci]
                    if word_target_value - word_context_value > window_range:
                        lb_ci = ci+1
                        continue
                    if self.word_to_wid.get(word_context) is None:
                        continue
                    wid_context = self.word_to_wid[word_context]
                    if wid_target == wid_context:
                        continue
                    key = (wid_target,wid_context)
                    weight = (word_context_value - word_target_value + window_range)/window_range if distance_weighting else 1.0
                    if coo.get(key) is None:
                        coo[key] = 0.0
                    coo[key] += weight
                    rkey = (wid_context,wid_target)
                    if coo.get(rkey) is None:
                        coo[rkey] = 0.0
                    coo[rkey] += weight
        self.coo_records = list(coo.items())
        random.shuffle(self.coo_records)
    
    def initialize_weights(self):
        num_elem = int(np.prod(self.shape))
        self.Warr = mp.RawArray(ctypes.c_double, num_elem)
        Wall = np.frombuffer(self.Warr)
        Wall[:] = (np.random.rand(len(Wall)) - 0.5)/self.num_component # a pair of word_vector and bias # (100+1)*2
        Wall = Wall.reshape(*self.shape)
        self.W = Wall[:,:,:-1]
        self.B = Wall[:,:,-1]
        
        self.Garr = mp.RawArray(ctypes.c_double, num_elem)
        Gall = np.frombuffer(self.Garr)
        Gall[:] = np.ones(len(Gall))
        Gall = Gall.reshape(*self.shape)
        self.Gw = Gall[:,:,:-1]
        self.Gb = Gall[:,:,-1]
    
    def fit(self, force_initialize=False, num_iteration=50, num_procs=8, x_max=100, alpha=0.75, learning_rate=0.05, verbose=False):
        if verbose:
            print("training parameters = {}".format(dict(locals())))
        
        if force_initialize:
            initialize_weights()
        
        history = {'loss':[]}
        coo_list = self.coo_records
        for iter in range(num_iteration):
            if verbose:
                print("iteration # %d ... " % iter, end="")
            cost_list = [mp.Value('d', 0.0) for i in range(num_procs)]
            count_list = [mp.Value('i', 0) for i in range(num_procs)]

            arguments = [ (coo_list[rank*len(coo_list)//num_procs:(rank+1)*len(coo_list)//num_procs], self.Warr, self.Garr, self.shape, cost_list[rank], count_list[rank], x_max, alpha, learning_rate) for rank in range(num_procs)]
            procs = [mp.Process(target=glove_compute_and_update_grads, args=arguments[rank]) for rank in range(num_procs)]
            [proc.start() for proc in procs]
            [proc.join() for proc in procs]

            cost = 0
            count = 0
            for i in range(num_procs):
                cost += cost_list[i].value
                count += count_list[i].value
            history['loss'].append(cost/count)
            if verbose:
                print("loss = %f" % history['loss'][-1])
        self.word_vector = self.W[0] + self.W[1]
        return history
    
    def most_similar(self, word, number=5):
        wid = self.word_to_wid[word]
        word_vec = self.word_vector[wid]

        dst = (np.dot(self.word_vector, word_vec)
               / np.linalg.norm(self.word_vector, axis=1)
               / np.linalg.norm(word_vec))
        word_ids = np.argsort(-dst)

        return [(self.wid_to_word[x], dst[x]) for x in word_ids[:number] if x in self.wid_to_word][1:]