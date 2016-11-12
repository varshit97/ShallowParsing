from nltk.corpus import brown
import numpy as np
import collections
i = 1
index = {}
rev_index = ['']
index[''] = 0

VOCAB_SIZE = 100
WIN_SIZE = 20

words_used = ['' for _ in range(WIN_SIZE/2)]
for word in brown.words():
    words_used.append(word)
    if word not in index:
        index[word] = i
        rev_index.append(word)
        i+=1
    if i == VOCAB_SIZE:
        break
words_used += ['' for _ in range(WIN_SIZE/2)]
cooccurrence_matrix = [[0 for _ in range(VOCAB_SIZE)] for __ in range(VOCAB_SIZE)]
buffer = collections.deque(maxlen=WIN_SIZE)
for word in words_used[:WIN_SIZE]:
    buffer.append(word)
word_vector = np.array([0 for _ in range(VOCAB_SIZE)])
for window_word in buffer:
    # print window_word
    window_word_idx = index[window_word]
    word_vector[window_word_idx] = 1
for i, word in enumerate(words_used[WIN_SIZE/2:-WIN_SIZE/2]):
    word_idx = index[word]
    cur_vector = np.asarray(cooccurrence_matrix[word_idx])
    cooccurrence_matrix[word_idx] = list(cur_vector + word_vector)
    rem_word = buffer[0]
    word_vector[index[rem_word]] = 0
    add_word = words_used[WIN_SIZE/2 + i + WIN_SIZE/2]
    word_vector[index[add_word]] = 1
    buffer.append(add_word)


from sklearn.cluster import KMeans
X = np.array(cooccurrence_matrix)
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
clusters = kmeans.labels_
cluster_means = kmeans.cluster_centers_


#HMM class which can be reused
class HMM:

    def __init__(self):
        self.transition_prob = None

    def fit(self, seq1, seq2, emission_obj):

        #A function must be provided which can give emission probabilities
        assert 'get_emission_prob' in [method for method in dir(emission_obj) if callable(getattr(emission_obj, method))]
        assert len(seq1) == len(seq2)
        
        self.transition_prob = {}#np.array([[0 for __ in range(len(seq2))] for _ in range(len(seq2))])
        count = {}
        for i, label in enumerate(seq2[:-1]):
            transition = tuple((label, seq2[i+1]))
            if transition in self.transition_prob:
                self.transition_prob[transition] += 1
            else:
                self.transition_prob[transition] = 1
            if label in count:
                count[label] += 1
            else:
                count[label] = 1
        if count[seq2[-1]] in count:
            count[seq2[-1]] +=1
        else:
            count[seq2[-1]] = 1
        for trans in self.transition_prob:
            self.transition_prob[trans] = float(self.transition_prob[trans])/float(count[trans[0]])
            print trans, self.transition_prob[trans]

class w2k_emission:

    def __init__(self, word_embeddings, cluster_means):
        self.word_embeddings = word_embeddings
        self.cluster_means = cluster_means

    def get_emission_prob(self,w_id, k_id):
        import numpy as np
        w_embed = self.word_embeddings[w_id]
        c_mean = self.cluster_means[k_id]
        return np.linalg.norm(np.array(w_embed) - np.array(c_mean))

indexed_seq = [index[w] for w in words_used[WIN_SIZE/2:-WIN_SIZE/2]]
cluster_tag = [clusters[index[w]] for w in words_used[WIN_SIZE/2:-WIN_SIZE/2]]
w2k = w2k_emission(cooccurrence_matrix, cluster_means)

w2k_hmm = HMM().fit(indexed_seq, cluster_tag, w2k)
