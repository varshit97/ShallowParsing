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
        self.labels = None
        self.emission_obj = None

    def fit(self, seq1, seq2, emission_obj):

        #A function must be provided which can give emission probabilities
        assert 'get_emission_prob' in [method for method in dir(emission_obj) if callable(getattr(emission_obj, method))]
        assert len(seq1) == len(seq2)
        
        self.emission_obj = emission_obj
        self.transition_prob = {}  #Instead of dictionary implement it id wise for better performance
                                   #np.array([[0 for __ in range(len(seq2))] for _ in range(len(seq2))])
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
        self.labels = count.keys()
    
    def predict(self, seq):
        prob_prev = [1 for _ in range(len(self.labels))]
        back_track_mat = []
        
        for w in seq:
            back_track_vec = []
            prob_cur = []
            for l in self.labels:
                emission_prob = self.emission_obj.get_emission_prob(w,l)
                selected_transition = None
                prob = -1
                for l_prev in self.labels:
                    transition = tuple((l_prev, l))
                    transition_prob = 0
                    if transition in self.transition_prob:
                        transition_prob = self.transition_prob[transition]
                    label_prob = emission_prob * transition_prob * prob_prev[l_prev]
                    if label_prob > prob:
                        prob = label_prob
                        selected_transition = l_prev
                back_track_vec.append(selected_transition)
                prob_cur.append(prob)
            back_track_mat.append(back_track_vec)
            prob_prev = prob_cur[:]
        idx = prob_prev.index(max(prob_prev))
        labeled_seq = [idx]
        
        for b in reversed(back_track_mat):
            idx = b[idx]
            labeled_seq = [idx] + labeled_seq
        labeled_seq =  labeled_seq[1:]
        
        return labeled_seq

class w2k_emission:

    def __init__(self, word_embeddings, cluster_means):
        self.word_embeddings = word_embeddings
        self.cluster_means = cluster_means

    def get_emission_prob(self,w_id, k_id):
        import numpy as np
        w_embed = self.word_embeddings[w_id]
        c_mean = self.cluster_means[k_id]
        return np.linalg.norm(np.array(w_embed) - np.array(c_mean))

class k2t_emission:

    def __init__(self, tag_embeddings, cluster_means):
        self.tag_embeddings = tag_embeddings
        self.cluster_means = cluster_means
        
    def get_emission_prob(self, k_id, t_id):
        import numpy as np
        t_embed = self.tag_embeddings[t_id]
        c_mean = self.cluster_means[k_id]
        return np.linalg.norm(np.array(t_embed) - np.array(c_mean))

indexed_seq = [index[w] for w in words_used[WIN_SIZE/2:-WIN_SIZE/2]]
cluster_tag = [clusters[index[w]] for w in words_used[WIN_SIZE/2:-WIN_SIZE/2]]

w2k = w2k_emission(cooccurrence_matrix, cluster_means)
w2k_hmm = HMM()
w2k_hmm.fit(indexed_seq, cluster_tag, w2k)
cluster_seq = w2k_hmm.predict(indexed_seq[:10])

from nltk import pos_tag
tagged_words = pos_tag(words_used[WIN_SIZE/2:-WIN_SIZE/2])
#Calculate tag embeddings as mean of word embeddings
t_embed = []
t_index = {}
t_count = []
ind = 0
for word, tag in tagged_words:
    w_embed = cooccurrence_matrix[index[word]]
    if tag in t_index:
        t_id = t_index[tag]
        t_embed[t_id] = list(np.array(t_embed[t_id]) + np.array(w_embed))
        assert len(t_embed[t_id]) == VOCAB_SIZE
        t_count[t_id]+=1
    else:
        t_index[tag] = ind
        t_embed.append(w_embed)
        t_count.append(1)
        ind+=1

t_embed = [list(np.array(t_embed[t_id])/float(t_count[t_id])) for t_id in range(len(t_count))]
t_indexed_seq = [t_index[t[1]] for t in tagged_words]

k2t = k2t_emission(t_embed, cluster_means)
k2t_hmm = HMM()
k2t_hmm.fit(cluster_tag, t_indexed_seq, k2t)
final_tag_seq = k2t_hmm.predict(cluster_seq)
print final_tag_seq
