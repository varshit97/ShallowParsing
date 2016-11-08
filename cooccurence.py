from nltk.corpus import brown
import numpy as np
import collections
i = 1
index = {}
index[''] = 0
words_used = ['' for _ in range(100)]
for word in brown.words():
    words_used.append(word)
    if word not in index:
        index[word] = i
        i+=1
    if i == 1000:
        break
words_used += ['' for _ in range(100)]
print len(words_used), words_used
window_size = 200
cooccurrence_matrix = [[0 for _ in range(1000)] for __ in range(1000)]
buffer = collections.deque(maxlen=window_size)
for word in words_used[:window_size]:
    buffer.append(word)
word_vector = np.array([0 for _ in range(1000)])
for window_word in buffer:
    # print window_word
    window_word_idx = index[window_word]
    word_vector[window_word_idx] = 1
for i, word in enumerate(words_used[100:-100]):
    word_idx = index[word]
    cur_vector = np.asarray(cooccurrence_matrix[word_idx])
    cooccurrence_matrix[word_idx] = list(cur_vector + word_vector)
    rem_word = buffer[0]
    word_vector[index[rem_word]] = 0
    add_word = words_used[100 + i + 100]
    word_vector[index[add_word]] = 1
    buffer.append(add_word)
    print len(buffer)

from sklearn.cluster import KMeans
X = np.array(cooccurrence_matrix)
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
print kmeans.labels_
