from nltk.corpus import brown
import sys
import pickle
sents = brown.tagged_sents()
def smooth(freq):
	smoothed_freq = {}
	freq_freq = {}
	for ngram in freq.keys():
		f = freq[ngram]
		if f in freq_freq.keys():
			freq_freq[f] = freq_freq[f] + 1
		else:
			freq_freq[f] = 1
	freq_sort = sorted(freq_freq.items())
	highest = freq_sort[-1][0]
	breakpoint = 0
	for val in range(1,highest+1):
		if val not in freq_freq.keys():
			breakpoint = val
			break
#print breakpoint
	peak_for_stop = freq_freq[breakpoint - 1]
	for val in range(breakpoint,highest + 2):
		freq_freq[val] = peak_for_stop * pow(0.98,val-breakpoint)
#	print freq_freq	
	for ngram in freq.keys():
		r = freq[ngram]
#		if r+1 not in freq_freq.keys():
#			freq_freq[r+1] = 0
		smoothed_freq[ngram] = float((r+1)*freq_freq[r+1])/float(freq_freq[r])
	return smoothed_freq


sents = sents[:3000]
word_tag = {}
tag_count = {}
tag_bigram = {}
cnt = 0
for sent in sents:
    cnt += 1
    sys.stdout.write('\rpercentage complete: '+ str(float(cnt*100)/float(len(sents))))
    for w_t in sent:
        if w_t not in word_tag.keys():
            word_tag[w_t] = 1
        else:
            word_tag[w_t] += 1
        tag = w_t[1]
        if tag  not in tag_count.keys():
            tag_count[tag] = 1
        else:
            tag_count[tag] += 1
    tag_list = [w_t[1] for w_t in sent]
    tag_bigrams = zip(*[tag_list[i:] for i in range(2)])
    for t_b in tag_bigrams:
        if t_b not in tag_bigram.keys():
            tag_bigram[t_b] = 1
        else:
            tag_bigram[t_b] += 1
print word_tag
print tag_count
print tag_bigram
print len(tag_count.keys())
pickle.dump(word_tag, open('word_tag_2','w'))
pickle.dump(tag_count, open('tag_count_2','w'))
pickle.dump(tag_bigram, open('tag_bigram_2','w'))
# pickle.dump(smooth(word_tag), open('word_tag_smoothed','w'))
# pickle.dump(smooth(tag_bigram), open('tag_bigram_smoothed','w'))
