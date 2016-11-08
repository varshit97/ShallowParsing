import pickle
import sys
from nltk.corpus import brown
def smooth(freq, heldout):
        unseen = 0
        for i in heldout:
            if i not in freq.keys():
                unseen += 1
        smoothed_freq = {}
        freq_freq = {}
        for ngram in freq.keys():
            f = freq[ngram]
            if f in freq_freq.keys():
                freq_freq[f] = freq_freq[f] + 1
            else:
                freq_freq[f] = 1
        unseen_freq = float(freq_freq[1]/float(unseen))
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
            freq_freq[val] = peak_for_stop * pow(0.992,val-breakpoint)
#	print freq_freq	
        for ngram in freq.keys():
            r = freq[ngram]
#		if r+1 not in freq_freq.keys():
#			freq_freq[r+1] = 0
            smoothed_freq[ngram] = float((r+1)*freq_freq[r+1])/float(freq_freq[r])
        smoothed_freq['<unseen>'] = unseen_freq
        return smoothed_freq

sents = brown.tagged_sents()
sents = sents[30001:32000]
held_out = []
word_tag_heldout = []
for sent in sents:
    tag_list = [w_t[1] for w_t in sent]
    bigrams = zip(*[tag_list[i:] for i in range(2)])
    held_out += bigrams
    word_tag_heldout += sent

# print held_out
# print word_tag_heldout
# sys.exit(0)

word_tag = pickle.load(open('word_tag','r'))
tag_count = pickle.load(open('tag_count','r'))
tag_bigram = pickle.load(open('tag_bigram','r'))
word_tag = smooth(word_tag, held_out)
tag_bigram = smooth(tag_bigram, word_tag_heldout)
# sys.exit(0)
store = sents
sents = [[w[0] for w in sent] for sent in sents]
sents = sents[:1]
# print len(sents)
# print len(tag_count)
p_last_tag = 1
for i in range(len(sents[0])-1, -1, -1):
    print (sents[0][i])

for sent in sents:
    final_prob = {}
    last_tag = '<s>'
    final_prob = dict(zip([i for i in range(len(sent))], [{}]*len(sent)))
    # print final_prob.keys()
    # sys.exit(0)
    c = 0
    for w in sent:
        p = {}
        w_last = ''
        mx_tag = -1
        for t in tag_count.keys():
            # print t, tag_count[t]
            if tuple((w,t)) in word_tag.keys():
                # print 'YOYO'
                p[t] = word_tag[tuple((w,t))]
            else:
                p[t] = word_tag['<unseen>']
            # for t_last in tag_count.keys():
            #     pass
            ptt = {}
            mx = -1
            mx_last = ''

            for t_last in tag_count.keys():
                if last_tag == '<s>':
                    ptt[tuple((t_last, t))] = 1
                elif tuple((t_last,t)) in tag_bigram.keys():
                    # print 'yes'
                    ptt[tuple((t_last, t))] = tag_bigram[tuple((t_last, t))]
                else:
                    ptt[tuple((t_last, t))] = tag_bigram['<unseen>']
                # print w, t, t_last, p[t], ptt[tuple((t_last, t))]
                pr = float(p[t])*float(ptt[tuple((t_last, t))])
                if pr > mx:
                    mx = pr
                    mx_last = t_last
            print (w, t, mx_last, mx)
            final_prob[c][t] = mx_last 
            if mx > mx_tag:
                mx_tag = mx
                w_last = t
        c+=1
        last_tag = '!<s>'
    # print w_last
    # print final_prob[len(sent)-1][w_last]
    for i in range(len(sent)-1, -1, -1):
        print (sent[i], w_last)
        w_last = final_prob[i][w_last]

'''
word_tag = pickle.load(open('word_tag','r'))
tag_count = pickle.load(open('tag_count','r'))
tag_bigram = pickle.load(open('tag_bigram','r'))
print smooth(word_tag)
print smooth(tag_count)
print smooth(tag_bigram)
'''
