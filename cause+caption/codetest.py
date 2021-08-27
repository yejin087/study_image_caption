import pickle
import processData

vocab_path = './data/annotaions/uniq_vocab.pkl'
caption = pickle.load(open('./data/annotations/vocab.pkl','rb'))
uni = pickle.load(open('./data/annotations/unique_vocab.pkl','rb'))

caption_word_list = caption.idx2word.values() 
uni_word_list = uni.idx2word.values()

print(len(caption_word_list), len(uni_word_list))

caption_set = set()
for word in caption_word_list:
    caption_set.add(word)

uni_set = set()
for word in uni_word_list:
    uni_set.add(word)
    
diff = caption_set.difference(uni_set)
print( diff )

