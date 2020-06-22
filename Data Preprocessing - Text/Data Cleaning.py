# prepare translation table for removing punctuation
table = str.maketrans('', '', string.punctuation)

for key, desc_list in descriptions.items():
    for i in range(len(desc_list)):
        desc = desc_list[i]
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each token
        desc = [w.translate(table) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word)>1]
        # remove tokens with numbers in them
        desc = [word for word in desc if word.isalpha()]
        # store as string
        desc_list[i] =  ' '.join(desc)
        
        
vocabulary = set()
for key in descriptions.keys():
    [vocabulary.update(d.split()) for d in descriptions[key]]
print('Original Vocabulary Size: %d' % len(vocabulary))
# 8763

# Create a list of all the training captions
all_train_captions = []
for key, val in train_desc.items():
    for cap in val:
        all_train_captions.append(cap)

# Consider only words which occur at least 10 times in the corpus
'''word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

print('preprocessed words %d ' % len(vocab))
print(len(word_counts))'''

words = [i.split() for i in all_train_captions]
unique = []
for i in words:
    unique.extend(i)

unique = list(set(unique))

# len(unique) = 7578

# Creating index to word and word to index dictionary
ix_to_word = {}
word_to_ix = {}

ix = 1
for w in unique:
    word_to_ix[w] = ix
    ix_to_word[ix] = w
    ix += 1
    

vocab_size = len(ix_to_word) + 1 # one for appended 0's
vocab_size

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# determine the maximum sequence length
max_length = max_length(train_desc)
print('Description Length: %d' % max_length)
## max_length = 34
