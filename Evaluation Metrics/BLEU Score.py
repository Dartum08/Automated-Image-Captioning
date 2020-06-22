candidates = []
for z in range(1000):
  pic = list(encoding_test.keys())[z]
  img = encoding_test[pic]
  image = encoding_test[pic].reshape((1,2048))
  candidates += [beam_search(image, model).split()]

  #print("Generated Caption: ", z, " : ", greedySearch(image))
# Good Predictions: z = 4, 6, 14, 18, 23, 28, 62
# Very Bad Predictions: z = 10, 11, 12, 17, 27, 32, 36
  
  
references = []
for key, lis in test_desc.items():
  
  refs = []

  for ref_text in test_desc[key]:

    refs += [ref_text.split()]

  references.append(refs)
  
  
# Calculating Corpus Bleu Score for beam search with k = 5
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

corpus_bleu(references, candidates)
