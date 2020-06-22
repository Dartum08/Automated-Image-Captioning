def load_clean_descriptions(filename):

  train_doc = load_doc(filename)
  train_text = list()

  for line in train_doc.split('\n'):
      identifier = line.split('.')[0]
      train_text.append(identifier)

  train_desc = dict()

  for txt in train_text:
    
      if txt in descriptions:
        
          if txt not in train_desc:
              train_desc[txt] = []
            
          for desc in descriptions[txt]:
              # wrap description in tokens
              train_desc[txt].append('sos ' + desc + ' eos')

  return train_text, train_desc

### Loading training image text file
filename = '/content/drive/My Drive/Image Captioning Data/Text Data/Flickr_8k.trainImages.txt'

train_text, train_desc = load_clean_descriptions(filename)

print('Dataset: %d' % len(train_text))
#Dataset: 6001

def load_clean_descriptions_test(filename):

  train_doc = load_doc(filename)
  train_text = list()

  for line in train_doc.split('\n'):
      identifier = line.split('.')[0]
      train_text.append(identifier)

  train_desc = dict()

  for txt in train_text:
    
      if txt in descriptions:
        
          if txt not in train_desc:
              train_desc[txt] = []
            
          for desc in descriptions[txt]:
              # wrap description in tokens
              train_desc[txt].append(desc)

  return train_text, train_desc

# Loading validation descriptions
# Loading val_image text file

filename = '/content/drive/My Drive/Image Captioning Data/Text Data/Flickr_8k.devImages.txt'

val_text, val_desc = load_clean_descriptions_test(filename)

print('Dataset: %d' % len(val_text))

# Loading test descriptions
# Loading test_image text file
filename = '/content/drive/My Drive/Image Captioning Data/Text Data/Flickr_8k.testImages.txt'

test_text, test_desc = load_clean_descriptions_test(filename)

print('Dataset: %d' % len(test_text))
