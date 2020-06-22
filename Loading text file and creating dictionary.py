#filename = 'C:/Users/Arun/Documents/Image Captioning/Flickr8k.token.txt'
filename = '/content/drive/My Drive/Image Captioning Data/Text Data/Flickr8k.token.txt'

def load_doc(filename):
    file = open(filename, 'r')
    doc = file.read()
    return doc

doc = load_doc(filename)

# Creating a dictionary of keys as names of the images and value as caption
def load_description(text):
    mapping = dict()

    for line in text.split('\n'):
    
        tokens = line.split()
        
        if len(line) < 2:
            continue
    
        # take the first token as image id, the rest as description
        image_id, image_desc = tokens[0], tokens[1:]
    
        # extract filename from image id
        image_id = image_id.split('.')[0]
        
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
            mapping[image_id].append(image_desc)
            
    return mapping
    
descriptions = load_description(doc)