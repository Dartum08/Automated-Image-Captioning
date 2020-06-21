# Get the InceptionV3 model trained on imagenet data
in_layer = Input(shape=(229, 229, 3))
inc_model = InceptionV3(weights='imagenet',input_tensor=in_layer)

# Remove the last layer (output softmax layer) from the inception v3
model_new = Model(inc_model.input, inc_model.layers[-2].output)

### It will take a lot of time to run (approx. 1 - 2 hrs)
# extract features from each photo in the directory

def extract_features(filepath):
    # load the model
    in_layer = Input(shape=(229, 229, 3))
    inc_model = InceptionV3(weights='imagenet',input_tensor=in_layer)
    model_new = Model(inc_model.input, inc_model.layers[-2].output)
    #print(model.summary())
    # extract features from each photo
    file = os.listdir(filepath)
    features = dict()
    
    for name in file:
        
        img = image.load_img(os.path.join(filepath, name), target_size=(229, 229))
        img = image.img_to_array(img)
		 # reshape data for the model
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
		 # prepare the image for the inception model
        img = preprocess_input(img)
		 # get features
        feature = model_new.predict(img, verbose=0)
        # Reshaping feature from (1, 2048) to (2048, )
        feature = np.reshape(feature, feature.shape[1])
		 # get image id
        image_id = name.split('.')[0]
		 # store feature
        features[image_id] = feature
        print('>%s' % name)
        
    return features

# extract features from all images
# Warning: It will take a lot of time (1 - 3 hrs)
directory = '/content/drive/My Drive/Image Captioning Data/Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))

# save to file
with open(r"C:\Users\Arun\Documents\Image Captioning\Pickle\encoded_images.pl", "wb") as encoded_pickle:
    pickle.dump(features, encoded_pickle)
    
features = load(open("/content/drive/My Drive/Image Captioning Data/Pickle/encoded_images.pl", "rb"))
print('Photos: train=%d' % len(features))
# 8091

# Encoding Train Images

encoding_train = {}

for train_img in train_desc.keys():

    encoding_train[train_img] = features[train_img]

# save to file
with open(r"C:\Users\Arun\Documents\Image Captioning\Pickle\train_images_encoded.pl", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)
    

train_features = load(open("/content/drive/My Drive/Image Captioning Data/Pickle/train_images_encoded.pl", "rb"))
print('Photos: train=%d' % len(train_features))
# 6000

# Encoding dev set images

encoding_val = {}

for val_img in val_desc.keys():

    encoding_val[val_img] = features[val_img]

with open(r"/content/drive/My Drive/Image Captioning Data/Pickle/dev_images_encoded.pl", "wb") as encoded_pickle:
    pickle.dump(encoding_val, encoded_pickle)
    
# Loading dev images features

val_features = load(open("/content/drive/My Drive/Image Captioning Data/Pickle/dev_images_encoded.pl", "rb"))
print('Photos: dev=%d' % len(val_features))
# 1000

# Encoding Test Images
    
encoding_test = {}

for test_img in test_desc.keys():

    encoding_test[test_img] = features[test_img]