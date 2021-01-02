import os
import numpy as np
import logging
import string
import tqdm
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

logger = tf.get_logger()
logger.setLevel(logging.INFO)

class ImageDataset:

    def __init__(self, config):
        self.config = config
        #self.tracker = tracker

        self.features = self.prepare_img_features()

    ### It will take a lot of time to run (approx. 1 - 2 hrs)
    # extract features from each photo in the directory
    def extract_img_features(self):

        logger.info('Extracting features from each image. It will take approx 1 - 2 hrs depending on the machine. Go have a cup of coffee or something.')

        filepath = self.config['Dataset']['image']['data_folder']

        model_path = self.config['Dataset']['image']['model_path']
        model = load_model(model_path)

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
            feature = model.predict(img, verbose=0)
            # Reshaping feature from (1, 2048) to (2048, )
            feature = np.reshape(feature, feature.shape[1])
            # get image id
            image_id = name.split('.')[0]
            # store feature
            features[image_id] = feature
            #print('>%s' % name)

        logger.info('Features extracted from each image')

        return features

    def prepare_img_features(self):

        if self.config['Dataset']['image']['extract_features']:

            self.features = self.extract_img_features()

            return self.features

        feature_path = self.config['Dataset']['image']['feature_path']

        features = pickle.load(open(feature_path, "rb"))

        logger.info('Extracted Image features loaded from {}'.format(feature_path))

        return features


class TextDataset:

    def __init__(self, config):
        self.config = config
        #self.tracker = tracker

        self.prepare_text_data()
        self.get_dict()
        self.get_max_len()

    def load_description(self):
        """
        Creating a dictionary of keys as names of the images and value as caption
        :param filepath: path of the text data
        :return: dictionary with image id as key and it's descriptions as value
        """

        logger.info('Preparing dictionary of keys as names of the images and value as caption')

        filepath = self.config['Dataset']['text']['data_folder']

        file = open(filepath, 'r')
        text = file.read()

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

        logger.info('Loaded complete description file as a dictionaru')

        return mapping

    def clean_text_data(self, descriptions):
        """
        Clean text file by making all the letters lower case, removing punctuation, etc..
        :param descriptions: Text file loaded from load_description
        :return:
        """

        logger.info('Cleaning text data')

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
                desc = [word for word in desc if len(word) > 1]
                # remove tokens with numbers in them
                desc = [word for word in desc if word.isalpha()]
                # store as string
                desc_list[i] = ' '.join(desc)

        logger.info('Text cleaning complete')

    def load_clean_descriptions_for_training(self, filepath:str, mode:str):

        logger.info('Preparing clean {} description for training or evaluation or testing'.format(mode))

        filename = open(filepath, 'r')
        train_doc = filename.read()

        train_text = list()

        for line in train_doc.split('\n'):
            identifier = line.split('.')[0]
            train_text.append(identifier)

        train_desc = dict()

        for txt in train_text:

            if txt in self.descriptions:

                if txt not in train_desc:
                    train_desc[txt] = []

                for desc in self.descriptions[txt]:
                    # wrap description in tokens
                    if mode == 'training':
                        train_desc[txt].append('sos ' + desc + ' eos')
                    else:
                        train_desc[txt].append(desc)

        logger.info('Loaded {} description of length {:d}'.format(mode, len(train_desc)))

        return train_text, train_desc

    def prepare_text_data(self):

        logger.info('Preparing text data for the experiment')

        self.descriptions = self.load_description()
        self.clean_text_data(self.descriptions)

        train_path = self.config['Dataset']['text']['train_filepath']
        val_path = self.config['Dataset']['text']['val_filepath']
        test_path = self.config['Dataset']['text']['test_filepath']

        self.train_text, self.train_desc = self.load_clean_descriptions_for_training(train_path, 'training')

        self.val_text, self.val_desc = self.load_clean_descriptions_for_training(val_path, 'validation')

        self.test_text, self.test_desc = self.load_clean_descriptions_for_training(test_path, 'testing')

        logger.info('Text data prepared')

    def get_all_train_desc(self):

        all_train_captions = []
        for key, val in self.train_desc.items():
            for cap in val:
                all_train_captions.append(cap)

        return all_train_captions

    def get_max_len(self):

        lines = self.get_all_train_desc()
        self.max_len = max(len(d.split()) for d in lines)

    def get_dict(self):

        logger.info('Creating dictionary of word to index and index to word')

        all_train_captions = self.get_all_train_desc()

        words = [i.split() for i in all_train_captions]
        unique = []
        for i in words:
            unique.extend(i)

        unique = list(set(unique))

        # Creating index to word and word to index dictionary
        self.ix_to_word = {}
        self.word_to_ix = {}

        ix = 1
        for w in unique:
            self.word_to_ix[w] = ix
            self.ix_to_word[ix] = w
            ix += 1

        self.vocab_size = len(self.ix_to_word) + 1  # one for appended 0's

        logger.info('Created. Done!')

    def embedding_vectors(self):

        # Load Glove vectors
        #glove_dir = "/content/drive/My Drive/Image Captioning Data/GloVe"

        glove_dir = self.config['Dataset']['text']['embedding_dir']

        logger.info('Loading embedding vectors from {}'.format(glove_dir))

        embeddings_index = {}
        f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        logger.info('Found %s word vectors.' % len(embeddings_index))
        # Found 400000 word vectors.

    def get_embedding_matrix(self):

        embeddings_index = self.embedding_vectors()

        logger.info('Preparing embedding matrix')

        embedding_dim = self.config['Dataset']['text']['embedding_dim']

        # Get 200-dim dense vector for each of the 10000 words in out vocabulary
        embedding_matrix = np.zeros((self.vocab_size, embedding_dim))

        for word, i in self.word_to_ix.items():
            # if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in the embedding index will be all zeros
                embedding_matrix[i] = embedding_vector

        logger.info('Created embedding matrix with shape {}'.format(embedding_matrix.shape))

        return embedding_matrix


class BaseDataset(TextDataset, ImageDataset):

    def __init__(self, config):
        super().__init__(config)

    def prepare_img_features_from_desc(self):

        logger.info('Extracting train, val and test image features from their respective descriptions')

        self.features = self.prepare_img_features()

        train_features = {}
        for train_img in self.train_desc.keys():
            train_features[train_img] = self.features[train_img]

        val_features = {}
        for train_img in self.val_desc.keys():
            val_features[train_img] = self.features[train_img]

        test_features = {}
        for train_img in self.test_desc.keys():
            test_features[train_img] = self.features[train_img]

        logger.info('Extracted train, val and test image features')

        return train_features, val_features, test_features

    def prepare_data(self):

        number_pics_per_bath = 128
        train_features, val_features, test_features = self.prepare_img_features_from_desc()

        self.train_gen = self.data_generator(self.train_desc, train_features, self.word_to_ix, self.max_len, number_pics_per_bath)
        self.val_gen = self.data_generator(self.val_desc, val_features, self.word_to_ix, self.max_len, number_pics_per_bath)

    # data generator, intended to be used in a call to model.fit()
    def data_generator(self, descriptions, photos, wordtoix, max_length, num_photos_per_batch=32):

        X1, X2, y = list(), list(), list()
        n = 0
        # loop for ever over images
        while 1:
            for key, desc_list in descriptions.items():
                n += 1
                # retrieve the photo feature
                photo = photos[key]
                for desc in desc_list:
                    # encode the sequence
                    seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                    # split one sequence into multiple X, y pairs
                    for i in range(1, len(seq)):
                        # split into input and output pair
                        in_seq, out_seq = seq[:i], seq[i]
                        # pad input sequence
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        # encode output sequence
                        out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                        # store
                        X1.append(photo)
                        X2.append(in_seq)
                        y.append(out_seq)
                # yield the batch data
                if n == num_photos_per_batch:
                    yield ([np.array(X1), np.array(X2)], np.array(y))
                    X1, X2, y = list(), list(), list()
                    n = 0
