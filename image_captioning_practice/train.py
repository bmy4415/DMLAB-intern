#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load data
# overal 8092 images, trainset in 'Flickr_8k.trainImages.txt'
# overal 8092 images, validset in 'Flickr_8k.devImages.txt'
# overal 8092 images, testset in 'Flickr_8k.testImages'


# In[2]:


import os
import pickle
import time
import numpy as np
from pprint import pprint
from IPython.display import SVG
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.callbacks import ModelCheckpoint


# In[3]:


def prepare_dataset(filename):
    image_ids = list()
    image_features = dict()
    captions = dict()
    
    # get image id
    with open(filename, 'r') as f:
        text = f.read()
        image_filenames = text.split('\n') # get image filenames in dataset: (train, valid, test)
        image_ids = [x.split('.')[0] for x in image_filenames if x] # remove '.jpg', igore empty filename

    # get image feature
    with open(os.getcwd() + '/image_features.pkl', 'rb') as f:
        all_features = pickle.load(f)
        image_features = { x: all_features[x] for x in image_ids }
        
    # get image caption
    with open(os.getcwd() + '/captions.txt', 'r') as f:
        text = f.read()
        lines = text.split('\n')
        for line in lines:
            space = line.index(' ')
            image_id = line[:space]
            caption = line[space+1:]
            
            # ignore empty caption
            if not caption:
                continue

            caption = 'startseq ' + caption + ' endseq' # add <start>, <end> token
            
            # append captions only in specified image_ids
            if image_id not in image_ids:
                continue

            if image_id in captions:
                captions[image_id].append(caption)
            else:
                captions[image_id] = [caption]
                    
    return image_ids, image_features, captions
        


# In[4]:



# # get image id
# with open(os.getcwd() + '/Flickr8k_text/Flickr_8k.trainImages.txt', 'r') as f:
#     text = f.read()
#     filenames = text.split('\n')
#     train_image_ids = [x.split('.')[0] for x in filenames if x] # remove '.jpg', ignore empty line

# # get image feature
# with open(os.getcwd() + '/image_features.pkl', 'rb') as f:
#     all_features = pickle.load(f)
#     train_features = { x: all_features[x] for x in train_image_ids }
    
# # get image caption
# with open(os.getcwd() + '/captions.txt', 'r') as f:
#     text = f.read()
#     for line in text.split('\n'):
#         space_idx = line.index(' ')
#         image_id = line[:space_idx]
#         image_caption = line[space_idx+1:]
#         image_caption = ['startseq'] + image_caption.split() + ['endseq'] # add <start>, <end> token, keras tokenizer will remove '<'(punctuation)
#         image_caption = ' '.join(image_caption) # back to string
#         if image_id in train_image_ids:
#             if image_id in train_captions:
#                 train_captions[image_id].append(image_caption)
#             else:
#                 train_captions[image_id] = [image_caption]


# In[5]:


# load each dataset

train_info_filename = os.getcwd() + '/Flickr8k_text/Flickr_8k.trainImages.txt'
valid_info_filename = os.getcwd() + '/Flickr8k_text/Flickr_8k.devImages.txt'
train_image_ids, train_feature_dict, train_caption_dict = prepare_dataset(train_info_filename)
valid_image_ids, valid_feature_dict, valid_caption_dict = prepare_dataset(valid_info_filename)

print('# of images in train set:', len(train_image_ids))
print('# of features in train set:', len(train_feature_dict)) # this should be same with above
print('# of images in train caption dict:', len(train_caption_dict))
print('# of images in valid set:', len(valid_image_ids))
print('# of features in valid set:', len(valid_feature_dict)) # this should be same with above
print('# of images in valid caption dict:', len(valid_caption_dict))

print('-------------------------------------------------------------------')
print('Exmaples in each variable')
print('train_image_ids[0]:', train_image_ids[0])
print('train_feature_dict[train_image_ids[0]].shape:', train_feature_dict[train_image_ids[0]].shape)
print('train_caption_dict[train_image_ids[0]]:')
pprint(train_caption_dict[train_image_ids[0]])


# In[6]:


def get_all_captions(captions_dict):
    result = []
    for image_id in captions_dict:
        caption_list = captions_dict[image_id]
        [result.append(x) for x in caption_list]
    
    return result


# In[7]:


def get_tokenizer(all_captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1 # cf) https://github.com/keras-team/keras/issues/7551
    return tokenizer, vocab_size


# In[8]:


tokenizer, vocab_size = get_tokenizer(get_all_captions(train_caption_dict))

print('len all train captions:', len(get_all_captions(train_caption_dict)))

word_set = set()
for caption in get_all_captions(train_caption_dict):
    caption = caption.split()[:]
    word_set.update(caption)
    
print('# of distinct word in all captions:', len(word_set))
print('vocab_size:', vocab_size)
print(set(tokenizer.word_docs.keys()) == word_set) # True


# In[9]:


# get maximun length of caption
def get_max_length(all_captions):
    length = 0
    for caption in all_captions:
        if len(caption.split()) > length:
            length = len(caption.split())
            
    return length


# In[10]:


'''
our model receive 2 inputs: image feature vector, sequence of words
our model outputs 1 word
if given image feature and caption: 'there is dog', we reuse subsequence of caption
image, 'startseq' => 'there'
image, 'startseq', 'there' => 'is'
image, 'startseq', 'there', 'is' => 'dog'
image, 'startseq', 'there', 'is', 'dog' => 'endseq'

tokenizer: tokenizer from train set
maxlen: maximum length of captions in train set
vocab_size: vocab_size extracted from train set
image_ids: image ids of data set which will be fed into model(train, valid, test)
feature_dict: feature dict of data set which will be fed into model(train, valid, test)
caption_dict: caption dict of data set which will be fed into model(train, valid, test)
'''

def get_subsequence_inputs(tokenizer, maxlen, vocab_size, image_ids, feature_dict, caption_dict):
    image, subsequence, output = list(), list(), list()
    for image_id in image_ids:
        captions = caption_dict[image_id]
        for caption in captions:
            # convert 'word sequnce' to 'int sequence'
            # each int implies word in vocab
            vector = tokenizer.texts_to_sequences([caption])[0] 
            for i in range(1, len(vector)):
                subseq, out = vector[:i], vector[i]
                # add padding to make input sequence has same fixed length
                subseq = pad_sequences([subseq], maxlen=maxlen)[0]
                # one hot encoding
                out = to_categorical([out], num_classes=vocab_size)[0]
                image.append(feature_dict[image_id][0]) # image feature vector
                subsequence.append(subseq) # vetorized subsequence of fixed size
                output.append(out) # one-hot encoded word vector
                
    return np.array(image), np.array(subsequence), np.array(output)


# In[11]:



# for image_id, caption_list in train_captions.items():
#     for caption in caption_list:
#         vec = tokenizer.texts_to_sequences([caption])[0] # convert to vector such that mapping each word to int based on vocab_size
#         for i in range(1, len(vec)):
#             seq_in, seq_out = vec[:i], vec[i] # split into pair
#             seq_in = pad_sequences([seq_in], maxlen=max_caption_length)[0] # add padding to make all caption vector has same length
#             seq_out = to_categorical([seq_out], num_classes=vocab_size)[0] # one hot encoding
#             x1.append(train_features[image_id][0]) # feature vector
#             x2.append(seq_in) # input word sequence to fixed length int vector: max length of input caption
#             y.append(seq_out) # output word to one-hot encoded vector of fixed length: vocab_size


# In[12]:


start = time.time()
maxlen = get_max_length(get_all_captions(train_caption_dict))
x1_train, x2_train, y_train = get_subsequence_inputs(tokenizer, maxlen, vocab_size,
                                                     train_image_ids, train_feature_dict, train_caption_dict)            
x1_valid, x2_valid, y_valid = get_subsequence_inputs(tokenizer, maxlen, vocab_size,
                                                     valid_image_ids, valid_feature_dict, valid_caption_dict)            

end = time.time()


# In[13]:


print('Elapsed time:', end-start, 'sec')
print('maxlen:', maxlen)
print('total # of inputs to model:', len(x1_train))
print('Example of subsequnce inputs below')
print('x1_train:', x1_train[2].shape) # image feature vector
pprint(x1_train[2])
print('x2_train:', x2_train[2].shape) # vectorized fixed length word sequence
pprint(x2_train[2])
print('y_train:',y_train[2].shape) # one hot encoded word vector
pprint(y_train[2])


# In[14]:


# define model
# vocab_size, maxlen already described above
def get_model(vocab_size, maxlen):
    # image
    inputs1 = Input(shape=(4096,)) # feature extracted from pretrained VGG16
    image = Dropout(0.5)(inputs1) # dropout layer
    image = Dense(256, activation='relu')(image) # fc layer outputs 256 features
    
    # caption
    inputs2 = Input(shape=(maxlen,)) # vectorized word sequence
    # word embedding base on 'vocab_size' to make simliar word has similar vector
    # shape: (maxlen) -> (maxlen, 256)
    # 'make_zero=True' option will ignore padded word
    caption = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    caption = Dropout(0.5)(caption) # dropout layer
    caption = LSTM(256)(caption) # LSTM layer
    
    # decoder
    decoder = add([image, caption])
    decoder = Dense(256, activation='relu')(decoder) # fc layer
    outputs = Dense(vocab_size, activation='softmax')(decoder) # probability of each vocab
    
    # make model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam') # set loss function and optimizer
    
    # visualize model summary
    print(model.summary())

    return model


# In[16]:


# possible to use this cell when you have enough memory
'''
# model checkpoint
start = time.time()

filepath = 'model-epoch{epoch:03d}-loss{loss:.3f}-validset_loss{val_loss:.3f}.h5' # format string
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# fit model
model = get_model(vocab_size, maxlen)
print('x1_train:', x1_train[2].shape) # image feature vector
print('x2_train:', x2_train[2].shape) # vectorized fixed length word sequence
print('y_train:',y_train[2].shape) # one hot encoded word vector

print('x1_valid:', x1_valid[2].shape) # image feature vector
print('x2_valid:', x2_valid[2].shape) # vectorized fixed length word sequence
print('y_valid:',y_valid[2].shape) # one hot encoded word vector


model.fit([x1_train, x2_train], y_train, epochs=1, verbose=2, callbacks=[checkpoint], validation_data=([x1_valid, x2_valid], y_valid))

end = time.time()
print('Elapsed:', end-start, 'sec')
'''


# In[17]:


# data generator to fit in workstation memory
def data_generator(caption_dict, feature_dict, tokenizer, maxlen):
    while True:
        for image_id, caption_list in caption_dict.items():
            image_in = feature_dict[image_id][0]
            image_in, seq_in, out_word = get_subsequence_inputs(tokenizer, maxlen, caption_list, image_in)
            yield [[image_in, seq_in], out_word]
            
def get_subsequence_inputs(tokenizer, maxlen, caption_list, image_in):
    x1, x2, y = list(), list(), list()
    vocab_size = len(tokenizer.word_index) + 1
    
    for caption in caption_list:
        seq = tokenizer.texts_to_sequences([caption])[0]
        
        # make subsequence inputs
        for i in range(1, len(seq)):
            seq_in, out_word = seq[:i], seq[i]
            seq_in = pad_sequences([seq_in], maxlen=maxlen)[0]
            out_word = to_categorical([out_word], num_classes=vocab_size)[0]
            x1.append(image_in)
            x2.append(seq_in)
            y.append(out_word)
            
    return np.array(x1), np.array(x2), np.array(y)


# In[18]:


generator = data_generator(train_caption_dict, train_feature_dict, tokenizer, maxlen)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)


# In[ ]:


# train using generator for saving memory
model = get_model(vocab_size, maxlen)
epochs = 20
steps = len(train_caption_dict)
start = time.time()
for i in range(epochs):
    generator = data_generator(train_caption_dict, train_feature_dict, tokenizer, maxlen)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('model_{}.h5'.format(i))
    
end = time.time()
print('Elaspsed:', end-start, 'sec')


# In[ ]:




