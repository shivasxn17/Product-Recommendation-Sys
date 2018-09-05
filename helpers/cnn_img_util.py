import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import requests
from PIL import Image
import pandas as pd
import pickle

# https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'images/' # images directory
nb_train_samples = 16042
epochs = 50
batch_size = 1

def save_bottlebeck_features():
  asins = []
  datagen = ImageDataGenerator(rescale=1. / 255)
  
  # build the VGG16 network
  model = applications.VGG16(include_top=False, weights='imagenet')
  generator = datagen.flow_from_directory(
      train_data_dir,
      target_size=(img_width, img_height),
      batch_size=batch_size,
      class_mode=None,
      shuffle=False)

  for i in generator.filenames:
    asins.append(i[2:-5])

  bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)
  bottleneck_features_train = bottleneck_features_train.reshape((16042,25088))
  
  np.save(open('models/data_cnn_features.npy', 'wb'), bottleneck_features_train)
  np.save(open('models/data_cnn_feature_asins.npy', 'wb'), np.array(asins))
    
save_bottlebeck_features()

# loading the features 
bottleneck_features_train = np.load('models/data_cnn_features.npy')
asins = np.load('models/data_cnn_feature_asins.npy')

# get the most similar apparels using euclidean distance measure
data = pd. #loading data to be used
df_asins = list(data['asin'])
asins = list(asins)

from IPython.display import display, Image, SVG, Math, YouTubeVideo

def get_similar_products_cnn(doc_id, num_results):
  doc_id = asins.index(df_asins[doc_id])
  pairwise_dist = pairwise_distances(bottleneck_features_train, bottleneck_features_train[doc_id].reshape(1,-1))

  indices = np.argsort(pairwise_dist.flatten())[0:num_results]
  pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

  for i in range(len(indices)):
    rows = data[['medium_image_url','title']].loc[data['asin']==asins[indices[i]]]
    for indx, row in rows.iterrows():
      display(Image(url=row['medium_image_url'], embed=True))
      print('Product Title: ', row['title'])
      print('Euclidean Distance from input image:', pdists[i])
      print('Amazon Url: www.amzon.com/dp/'+ asins[indices[i]])

get_similar_products_cnn(12566, 10)