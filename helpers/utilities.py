import requests
import PIL.Image
from io import BytesIO
from IPython.display import display, Image, SVG, Math, YouTubeVideo 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns # to generate heatmap
from collections import Counter
from matplotlib import gridspec
import plotly
import plotly.figure_factory as ff

def get_word_vec(sentence, doc_id, m_name):
  # sentence : title of the apparel
  # doc_id: document id in our corpus
  # m_name: model information it will take two values
    # if  m_name == 'avg', we will append the model[i], w2v representation of word i
    # if m_name == 'weighted', we will multiply each w2v[word] with the idf(word)
  vec = []
  for i in sentence.split():
      if i in vocab:
          if m_name == 'weighted' and i in  idf_title_vectorizer.vocabulary_:
              vec.append(idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[i]] * model[i])
          elif m_name == 'avg':
              vec.append(model[i])
      else:
          # if the word in our courpus is not there in the google word2vec corpus, we are just ignoring it
          vec.append(np.zeros(shape=(300,)))
  # we will return a numpy array of shape (#number of words in title * 300 ) 300 = len(w2v_model[word])
  # each row represents the word2vec representation of each word (weighted/avg) in given sentance 
  return  np.array(vec)

def get_distance(vec1, vec2):
  # vec1 = np.array(#number_of_words_title1 * 300), 
  # each row is a vector of length 300 corresponds to each word in give title
  # vec2 = np.array(#number_of_words_title2 * 300), 
  # each row is a vector of length 300 corresponds to each word in give title
  
  final_dist = []
  # for each vector in vec1 we caluclate the distance(euclidean) to all vectors in vec2
  for i in vec1:
      dist = []
      for j in vec2:
          # np.linalg.norm(i-j) will result the euclidean distance between vectors i, j
          dist.append(np.linalg.norm(i-j))
      final_dist.append(np.array(dist))
  # final_dist = np.array(#number of words in title1 * #number of words in title2)
  # final_dist[i,j] = euclidean distance between vectors i, j
  return np.array(final_dist)

#Display an image
def display_img(url,ax,fig):
  # we get the url of the apparel and download it
  response = requests.get(url)
  img = PIL.Image.open(BytesIO(response.content))
  # we will display it in notebook 
  plt.imshow(img)

def heat_map_w2v_brand(sentance1, sentance2, url, doc_id1, doc_id2, df_id1, df_id2, model):
  
  # sentance1 : title1, input apparel
  # sentance2 : title2, recommended apparel
  # url: apparel image url
  # doc_id1: document id of input apparel
  # doc_id2: document id of recommended apparel
  # df_id1: index of document1 in the data frame
  # df_id2: index of document2 in the data frame
  # model: it can have two values, 1. avg 2. weighted
  
  # each row is a vector(weighted/avg) of length 300 corresponds to each word in give title
  # s1_vec = np.array(#number_of_words_title1 * 300)
  s1_vec = get_word_vec(sentance1, doc_id1, model)
  # each row is a vector(weighted/avg) of length 300 corresponds to each word in give title
  # s2_vec = np.array(#number_of_words_title2 * 300)
  s2_vec = get_word_vec(sentance2, doc_id2, model)
  
  # s1_s2_dist = np.array(#number of words in title1 * #number of words in title2)
  # s1_s2_dist[i,j] = euclidean distance between words i, j
  s1_s2_dist = get_distance(s1_vec, s2_vec)
 
  data_matrix = [['Asin','Brand', 'Color', 'Product type'],
             [data['asin'].loc[df_id1],brands[doc_id1], colors[doc_id1], types[doc_id1]], # input apparel's features
             [data['asin'].loc[df_id2],brands[doc_id2], colors[doc_id2], types[doc_id2]]] # recommonded apparel's features
  
  colorscale = [[0, '#1d004d'],[.5, '#f2e5ff'],[1, '#f2e5d1']] # to color the headings of each column 
  
  # we create a table with the data_matrix
  table = ff.create_table(data_matrix, index=True, colorscale=colorscale)
  # plot it with plotly
  plotly.offline.iplot(table, filename='simple_table')
  
  # devide whole figure space into 25 * 1:10 grids
  gs = gridspec.GridSpec(25, 15)
  fig = plt.figure(figsize=(25,5))
  
  # in first 25*10 grids we plot heatmap
  ax1 = plt.subplot(gs[:, :-5])
  # ploting the heap map based on the pairwise distances
  ax1 = sns.heatmap(np.round(s1_s2_dist,6), annot=True)
  # set the x axis labels as recommended apparels title
  ax1.set_xticklabels(sentance2.split())
  # set the y axis labels as input apparels title
  ax1.set_yticklabels(sentance1.split())
  # set title as recommended apparels title
  ax1.set_title(sentance2)

  # in last 25 * 10:15 grids we display image
  ax2 = plt.subplot(gs[:, 10:16])
  # we dont display grid lins and axis labels to images
  ax2.grid(False)
  ax2.set_xticks([])
  ax2.set_yticks([])
  
  # pass the url it display it
  display_img(url, ax2, fig)
  plt.show()

