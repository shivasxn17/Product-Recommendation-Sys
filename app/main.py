import pandas as pd
import pickle

data = # load dataset

# generating vector for words in a title
with open('word2vec_model', 'rb') as handle:
   model = pickle.load(handle)

vocab = model.keys()
idf_title_vectorizer = CountVectorizer()

# generating  the model for column title which can be anyother column as well
idf_title_features = idf_title_vectorizer.fit_transform(data['title'])

def build_avg_vec(sentence, num_features, doc_id, m_name):
  # sentence: its title of the apparel
  # num_features: the length of word2vec vector, its values = 300
  # m_name: model information it will take two values
      # if  m_name == 'avg', we will append the model[i], w2v representation of word i
      # if m_name == 'weighted', we will multiply each w2v[word] with the idf(word)
  featureVec = np.zeros((num_features,), dtype="float32")
  # we will intialize a vector of size 300 with all zeros
  # we add each word2vec(wordi) to this fetureVec
  nwords = 0
  
  for word in sentence.split():
    nwords += 1
    if word in vocab:
      if m_name == 'weighted' and word in  idf_title_vectorizer.vocabulary_:
        featureVec = np.add(featureVec, idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[word]] * model[word])
      elif m_name == 'avg':
        featureVec = np.add(featureVec, model[word])
  if(nwords>0):
    featureVec = np.divide(featureVec, nwords)
  # returns the avg vector of given sentance, its of shape (1, 300)
  return featureVec


doc_id = 0
w2v_title_weight = []
# for every title we build a weighted vector representation
for i in data['title']:
  w2v_title_weight.append(build_avg_vec(i, 300, doc_id,'weighted'))
  doc_id += 1

# w2v_title = np.array(# number of doc in courpus * 300), each row corresponds to a doc 
w2v_title_weight = np.array(w2v_title_weight)

data['brand'].fillna(value="Not given", inplace=True)

brands = [x.replace(" ", "-") for x in data['brand'].values]
types = [x.replace(" ", "-") for x in data['product_type_name'].values]
colors = [x.replace(" ", "-") for x in data['color'].values]

brand_vectorizer = CountVectorizer()
brand_features = brand_vectorizer.fit_transform(brands)

type_vectorizer = CountVectorizer()
type_features = type_vectorizer.fit_transform(types)

color_vectorizer = CountVectorizer()
color_features = color_vectorizer.fit_transform(colors)

extra_features = hstack((brand_features, type_features, color_features)).tocsr()


#load the features and corresponding ASINS info.
bottleneck_features_train = np.load('16k_data_cnn_features.npy')
asins = np.load('16k_data_cnn_feature_asins.npy')
asins = list(asins)

# load the original 16K dataset
data = pd.read_pickle('pickels/16k_apperal_data_preprocessed')
df_asins = list(data['asin'])


def idf_w2v_brand_vgg16(doc_id, w1, w2, w3, num_results):
  # w1: weight for  w2v features
  # w2: weight for brand and color features
  # w2: weight for img features

  idf_w2v_dist  = pairwise_distances(w2v_title_weight, w2v_title_weight[doc_id].reshape(1,-1))
  ex_feat_dist = pairwise_distances(extra_features, extra_features[doc_id])
  img_feat_dist = pairwise_distances(bottleneck_features_train, bottleneck_features_train[doc_id].reshape(1,-1))
  pairwise_dist   = (w1 * idf_w2v_dist +  w2 * ex_feat_dist + w3 * img_feat_dist)/float(w1 + w2 + w3)

  indices = np.argsort(pairwise_dist.flatten())[0:num_results]
  pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
  df_indices = list(data.index[indices])

  for i in range(0, len(indices)):
      heat_map_w2v_brand(data['title'].loc[df_indices[0]],data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], indices[0], indices[i],df_indices[0], df_indices[i], 'weighted')
      print('ASIN :',data['asin'].loc[df_indices[i]])
      print('Brand :',data['brand'].loc[df_indices[i]])
      print('euclidean distance from input :', pdists[i])
      print('='*125)

# title vector weight = 10
# brand and color weight = 25
# img feature weight = 1

# i am giving more preference to print of the product that image similarity.
idf_w2v_brand_vgg16(12566, 10, 25, 1, 20)