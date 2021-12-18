import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import json
#from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import timeit
#import gensim
import multiprocessing as mp
from tqdm import tqdm_notebook as tqdm
#from gensim.models import KeyedVectors
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing

def method1(cust_id):

    store_df=pd.read_csv('recommendation_dataset.csv')
    features = ['Customer_ID', 'product_name', 'Product_ID', 'Review Score']
    store_df = store_df[features]
    store_df = store_df.rename(columns={'Review Score': 'Review_Score'})
    store_df = store_df.drop_duplicates()
    store_df = store_df[:50000]
    store_df.Customer_ID.unique
    product_features_df = store_df.reset_index().pivot_table(
    index='Customer_ID',
    columns='Product_ID',
    values='Review_Score'
    ).fillna(0)
    #convert dataframe of movie features to scipy sparse matrix
    product_features_matrix = csr_matrix(product_features_df.values)
    X = product_features_df.T
    SVD = TruncatedSVD(n_components=10)
    decomp_matrix = SVD.fit_transform(X)
    print('Shape of decomp_matrix = ',decomp_matrix.shape)

    corr_matrix = np.corrcoef(decomp_matrix)
    print('Shape of corr_matrix = ',corr_matrix.shape)
    fav_product = store_df.groupby(['Customer_ID']).max()['Product_ID'].to_frame()
    fav_product = fav_product.reset_index()

    def get_product_id(customer_id):
        prod_id = fav_product[fav_product.Customer_ID == customer_id]['Product_ID']
        return prod_id
        
    prd_id = get_product_id(cust_id)
    Product_id = prd_id.iloc[0]

    prod_name = list(X.index)
    prod_id_index = prod_name.index(Product_id)
    corr_product_id = corr_matrix[prod_id_index]
    recommend = list(X.index[corr_product_id > 0.60])
    # Removes the item already bought by the customer
    recommend.remove(Product_id) 

    #Getting Product names from prediction 
    predictions = pd.DataFrame(recommend[:20])
    predictions.columns = ['Product_ID']
    predictions['Product_Name'] = predictions.Product_ID.apply(lambda x : store_df[store_df.Product_ID == x]['product_name'].unique()[0])
    predictions[:20]
    store_df.to_csv('store_df_ALS.csv', index=False)
    X.to_csv('X_ALS.csv')
    store_df = pd.read_csv('store_df_ALS.csv')
    X = pd.read_csv('X_ALS.csv', index_col=0)
    with open('correlation_matrix_ALS.txt') as f:
        corr_matrix = json.load(f)
    corr_matrix = np.array(corr_matrix)

    def product_recommendations_ALS(Customer_id):
        fav_product = store_df.groupby(['Customer_ID']).max()['Product_ID'].to_frame()
        fav_product = fav_product.reset_index()
    
        prd_id = fav_product[fav_product.Customer_ID == Customer_id]['Product_ID']
        product_id = prd_id.iloc[0]
        prod_name = list(X.index)
        product_id_index = prod_name.index(product_id)
    
        corr_product_ID = corr_matrix[product_id_index]
    
        recommend = list(X.index[corr_product_ID > 0.70])
        recommend.remove(product_id) 

        prod_predictions = pd.DataFrame(recommend[:20])
        prod_predictions.columns = ['Product_ID']

        prod_predictions['Product Name'] = prod_predictions.Product_ID.apply(lambda x : store_df[store_df.Product_ID == x]['product_name'].unique()[0])
        Recommendations = predictions[:10]
        return Recommendations

    prod_recommendations = product_recommendations_ALS(cust_id)
    prod_recommendations[:10] 
    
    
