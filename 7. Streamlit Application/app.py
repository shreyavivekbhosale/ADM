import seaborn as sns
import streamlit as st

import pandas as pd
import numpy as np


import base64
import datetime
from urllib.parse import urlencode


import json
import re
import sys
import itertools

from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import plotly.express as px

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

import warnings
warnings.filterwarnings("ignore")

from skimage import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import webbrowser
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import calendar

from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import timeit
import gensim
import multiprocessing as mp
from tqdm import tqdm_notebook as tqdm
from gensim.models import KeyedVectors
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing
from datetime import datetime
import squarify



html = """
  <style>
    .reportview-container {
      flex-direction: row;
    }
    header > .toolbar {
      flex-direction: row;
      left: 2rem;
      right: auto;
    }
    body {
    color: #fff;
    background-color: #ff0000;
    }
    .sidebar .sidebar-collapse-control,
    .sidebar.--collapsed .sidebar-collapse-control {
      left: auto;
      right: 0.5rem;
      background-color: #ff6347
    }
    .sidebar .sidebar-content {
      transition: margin-right .6s, box-shadow .6s;
      color: #fff;
      background-color: #ff6347;
    }
    .sidebar.--collapsed .sidebar-content {
      margin-left: auto;
      margin-right: -18rem;
      background-color: #ff6347;
    }
    @media (max-width: 991.98px) {
      .sidebar .sidebar-content {
        margin-left: auto;
        background-color: #ff6347
      }
    }
  </style>
"""

st.markdown(html, unsafe_allow_html=True)

add_selectbox = st.sidebar.radio(
    "Please Select anyone",
    ("Introduction","Exploratory Data Analysis","Market Analysis","Customer Segmentation using RFM","Churn Analysis","Recommendation System","Recommendation System for Loyal Customers","Get Popular Products")
)


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def get_popular_products(cust_id):
    
    # converting date columns to datetime
    Olist = pd.read_csv('rec_dataset.csv')
    print(Olist.columns)
    date_columns = ['Purchase Timestamp']
    for col in date_columns:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        Olist[col] = pd.to_datetime(Olist[col], format='%d/%m/%y %H:%M')
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(Olist[col])

    currentMonth = datetime.now().month
    Olist['rating_month'] = Olist['Purchase Timestamp'].apply(lambda x: x.month)

    temp = Olist[Olist.rating_month == currentMonth]

    popular_products = pd.DataFrame(temp.groupby(['rating_month', 'product_name'], as_index=False).agg({'Review Score': ['count', 'mean']}))
    popular_products.columns = ['Rating Month','Product Name', 'Popularity', 'Average Review Ratings']
    popular_products = popular_products.sort_values('Popularity', ascending=False)
    pop = ['Product Name', 'Average Review Ratings']

    popular_products[pop][:10]

def Recommendations_Cust2Vec(cust_id):
    prev_orders = pd.read_csv('rec_dataset.csv')
    prev_orders = prev_orders[prev_orders['Customer Segment'] == 'Loyal Customers']
    prev_orders_customers= prev_orders.customer_id.unique()
    prev_orders_customers = prev_orders_customers[:5000]
    prev_orders_details = prev_orders[prev_orders.customer_id.isin(prev_orders_customers)].copy()
    
    # Create basic user features: relative purchase frequences in each depertment/aisle

    dept_feat = pd.pivot_table(prev_orders_details, index=['customer_id'], values=['product_id'], columns=['department'], aggfunc='count', fill_value=0)
    dept_feat = dept_feat.div(dept_feat.sum(axis=1), axis=0)
    dept_feat.columns = dept_feat.columns.droplevel(0)
    dept_feat = dept_feat.reset_index()
    aisle_feat = pd.pivot_table(prev_orders_details, index=['customer_id'], values=['product_id'], columns=['aisle'], aggfunc='count', fill_value=0)
    aisle_feat = aisle_feat.div(aisle_feat.sum(axis=1), axis=0)
    aisle_feat.columns = aisle_feat.columns.droplevel(0)
    aisle_feat = aisle_feat.reset_index()
    feature_df = dept_feat.merge(aisle_feat, how='left', on='customer_id').set_index('customer_id')
    prev_orders["product_id"] = prev_orders["product_id"].astype(str)
    #sorting order and products in orders chronologically
    prev_orders.sort_values(by=['customer_id','order_id'], inplace=True)
    allorders_by_custid = prev_orders.groupby("customer_id").apply(lambda order: ' '.join(order['product_id'].tolist()))
    allorders_by_custid = pd.DataFrame(allorders_by_custid,columns=['all_orders'])
    allorders_by_custid.reset_index(inplace=True)
    allorders_by_custid.customer_id = allorders_by_custid.customer_id.astype(str)
    TRAIN_USER_MODEL = True   
    MODEL_DIR = 'models'
    dim_embeddings = 100    # dimensionality of user representation
    filename = '/home/shreya/FinalProject/7. Streamlit Application/models/customer2vec.{dim_embeddings}d.model'
    if TRAIN_USER_MODEL:
        class TaggedDocumentIterator(object):
            def __init__(self, df):
                self.df = df
            def __iter__(self):
                for row in self.df.itertuples():
                    yield TaggedDocument(words=dict(row._asdict())['all_orders'].split(),tags=[dict(row._asdict())['customer_id']])
       
        it = TaggedDocumentIterator(allorders_by_custid)
        doc_model = gensim.models.Doc2Vec(vector_size=dim_embeddings, 
                                      window=5, 
                                      min_count=10, 
                                      workers=mp.cpu_count(),
                                      alpha=0.055, 
                                      min_alpha=0.055,
                                      epochs=15)   # use fixed learning rate
        train_corpus = list(it)
        doc_model.build_vocab(train_corpus)
        for epoch in tqdm(range(10)):
            doc_model.alpha -= 0.005                    # decrease the learning rate
            doc_model.min_alpha = doc_model.alpha       # fix the learning rate, no decay
            doc_model.train(train_corpus, total_examples=doc_model.corpus_count, epochs=doc_model.epochs)
        
        doc_model.save(filename)  
    vocab_doc = list(doc_model.docvecs.doctags.keys())
    doc_vector_dict = {arg:doc_model.docvecs[arg] for arg in vocab_doc}
    X_doc = pd.DataFrame(doc_vector_dict).T.values
    user_ids_sample_str = set([str(id) for id in prev_orders_customers])
    idx = []
    for i, user_id in enumerate(doc_vector_dict):
        if user_id in user_ids_sample_str:
            idx.append(i)
    X_doc_subset = X_doc[idx] # only sampled user IDs
    doc_vec_subset = pd.DataFrame(doc_vector_dict).T.iloc[idx]
    distance_matrix_doc = pairwise_distances(X_doc_subset, X_doc_subset, metric='cosine', n_jobs=-1)
    tsne_doc = TSNE(metric="precomputed", n_components=2, verbose=1, perplexity=30, n_iter=500)
    tsne_results_doc = tsne_doc.fit_transform(distance_matrix_doc)
    tsne_doc = pd.DataFrame()
    
    tsne_doc['tsne-2d-one'] = tsne_results_doc[:,0]
    tsne_doc['tsne-2d-two'] = tsne_results_doc[:,1]
    
    def cosine_cluster(X, k):
    # normalization is equivalent to cosine distance
        return KMeans(n_clusters=k).fit(preprocessing.normalize(X_doc_subset)).labels_.astype(float)

    silhouette_list = []
    for k in tqdm(range(2, 15, 1)):
        latent_clusters = cosine_cluster(X_doc_subset, k)
        silhouette_avg = silhouette_score(X_doc_subset, latent_clusters, metric="cosine")
        silhouette_list.append(silhouette_avg)
        
        
    N_CLUSTER = 7

    latent_clusters = cosine_cluster(X_doc_subset, N_CLUSTER)
    doc_vec_end = doc_vec_subset.copy()
    doc_vec_end['label'] = latent_clusters
    tsne_doc['cluster'] = latent_clusters 
    
    feature_df['latent_cluster'] = latent_clusters

    dept_names = np.setdiff1d(prev_orders_details['department'].unique(), ['other', 'missing'])
    interpret_department = feature_df.groupby('latent_cluster')[dept_names].mean()

    interpret_department.T.div(interpret_department.sum(axis=1)).round(3)
    interpetation_aisle = feature_df.groupby('latent_cluster')[feature_df.columns.values[16:-1]].mean()
    interpetation_aisle.T.div(interpetation_aisle.sum(axis=1)).round(3).head(20)
    prev_orders_details_clust = prev_orders_details.copy()
    prev_orders_details_clust = prev_orders_details_clust.merge(feature_df['latent_cluster'], on='customer_id', how='left')

    for cluster_id in [1.0, 2.0, 4.0, 3.0]:
        prev_orders_details_clust[prev_orders_details_clust['latent_cluster']==cluster_id][['customer_id', 'product_name']].groupby("customer_id").apply(lambda order: ' >'.join(order['product_name'])).reset_index().head(10)
    
    prev_orders_details_all = prev_orders_details_clust
    
    orders_details = pd.read_csv('Recomm_Customer2vec.csv')
    recommendations_c2v = orders_details[['customer_id', 'product_name', 'Product Category', 'Review Score']]
    recommendations_c2v = recommendations_c2v[recommendations_c2v.customer_id == cust_id]
    recommendations_c2v


def Recommendation_ALS(cust_id):

    store_df=pd.read_csv('rec_dataset.csv')
    features = ['customer_id', 'product_name', 'product_id', 'Review Score']
    store_df = store_df[features]
    store_df = store_df.rename(columns={'Review Score': 'Review_Score'})
    store_df = store_df.drop_duplicates()
    store_df = store_df[:5000]
    product_features_df = store_df.reset_index().pivot_table(
    index='customer_id',
    columns='product_id',
    values='Review_Score'
    ).fillna(0)
    
    #convert dataframe of movie features to scipy sparse matrix
    product_features_matrix = csr_matrix(product_features_df.values)
    X = product_features_df.T
    SVD = TruncatedSVD(n_components=10)
    decomp_matrix = SVD.fit_transform(X)

    corr_matrix = np.corrcoef(decomp_matrix)
    
    fav_product = store_df.groupby(['customer_id']).max()['product_id'].to_frame()
    fav_product = fav_product.reset_index()

    def get_product_id(customer_id):
        prod_id = fav_product[fav_product.customer_id == customer_id]['product_id']
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
    predictions['Product_Name'] = predictions.Product_ID.apply(lambda x : store_df[store_df.product_id == x]['product_name'].unique()[0])
    #predictions[:20]
    store_df.to_csv('store_df_ALS.csv', index=False)
    X.to_csv('X_ALS.csv')
    with open('correlation_matrix_ALS1.txt', 'w') as filehandle:
        json.dump(corr_matrix.tolist(), filehandle)
    
    store_df = pd.read_csv('store_df_ALS.csv')
    
    X = pd.read_csv('X_ALS.csv', index_col=0)
    
    with open('correlation_matrix_ALS1.txt') as f:
        corr_matrix = json.load(f)
    corr_matrix = np.array(corr_matrix)

    def product_recommendations_ALS(Customer_id):
        fav_product = store_df.groupby(['customer_id']).max()['product_id'].to_frame()
        fav_product = fav_product.reset_index()
    
        prd_id = fav_product[fav_product.customer_id == Customer_id]['product_id']
        product_id = prd_id.iloc[0]
        prod_name = list(X.index)
        product_id_index = prod_name.index(product_id)
    
        corr_product_ID = corr_matrix[product_id_index]
    
        recommend = list(X.index[corr_product_ID > 0.70])
        recommend.remove(product_id) 

        prod_predictions = pd.DataFrame(recommend[:20])
        prod_predictions.columns = ['Product_ID']

        prod_predictions['Product Name'] = prod_predictions.Product_ID.apply(lambda x : store_df[store_df.product_id == x]['product_name'].unique()[0])
        Recommendations = predictions[:10]
        return Recommendations

    prod_recommendations = product_recommendations_ALS(cust_id)
    prod_recommendations[:10]
    
    
if add_selectbox == 'Introduction':
    st.title("Global Distribution Superstore")
    st.image("/home/shreya/FinalProject/7. Streamlit Application/index.png")

    st.markdown(
        """
        
        Welcome to the web application with Sales Marketing Insights, Personalized Recommendation, RFM, and Churn Information required for the Marketing and Analysis team at Global Distribution Superstore, which will help them to view information of customers all in one place. Resulting in taking quick data-driven decisions for greater profits!
        
        """
    )  
        
elif add_selectbox == 'Market Analysis':
    link="https://app.powerbi.com/groups/me/dashboards/dc7e46b9-a78e-4d8e-b594-ed107c7f01a7"
    link2="https://public.tableau.com/app/profile/shreya.bhosale2894/viz/project_16397052840550/Sheet1?publish=yes"
    st.title("Market Analysis")
    st.write("-------------------------------------------------------------------------------------------------")	
    if st.button("GET POWERBI DASHBOARDS!"):
    	webbrowser.open_new_tab(link)
    elif st.button("GET TABLEAU DASHBOARDS!"):
        webbrowser.open_new_tab(link2)	
    
    #set_png_as_page_bg('/home/shreya/FinalProject/7. Streamlit Application/reports.png')
    st.image("/home/shreya/FinalProject/7. Streamlit Application/reports.png")
    
    
    
elif add_selectbox == 'Customer Segmentation using RFM':
    st.title("Customer Segmentation using RFM :")
    st.write("-------------------------------------------------------------------------------------------------")

    def get_rfm():
        return pd.read_csv('RFM_CustomerSegment.csv')

    rfm = get_rfm()
    
    def get_customers():
        return pd.read_csv('rfm_data_new.csv')


    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#FF00FF', '#C0C0C0']
    values = rfm['Monetary Value']
    labels = rfm['Customer_Segment']
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    st.write(fig)
    
    fig1=plt.figure(figsize=(8,8))
    temp = rfm['Customer_Segment'].value_counts().sort_values(ascending=False)
    sns.barplot(temp.index, temp.values)
    plt.ylabel("Number of Customers")
    plt.title("Visualizing customer segmentation clusters");
    plt.xticks(rotation=90);
    st.write(fig1)
    
    #Create RFM Segment treemap
    #fig2 = plt.gcf()
    #ax = fig2.add_subplot()
    #fig2.set_size_inches(12, 8)
    fig2=plt.figure(figsize = (12,8))
    squarify.plot(sizes=rfm['Customer_Segment'].value_counts(),
    label=['Core Best ', 'Loyal', 'Regular', 'Almost Lost','Lost','Slipping', 'Premium Payers'], alpha=0.8 )
    plt.title("RFM Customer Segments")
    plt.axis('off')
    st.write(fig2)



elif add_selectbox == 'Exploratory Data Analysis':
    st.title("Exploratory Data Analysis :")
    st.write("-------------------------------------------------------------------------------------------------")
    
    def get_orders_final():
        return pd.read_csv('cleaned_orders_dataset.csv')

    def get_rfm():
        return pd.read_csv('RFM_CustomerSegment.csv')

    df_year = get_rfm()
    df_year = df_year[:101]
    df_year_control = df_year.copy()
    
    df_year_control['Recency'] = df_year_control['Recency'] / df_year_control['Recency'].max()
    df_year_control['Frequency'] = df_year_control['Frequency'] / df_year_control['Frequency'].max()
    df_year_control['Monetary Value'] = df_year_control['Monetary Value'] / df_year_control['Monetary Value'].max()
    df_year_control['Rank'] = df_year_control['Rank'] / df_year_control['Rank'].max()
    df_year_control['Quartile_R'] = df_year_control['Quartile_R'] / df_year_control[
        'Quartile_R'].max()
    df_year_control['Quartile_F'] = df_year_control['Quartile_F'] / df_year_control['Quartile_F'].max()
    df_year_control['Quartile_M'] = df_year_control['Quartile_M'] / df_year_control['Quartile_M'].max()
    df_year_control['customer_id'] = df_year_control['customer_id'].astype(str)

    df_year_control.drop(["RFM_Class", "RFM_Score"], axis=1, inplace=True)
    df_year_control = df_year_control.melt("customer_id")

    fig = px.line_polar(df_year_control, r="value", theta="variable", line_close=True,
                        animation_frame="customer_id", template="plotly_dark", range_r=(0, 1))
    fig.update_traces(fill='toself')
    fig.update_layout(font_size=15)
    st.write(fig)
    
#1. Top Product Categories  
    st.write("                                                   ")    
    orders_final_df=  get_orders_final()
    orders_final_df['product_category_name_english']=[str(x) for x in orders_final_df['product_category_name_english']]
    categories=",".join(orders_final_df["product_category_name_english"])
    categories= [x.strip() for x in categories.split(',')]

    categories=Counter(categories)
    top_15_categories=[x for x in categories.most_common(15)]
    a,b=map(list,zip(*top_15_categories))
    fig1=plt.figure(figsize=(8,8))
    sns.barplot(b,a,palette="Spectral")
    plt.title("Top Product Categories in Global Superstore Sales")
    plt.ylabel("Types of Product Categories")
    plt.xlabel("Count")
    st.write(fig1)        

#2. Preferred Payment Types
    orders_final_df['payment_type']=[str(x) for x in orders_final_df['payment_type']]
    ptype=",".join(orders_final_df["payment_type"])
    ptype= [x.strip() for x in ptype.split(',')]
    ptype=Counter(ptype)
    top_ptype=[x for x in ptype.most_common(5)]
    a,b=map(list,zip(*top_ptype))
    fig2=plt.figure(figsize=(8,8))
    colors = sns.color_palette('pastel')[0:5]
    plt.pie(b,labels = a, colors = colors, autopct='%.0f%%')
    st.write(fig2)
   

#3. Top cities for orders
    fig3=plt.figure(figsize = (8,8))
    orders_final_df.customer_city.value_counts().nlargest(10).plot(kind='barh', color = 'yellow')
    plt.title("Number of orders by cities")
    plt.xlabel("Order counts")
    plt.ylabel("Cities")
    st.write(fig3)


#4. Approximate spending per state and city for order
    temp = orders_final_df.groupby(['customer_state','customer_city'])['total_payment'].agg('mean').sort_values(ascending = False).head(20)
    fig4=plt.figure(figsize = (25,10))
    ax = temp.plot(figsize=(25,8), grid=True, kind = 'area', color = 'pink')
    ax.set_xticks(range(len(temp)))
    ax.set_xticklabels(temp.index, rotation=90, fontsize=15)
    ax.set_xlabel('Order State and City', fontsize=20)
    ax.set_ylabel('Order Cost', fontsize=20)
    ax.set_title('Approximate spending per state and city', fontsize=20, pad = 10)    
    st.write(fig4)

#5. Top sellers with most products
    fig5=plt.figure(figsize=(16, 8))
    temp1 = orders_final_df.groupby(['seller_id'])['product_id'].count().sort_values(ascending=False).head(20)
    axis = sns.lineplot(temp1.index, temp1,palette='rocket')
    axis.set_title('Top 20 Sellers with most products sold')
    axis.set_ylabel('Product Count')
    axis.set_xlabel('Sellers')
    plt.xticks(rotation = 90)
    st.write(fig5)
    
    
#6. Top customers with most orders
    fig6=plt.figure(figsize=(16, 8))
    temp1 = orders_final_df.groupby("customer_unique_id")['order_id'].count().sort_values(ascending=False)[:20]
    axis = sns.barplot(temp1.index, temp1,palette='winter')
    axis.set_title('Top 20 Customers with most orders placed')
    axis.set_ylabel('Order Count')
    axis.set_xlabel('Customers')
    plt.xticks(rotation = 90)
    st.write(fig6)
    
#7. Number of sellers from different cities
    temp = orders_final_df.groupby(['seller_city'])['seller_id'].count().sort_values(ascending=False).iloc[0:10]
    fig7=plt.figure(figsize=(20,5))
    sns.barplot(temp.index, temp.values, palette="gist_stern")
    plt.ylabel("Counts")
    plt.title("Number of sellers from different cities");
    plt.xticks(rotation=90);   
    st.write(fig7) 
    
#8 State wise count of customers and sellers    
    temp = orders_final_df.groupby('customer_state')['seller_id', 'customer_unique_id'].count()
    fig7=plt.figure(figsize=(25, 8))
    ax = temp.plot(figsize=(25,8), color = ['purple','green'], kind = 'bar')
    ax.set_xticks(range(len(temp)))
    ax.set_xticklabels(temp.index, rotation=90, fontsize=15)
    ax.set_xlabel('State', fontsize=20)
    ax.set_ylabel('Number of Customers and Sellers', fontsize=20)
    ax.set_title('State wise count of customers and sellers', fontsize=20, pad = 10)
    st.write(fig7)  
    
    
#9  Customer Marital Status analysis

    #population_m = orders_final_df['customer_marital_status']
    fig8=plt.figure(figsize=(4, 4))
    orders_final_df.customer_marital_status.value_counts().plot(kind = 'pie', autopct = '%.2f', radius = 1,
                                        colors = ['#E74C3C', '#F1948A'], explode = [0,0.1], 
                                        labels = ['Yes', 'No'], shadow = True,
                                        textprops = dict(size=12), wedgeprops = dict(linewidth=2))
    plt.axis('off')
    plt.title('Customer Marital Status Analysis', pad = 10)
    st.write(fig8)
    
#10 Analysis of reviews per order

    temp1 = orders_final_df[orders_final_df['review_score'].isin(['1.0','2.0',])]
    temp = orders_final_df.groupby(['review_score'])['order_id'].count().sort_values(ascending=False).iloc[0:10]
    fig9=plt.figure(figsize=(20,15))
    sns.barplot(temp.index, temp.values, palette="viridis")
    plt.ylabel("Order Counts")
    plt.title("Analysis of reviews per order");
    plt.xticks(rotation=90); 
    st.write(fig9)   
    
    
    
elif add_selectbox == 'Churn Analysis':
    st.title("Churn Analysis : ")
    st.write("-------------------------------------------------------------------------------------------------")

    st.markdown(
        """
        # Purchase Frequency : 1.1805510432932644
        
        
        
        # Repeat Rate : 0.12437774447437093
               
        
        # Churn Rate : 0.8756222555256291
        """
                )

    def get_clv():
        return pd.read_csv('customer_clv_cltv.csv')

    def get_rfm_segment():
        return pd.read_csv('RFM_CustomerSegment.csv')

    def get_monthly_churn_rate():
        return pd.read_csv('monthly_churn_rate.csv')
        
    clv = get_clv()
    rfm2 = get_rfm_segment()
    mcr = get_monthly_churn_rate()

    id = st.selectbox("Select CustomerID : ",clv['customer_unique_id'])


    if id:
        count = -1
        for i in clv['customer_unique_id']:
            count = count + 1
            if id == i:
                st.title(rfm2['Customer_Segment'][count]) 
                st.title("Customer Lifetime Value: ")
                st.subheader(clv['Cust_Lifetime_Value_CLTV'][count])
                 
                
                
    id = st.selectbox("Order Date Month : ",mcr['order_date_month'])  
    
    if id:
        count = -1
        for i in mcr['order_date_month']:
            count = count + 1
            if id == i:
                st.title("Monthly Churn Rate: ")
                st.subheader(mcr['ChurnRate'][count])
                
                

elif add_selectbox == 'Recommendation System':
    st.title("Recommendation System")
    st.write("-------------------------------------------------------------------------------------------------") 
    def get_recommendation():
        return pd.read_csv('rec_dataset.csv')   
    
    clv=get_recommendation()
      
    id = st.selectbox("Select CustomerID : ",clv['customer_id'])
       
    if id:
     count = -1
     for i in clv['customer_id']:
        count = count + 1
        if id == i:
            Recommendation_ALS(i) 
            break              
            
            
elif add_selectbox == 'Recommendation System for Loyal Customers':
    st.title("Recommendation System for Loyal Customers")
    st.write("-------------------------------------------------------------------------------------------------") 
    
    def Recomm_Customer2vec():
        return pd.read_csv('Recomm_Customer2vec.csv')    
    
    clv1=Recomm_Customer2vec()            
                        
    id = st.selectbox("Select CustomerID : ",clv1['customer_id'])
    if id:
     count = -1
     for i in clv1['customer_id']:
        count = count + 1
        if id == i:
            Recommendations_Cust2Vec(i) 
            break 
            
elif add_selectbox == 'Get Popular Products':
    st.title("Get Popular Products")
    st.write("-------------------------------------------------------------------------------------------------") 
    
    def rec_dataset():
        return pd.read_csv('rec_dataset.csv')    
    
    clv1=rec_dataset()            
                        
    id = st.selectbox("Select CustomerID : ",clv1['customer_id'])
    if id:
     count = -1
     for i in clv1['customer_id']:
        count = count + 1
        if id == i:
            get_popular_products(i)
            break             
                
