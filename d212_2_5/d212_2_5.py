#!/usr/bin/env python
# coding: utf-8

# <h1>WGU D212 TASK 2 REV 5 - MATTINSON</h1>

# In[1]:


import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[2]:


# import and configure packages
from imports import *
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[3]:


from helpers import *


# In[ ]:


#constants
random_state = 42
plotColor = ['b','g','r','m','c', 'y']
markers = ['+','o','*','^','v','>','<']


# In[4]:


# import raw customer data
df_raw = pd.read_csv('data/churn_clean.csv')
df_raw.shape


# In[ ]:


# remove unwanted data
df_cleaned = df_raw.drop(columns=[
    'CaseOrder','UID', 'County', 
    'Interaction', 'City', 
    'Job', 'Zip','Population',
    'Lat', 'Lng','Item1','Item2',
    'Item3','Item4','Item5','Item6',
    'Item7','Item8'
])
df_cleaned.shape


# In[ ]:


# filter for lost customers
df_churn = df_cleaned.loc[(df_cleaned.Churn=="Yes")]
df_churn.shape


# In[ ]:


# filter numerical float variables
df_numerical = df_churn.select_dtypes(include="float")
df_numerical.info()
df_numerical.shape


# In[ ]:


# rename columns to facilitate output
df_numerical.rename(columns = {
    'Income':'INC', 
    'Outage_sec_perweek':'OUT',
    'Tenure':'TEN',
    'MonthlyCharge':'MCH',
    'Bandwidth_GB_Year':'BAN'
}, inplace = True)
df_numerical.info()
df_numerical.shape


# In[ ]:


# describe numerical data 
df_numerical.describe().round(2)


# In[ ]:


# describe variables as continuous or categorical
describe_dataframe_type(df_numerical)


# In[ ]:


save_course_table_csv(data=df_cleaned, 
    title='CLEANED', title_only=True )


# In[ ]:


# use heatmap graph to identify highly correlated variables
def Generate_heatmap_graph(corr, chart_title, mask_uppertri=False ):
    """ Based on features , generate correlation matrix """
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = mask_uppertri
    fig,ax = plt.subplots(figsize=(6,6))
    sns.heatmap(corr
                , mask = mask
                , square = True
                , annot = True
                , annot_kws={'size': 10.5, 'weight' : 'bold'}
                , cmap=plt.get_cmap("YlOrBr")
                , linewidths=.1)
    plt.title(chart_title, fontsize=14)
    plt.show()
    
Generate_heatmap_graph(
    round(df_numerical.corr(),2), 
    chart_title = 'Correlation Matrix',
    mask_uppertri = True)    


# In[ ]:


# remove highly correlated variables
df_final = df_numerical.drop(columns=['BAN'])
df_final.info()
df_final.shape


# In[ ]:


# standardize remaining numerical data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_final.values)
df_standardized = pd.DataFrame(scaled_features,
        index=df_final.index, 
      columns=df_final.columns)
df_standardized.describe().round(2)


# In[ ]:


# create scatter plot of lost customers
fig, ax = plt.subplots(figsize =(7, 5))
plt.plot(df_numerical["TEN"], df_numerical["MCH"], marker="x", linestyle="")
plt.xlabel("Tenure (TEN)")
plt.ylabel("Monthly Charge (MCH)")
plt.title("Lost Customers (Churn='Yes')")
fig.savefig("figures/fig_1", dpi=150) 


# In[ ]:


# use boxplot to look for outliers
fig, ax = plt.subplots(figsize =(7, 5))
ax = df_standardized.boxplot(vert=False)


# In[ ]:


# create knee plot, adapted code (Arvai, 2022)
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42 }
sse = [] # list of SSE values for each k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)
fig, ax = plt.subplots(figsize =(7, 5))
knee = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Knee Plot")
plt.axvline(x=knee.elbow, color='green', ls=':', lw=2,)
fig.savefig("figures/fig_2", dpi=150)


# In[ ]:


# optimum point on knee plot
'Optimum: ({}, {:.3f})'.format(knee.elbow, sse[knee.elbow-1])


# In[ ]:


# code to perform a K-means clustering analysis
n_clusters=knee.elbow
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42 }
kmeans = KMeans(n_clusters=n_clusters,
        **kmeans_kwargs)
kmeans.fit(scaled_features)


# In[ ]:


# lowest SSE value
kmeans.inertia_


# In[ ]:


# final locations of the centroid
kmeans.cluster_centers_


# In[ ]:


# number of iterations required to converge
kmeans.n_iter_


# In[ ]:


# final cluster labels
kmeans.labels_


# In[ ]:


# final K-means analysis plot
fig, ax = plt.subplots(figsize =(7, 5))
title = 'K-Means Clustering (k=' + str(n_clusters) + ') for Lost Customers'
ax.scatter(x=df_standardized['TEN'],y=df_standardized['MCH'],
    c=kmeans.labels_,cmap='brg')
ax.scatter(x=kmeans.cluster_centers_[:,2],
    y=kmeans.cluster_centers_[:,3],
    color='black', marker='X',s=400 )
ax.set_xlabel('Tenure (standardized)')
ax.set_ylabel('Monthly Charge (standardized)')
plt.title(title)
fig.savefig("figures/fig_3", dpi=150)


# In[ ]:


# python moment
favorite_python_quote = "None_shall pass._B.K."
words = favorite_python_quote.split(' ')
for w in words:
    print(pyfiglet.figlet_format(w))


# In[ ]:




