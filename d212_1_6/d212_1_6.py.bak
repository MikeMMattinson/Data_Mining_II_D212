# code to perform a K-means clustering analysis
n_clusters=4
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42 }
kmeans = KMeans(n_clusters=n_clusters,
        **kmeans_kwargs)
kmeans.fit(scaled_features)



# create scatter plot of lost customer data
fig, ax = plt.subplots(figsize =(7, 5))
plt.plot(df["TEN"], df["MCH"], marker="x", linestyle="")
plt.xlabel("Tenure")
plt.ylabel("Monthly Charge")
plt.title("Lost Customers (Churn='Yes')")
fig.savefig("figures/fig_1", dpi=150) 


# import raw customer data
df_raw = pd.read_csv('data/churn_clean.csv')

Out[]: (10000,50)


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

Out[]: (10000,32)



# filter for lost customers
df_churn = df_cleaned.loc[(df_cleaned.Churn=="Yes")]
df_churn.shape

Out[]: (2650,32)


# filter numerical float variables
df_numerical = df_churn.select_dtypes(include="float")
df_numerical.info()
df_numerical.shape

<class 'pandas.core.frame.DataFrame'>
Int64Index: 2650 entries, 1 to 9979
Data columns (total 5 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   Income              2650 non-null   float64
 1   Outage_sec_perweek  2650 non-null   float64
 2   Tenure              2650 non-null   float64
 3   MonthlyCharge       2650 non-null   float64
 4   Bandwidth_GB_Year   2650 non-null   float64
dtypes: float64(5)
memory usage: 124.2 KB

Out[]: (2650, 5)


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

<class 'pandas.core.frame.DataFrame'>
Int64Index: 2650 entries, 1 to 9979
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   INC     2650 non-null   float64
 1   OUT     2650 non-null   float64
 2   TEN     2650 non-null   float64
 3   MCH     2650 non-null   float64
 4   BAN     2650 non-null   float64
dtypes: float64(5)
memory usage: 124.2 KB

Out[]: (2650, 5)



# describe numerical data 
df_numerical.describe().round(3)

Out[]: 
+-------+-----------+---------+---------+---------+---------+
| STAT  |    INC    |   OUT   |   TEN   |   MCH   |   BAN   |
+-------+-----------+---------+---------+---------+---------+
| count |   2650.00 | 2650.00 | 2650.00 | 2650.00 | 2650.00 |
| mean  |  40085.76 |   10.00 |   13.15 |  199.30 | 1785.01 |
| std   |  28623.99 |    2.97 |   15.58 |   41.27 | 1375.37 |
| min   |    348.67 |    0.23 |    1.00 |   92.46 |  248.18 |
| 25%   |  19234.99 |    8.02 |    4.07 |  167.48 |  981.30 |
| 50%   |  33609.94 |    9.96 |    7.87 |  200.12 | 1357.83 |
| 75%   |  54178.77 |   11.95 |   13.76 |  232.64 | 1904.88 |
| max   | 189938.40 |   21.21 |   71.65 |  290.16 | 7096.49 |
+-------+-----------+---------+---------+---------+---------+



# describe variables as continuous or categorical
describe_dataframe_type(df_numerical)

1. INC is numerical (CONTINUOUS) - type: float64.
  Min: 348.670  Max: 189938.400  Std: 28623.988

2. OUT is numerical (CONTINUOUS) - type: float64.
  Min: 0.232  Max: 21.207  Std: 2.970

3. TEN is numerical (CONTINUOUS) - type: float64.
  Min: 1.000  Max: 71.646  Std: 15.577

4. MCH is numerical (CONTINUOUS) - type: float64.
  Min: 92.455  Max: 290.160  Std: 41.268

5. BAN is numerical (CONTINUOUS) - type: float64.
  Min: 248.179  Max: 7096.495  Std: 1375.370
  
  
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
    
    
    
# remove highly correlated variables
df_final = df_numerical.drop(columns=['BAN'])
df_final.info()
df_final.shape

<class 'pandas.core.frame.DataFrame'>
Int64Index: 2650 entries, 1 to 9979
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   INC     2650 non-null   float64
 1   OUT     2650 non-null   float64
 2   TEN     2650 non-null   float64
 3   MCH     2650 non-null   float64
dtypes: float64(4)
memory usage: 103.5 KB

Out[]: (2650, 3)


# standardize remaining numerical data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_final.values)
df_standardized = pd.DataFrame(scaled_features,
        index=df_final.index, 
      columns=df_final.columns)
df_standardized.describe().round(2)

+-------+---------+---------+---------+---------+
|  STD  |   INC   |   OUT   |   TEN   |   MCH   |
+-------+---------+---------+---------+---------+
| count | 2650.00 | 2650.00 | 2650.00 | 2650.00 |
| mean  |   -0.00 |    0.00 |   -0.00 |   -0.00 |
| std   |    1.00 |    1.00 |    1.00 |    1.00 |
| min   |   -1.39 |   -3.29 |   -0.78 |   -2.59 |
| 25%   |   -0.73 |   -0.67 |   -0.58 |   -0.77 |
| 50%   |   -0.23 |   -0.01 |   -0.34 |    0.02 |
| 75%   |    0.49 |    0.66 |    0.04 |    0.81 |
| max   |    5.24 |    3.77 |    3.76 |    2.20 |
+-------+---------+---------+---------+---------+


# use boxplot to look for outliers
fig, ax = plt.subplots(figsize =(7, 5))
ax = df_standardized.boxplot(vert=False)



# create knee plot
fig, ax = plt.subplots(figsize =(7, 5))
kl = KneeLocator(range(1, 11), sse, 
   curve="convex", direction="decreasing")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Knee Plot")
plt.axvline(x=kl.elbow, color='green', ls=':', lw=2,)
fig.savefig("figures/fig_2", dpi=150)


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



In [13]: kmeans_kwargs = {
   ...:     "init": "random",
   ...:     "n_init": 10,
   ...:     "max_iter": 300,
   ...:     "random_state": 42,
   ...: }
   ...:
   ...: # A list holds the SSE values for each k
   ...: sse = []
   ...: for k in range(1, 11):
   ...:     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
   ...:     kmeans.fit(scaled_features)
   ...:     sse.append(kmeans.inertia_)
   
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
plt.axvline(x=kl.elbow, color='green', ls=':', lw=2,)
fig.savefig("figures/fig_2", dpi=150)


# optimum point on knee plot
'Optimum: ({}, {:.3f})'.format(knee.elbow, sse[knee.elbow-1])

Out[]: 'Optimum: (4, 5326.264)'
