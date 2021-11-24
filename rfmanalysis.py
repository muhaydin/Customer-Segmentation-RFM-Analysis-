

#### Importing Relevant Python Modules ####

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os


#### Importing Data from Current Working Directory to Python ####
data = pd.read_excel(os.getcwd() + "\\data_mus_seg.xlsx")
data = data.set_index('Musteri_ID')
data2 = data[['R','F','M']]


#### The skewness determination is made by looking at the histograms of the R, F and M values. ####

plt.figure(figsize=(10,6))
sns.distplot(data2['R'],label='recency')
plt.title('Distribution of Recency')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
sns.distplot(data2['F'],label='frequency')
plt.title('Distribution of Frequency')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
sns.distplot(data2['M'],label='monetary')
plt.title('Distribution of Monetary')
plt.legend()
plt.show()

#### The logarithm was taken because the data were skewed. ####

data3 = np.log(data2)

#### Since the units of each of the data are different, standardization is applied. ####

scaler = StandardScaler()
scaler.fit(data3)
data_normalized = scaler.transform(data3)
data_normalized2 = pd.DataFrame(np.array(data_normalized))

#### Cluster number selection with Elbow method ####
sse={}
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data_normalized2)
    sse[k] = kmeans.inertia_

plt.figure(figsize=(10,6))
plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.text(4.5,60,"Largest Angle",bbox=dict(facecolor='lightgreen', alpha=0.5))
plt.show()
plt.savefig('elbow.png')

#### After selecting the number of clusters 6, the cluster the existing data belongs to is added to the data table. ####

kmeans = KMeans(n_clusters=6, random_state=1) 
kmeans.fit(data_normalized2)

cluster_labels = kmeans.labels_
data['Küme'] = cluster_labels+1

#### Cluster Centers ####

centers = kmeans.cluster_centers_

#### Data Labels ####

seg_map = {
        r'[1-2][1-2]': 'Hibernating',
        r'[1-2][3-4]': 'At Risk',
        r'[1-2]5': 'Can\'t Loose',
        r'3[1-2]': 'About to Sleep',
        r'33': 'Need Attention',
        r'[3-4][4-5]': 'Loyal Customers',
        r'41': 'Promising',
        r'51': 'New Customers',
        r'[4-5][2-3]': 'Potential Loyalists',
        r'5[4-5]': 'Champions'
}

data['Segment'] = data['Rscore'].astype(str) + data['Fscore'].astype(str)
data['Segment'] = data['Segment'].replace(seg_map, regex=True)

#### Cluster/Segment Report ####
data['RFMSegment'] = np.array(data['Küme'])
data4 = pd.DataFrame(data.groupby(['Küme','Segment'])['Segment'].count())

#### RFM /Cluster (Mean,Count) Report ####

data5 = data[["Küme", "R","F","M"]].groupby(["Küme"]).agg(["mean","count","sum"])

#Write Data to Excel

data.to_excel("rfm_analysis.xlsx", sheet_name='Clustered_Data') 
 



