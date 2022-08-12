# [Project 2](https://github.com/thatssweety/Mall-Customer-Segmentation) :  Mall Customer Segmentation
![dataset-cover](https://user-images.githubusercontent.com/81384066/184426662-f78de366-0053-4892-8d80-82948369491a.jpg)
## [DATASET USED](https://github.com/thatssweety/Mall-Customer-Segmentation/blob/main/DATASET.zip) : <br>
###             [Kaggle Mall Customer Segmentation Data (for learning purpose)](https://github.com/thatssweety/Mall-Customer-Segmentation/blob/main/DATASET.zip)<br>

<img width="563" alt="Screenshot (445)" src="https://user-images.githubusercontent.com/81384066/184425856-b19e0166-6fff-4704-bbfb-43563cda4cfa.png">


## Tech Stack used:
1. Numpy
2. Pandas
3. Sklearn
4. Matplotlib
# Code

## Importing libraries

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

```
## Data Collection and Analysis
```python
dataset_customers=pd.read_csv('/content/Mall_Customers.csv')
print(dataset_customers.head())
X=dataset_customers.iloc[:,[3,4]].values #only need last two columns
```
## Finding Apt Number of Clusters
```python
min_dis_sum=[]
for i in range(1,11):
    model= KMeans(n_clusters=i,init='k-means++',random_state=23)
    model.fit(X)
    min_dis_sum.append(model.inertia_)
  
```
##  ELBOW POINT GRAPH
```python
plt.plot(range(1,11),min_dis_sum)
plt.title('The Elbow point graph')
plt.xlabel('Number of clusters')
plt.ylabel('Within Clusters Sum of squares ')
plt.show()
```
![download (1)](https://user-images.githubusercontent.com/81384066/184424580-229302da-58c8-4c88-9cd0-259c6347cf03.png)
## Appropriate or optimum number of clusters = 5
### Plot Within Clusters Sum of squares to number of clusters
```python
model=KMeans(n_clusters=5,init='k-means++',random_state=0)
Y=model.fit_predict(X) #return a label for each data point on their cluster

```
## Visualizing the 5 clusters and their centroids
```python
plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0],X[Y==0,1],s=50,c='green',label='cluster 1')
plt.scatter(X[Y==1,0],X[Y==1,1],s=50,c='yellow',label='cluster 2')
plt.scatter(X[Y==2,0],X[Y==2,1],s=50,c='cyan',label='cluster 3')
plt.scatter(X[Y==3,0],X[Y==3,1],s=50,c='violet',label='cluster 4')
plt.scatter(X[Y==4,0],X[Y==4,1],s=50,c='blue',label='cluster 5')
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=100,c='red',label='Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.show()
```
![download (2)](https://user-images.githubusercontent.com/81384066/184424607-ac29d4c3-e5c9-4026-80d0-9b1beb2925cf.png)





