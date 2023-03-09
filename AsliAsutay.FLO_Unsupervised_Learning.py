# Görev 1:  Veriyi Hazırlama
# Adım1:  flo_data_20K.csv verisiniokutunuz

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("/Users/Asli1/datasets/flo_data_20K.csv")
df = df_.copy()
df.head()
df.describe().T


# aykırı değer olabilir

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


# NA değeri yok, zamanı gösteren özelliklerin tipi değişmeli

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


grab_col_names(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# Adım2:  Müşterilerisegmentlerkenkullanacağınızdeğişkenleriseçiniz
# Not: Tenure (Müşterininyaşı), Recency (enson kaçgünöncealışverişyaptığı) gibiyeni değişkenleroluşturabilirsiniz

# date_columns = df.columns[df.columns.str.contains("date")]
# df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Her bir müşterinin toplam alışveriş sayısı ve harcaması
df["TotalOrder"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["TotalPrice"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)

df["Recency"] = today_date - pd.to_datetime(df["last_order_date"])
df["frequency"] = df["TotalOrder"]
df["Monetory"] = df["TotalPrice"]
df["Tenure"] = today_date - pd.to_datetime(df["first_order_date"])

df.head()

# Görev 2:  K-Means ileMüşteriSegmentasyonu
# Adım 1: Değişkenleristandartlaştırınız

cat_cols = [col for col in df.columns if df[col].dtypes not in ["int", "int64", "float64"]]


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]

# Adım 2: Optimum kümesayısınıbelirleyiniz.

kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

kmeans = KMeans()
ssd = []
K = range(1, 20)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(3, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_
#7

# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

clusters_kmeans = kmeans.labels_
df["CLUSTER"] = clusters_kmeans
df["CLUSTER"] = df["CLUSTER"] + 1

kmeans_final = pd.read_csv("flo_data_20K.csv")
kmeans_final["CLUSTER"] = clusters_kmeans
kmeans_final["CLUSTER"] = kmeans_final["CLUSTER"] + 1
kmeans_final.head()

# Adım 4: Her bir segmenti istatistiksel olarak inceleyeniz.

kmeans_final.groupby("CLUSTER").agg(["count", "mean", "median"])

# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
# Adım 1: Görev2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)


hc_average = linkage(df, "average")

plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

#Adım 2: Modelinizi oluşturunuz vemüşterilerinizsegmentleyiniz.


cluster = AgglomerativeClustering(n_clusters=4, linkage="average")

clusters_hi = cluster.fit_predict(df)

df["CLUSTER_HI"] = clusters_hi

df["CLUSTER_HI"] = df["CLUSTER_HI"] + 1

df.head()

# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.

df.groupby("CLUSTER_HI").agg(["count", "mean", "median"])
