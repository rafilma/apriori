import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

df = pd.read_csv("penjualan barang.csv")

df['tanggal'] = pd.to_datetime(df['tanggal'], format= "%Y-%m-%d")

df["month"] = df['tanggal'].dt.month
df["day"] = df['tanggal'].dt.day

df["month"].replace([i for i in range (1, 12 + 1)], ["January","February","March","April","May","June","July","August","September","October","November","December"], inplace=True)
st.title ("Market Basket Analysis Dengan Apriori")

def get_data(month = '', day = ''):
    data = df.copy()
    filtered = data.loc[
        (data["month"].str.contains(month.title())) &
        (data["day"].astype(str).str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No result"

def user_input_features():
    item = st.selectbox("Item", ['Beras','Daging','Gula','Migor','Tepung'])
    month = st.select_slider("Month", ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    day = st.select_slider("Tanggal", [str(i) for i in range(1, 32)], value="1")
    return month, day, item

month, day, item = user_input_features()

data = get_data(month,day)

def encode(x):
    if x <=0:
        return 0
    elif x >= 1:
        return 1

if type(data) != type("No result"):
    item_count = df.groupby(["nama.pembeli", "nama.barang"])["nama.barang"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='nama.pembeli', columns='nama.barang', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.2
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_treshold = 1

    rules = association_rules(frequent_items, metric=metric, min_threshold=min_treshold)[["antecedents","consequents","support","confidence","lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)
    
if 'rules' in locals():
    st.write("Generated Association Rules:")
    st.write(rules)
else:
    st.warning("No rules generated. Please adjust your input criteria.")


def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

def return_item_df(item_antecedents):
    data_filtered = rules[(rules["antecedents"].apply(lambda x: item_antecedents in x))]
    if not data_filtered.empty:
        antecedent = parse_list(data_filtered['antecedent'].value[0])
        consequent = parse_list(data_filtered['consequent'].value[0])
        return [antecedent, consequent]
    else:
        return ["No result", "No result"]

if type(data) != type("No result"):
    st.markdown("Hasil Rekomendasi : ")
    st.success(f"Jika konsumen membeli **{item}**, maka membeli **{return_item_df(item)[0]}** secara bersamaan")
