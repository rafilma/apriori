import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori

df = pd.read_csv("penjualan barang.csv")

df['tanggal'] = pd.to_datetime(df['tanggal'], format="%Y-%m-%d")

df["month"] = df['tanggal'].dt.month
df["day"] = df['tanggal'].dt.day

df["month"].replace([i for i in range(1, 12 + 1)], ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], inplace=True)
st.title("Market Basket Analysis Dengan Apriori")

def get_data(month='', day=''):
    data = df.copy()
    filtered = data.loc[
        (data["month"].str.contains(month.title())) &
        (data["day"].astype(str).str.contains(day.title()))  # Convert day to string for matching
    ]
    return filtered if filtered.shape[0] else "No result"

def user_input_features():
    item = st.selectbox("Item", ['Beras', 'Daging', 'Gula', 'Migor', 'Tepung'])
    month = st.select_slider("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day = st.select_slider("Tanggal", [str(i) for i in range(1, 32)], value="1")  # Convert day to string for matching

    return month, day, item

month, day, item = user_input_features()

data = get_data(month, day)

def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1

rules = None  # Initialize rules outside the conditional block

if type(data) != type("No result"):
    item_count = df.groupby(["nama.pembeli", "nama.barang"])["nama.barang"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='nama.pembeli', columns='nama.barang', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.2
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_threshold = 1

    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)

# Display rules inside the conditional block
if rules is not None:
    st.write("Generated Association Rules:")
    st.write(rules)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    filtered_data = data.loc[data["antecedents"] == item_antecedents]

    if not filtered_data.empty:
        first_row = list(filtered_data.iloc[0, :])
        return first_row
    else:
        return ["No result", "No result"]

if rules is not None and type(data) != type("No result"):
    result = return_item_df(item)
    if result[0] == "No result":
        st.warning("No recommendation found.")
    else:
        st.markdown("Hasil Rekomendasi : ")
        st.success(f"Jika konsumen membeli **{item}**, maka membeli **{result[1]}** secara bersamaan")

