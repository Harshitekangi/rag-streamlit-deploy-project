# Self-contained Streamlit app (no backend required)
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="RAG-DB (Self-contained)", layout="wide")
st.title("RAG-DB — Self-contained Instacart Analytics")
st.write("This app uses CSVs stored inside the repo — no FastAPI backend needed.")

DATA_DIR = os.path.join("data", "data", "instacart")

def load_csv_safe(name, nrows=None):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, nrows=nrows)

st.sidebar.header("Options")
sample_nrows_prior = st.sidebar.number_input(
    "Number of rows to load from order_products__prior.csv (0 = full)", 
    min_value=0, 
    value=50000, 
    step=10000
)

mode_override = st.sidebar.selectbox("Force mode", ["auto","retrieval","aggregation"])

q = st.text_area("Ask a question:", 
                 value="Which products appear most frequently in prior orders?",
                 height=140)
run = st.button("Run Query")

@st.cache_data(ttl=3600)
def load_tables(nrows_prior=None):
    products = load_csv_safe("products")
    aisles = load_csv_safe("aisles")
    departments = load_csv_safe("departments")
    orders = load_csv_safe("orders")
    prior = load_csv_safe("order_products__prior", nrows=nrows_prior)
    return {
        "products": products,
        "aisles": aisles,
        "departments": departments,
        "orders": orders,
        "prior": prior
    }

if run:
    st.info("Loading CSVs from the repo...")
    nrows = None if sample_nrows_prior == 0 else int(sample_nrows_prior)
    tables = load_tables(nrows_prior=nrows)

    products = tables["products"]
    aisles = tables["aisles"]
    prior = tables["prior"]
    orders = tables["orders"]

    if products is None or prior is None:
        st.error("Missing CSVs! Upload products.csv and order_products__prior.csv to data/data/instacart/")
        st.stop()

    ql = q.lower().strip()

    # --- Basic intent detection ---
    agg_keywords = ["count","top","most","frequent","total","sum","avg","average","mean"]
    retrieval_keywords = ["show","list","example","what","which","find"]

    intent = (
        "aggregation" if any(k in ql for k in agg_keywords) 
        else "retrieval"
    )

    if mode_override != "auto":
        intent = mode_override

    st.markdown(f"### Intent detected: `{intent}`")

    # =============== AGGREGATION MODE =====================
    if intent == "aggregation":
        if "most" in ql and ("product" in ql or "order" in ql or "frequent" in ql):
            st.subheader("Top Products in Prior Orders")

            counts = prior["product_id"].value_counts().head(10)
            df = counts.rename_axis("product_id").reset_index(name="count")

            # join product names
            prod_idx = products.set_index("product_id")
            df["product_name"] = df["product_id"].apply(
                lambda x: prod_idx.loc[x]["product_name"] if x in prod_idx.index else "Unknown"
            )

            df = df[["product_name", "count"]].rename(columns={"product_name":"x","count":"y"})
            st.dataframe(df)

            fig, ax = plt.subplots(figsize=(10,4))
            ax.bar(df["x"], df["y"])
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

        else:
            st.info("Aggregation detected, but I don't have a handler for this query.")
    
    # =============== RETRIEVAL MODE ======================
    else:
        st.subheader("Keyword Search Results")

        results = []

        # Search products by name
        mask_prod = products["product_name"].str.lower().str.contains(ql, na=False)
        prod_hits = products[mask_prod].head(20)
        for _, r in prod_hits.iterrows():
            results.append(f"PRODUCT: {r['product_id']} | {r['product_name']}")

        # Search aisles
        if not results and aisles is not None:
            mask_aisle = aisles["aisle"].str.lower().str.contains(ql, na=False)
            aisle_hits = aisles[mask_aisle].head(20)
            for _, r in aisle_hits.iterrows():
                results.append(f"AISLE: {r['aisle_id']} | {r['aisle']}")

        if results:
            for item in results:
                st.write(item)
        else:
            st.warning("No matches found. Try simpler keywords like 'snacks', 'frozen', 'milk'.")
