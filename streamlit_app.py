# streamlit_app.py
"""
RAG-DB — Instacart explorer (self-contained + optional HF LLM)
Usage (local):
    streamlit run streamlit_app.py --server.port 8501

Put CSVs under: data/data/instacart/{products,aisles,departments,orders,order_products__prior}.csv
If you want LLM suggestions, provide HF_API_KEY in .env or paste in sidebar.
"""
import os, json, re, time
from pathlib import Path
from typing import Optional, Any, Dict
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv

# Optional HF client import - handled safely
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except Exception:
    HF_HUB_AVAILABLE = False

# load .env (if exists)
load_dotenv()

# --- Config / paths ---
st.set_page_config(page_title="RAG-DB — Instacart explorer", layout="wide")
BASE = Path(".").resolve()
DATA_DIR = BASE / "data" / "data" / "instacart"

# ---------- Helpers ----------
def load_csv_if_exists(name: str, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, nrows=nrows)

def detect_intent(q: str) -> str:
    ql = q.lower()
    agg_keywords = ["count","top","most","frequent","total","sum","avg","average","mean","least","how many","per product","reorder","ratio","percentage","orders by","which days","day of week"]
    retrieval_keywords = ["show","list","example","what are","what is","which aisles","give me","find","show me"]
    if any(k in ql for k in agg_keywords):
        return "aggregation"
    if any(k in ql for k in retrieval_keywords):
        return "retrieval"
    return "retrieval"

def tokenize(s: str):
    return [t for t in re.split(r"[^0-9a-z]+", s.lower()) if t]

def fuzzy_search_products(products: pd.DataFrame, q: str, top_k: int = 30):
    if products is None: return []
    qtokens = tokenize(q)
    out=[]
    for _,r in products.iterrows():
        name = str(r.get("product_name",""))
        ntoks = tokenize(name)
        if not ntoks: continue
        overlap = sum(1 for qt in qtokens if any(qt in nt for nt in ntoks))
        if overlap>0:
            score = overlap/len(ntoks)
            out.append({"product_id": r["product_id"], "product_name": name, "score": score})
    out = sorted(out, key=lambda x:-x["score"])
    # dedupe
    seen=set(); res=[]
    for it in out:
        if it["product_id"] in seen: continue
        seen.add(it["product_id"]); res.append(it)
        if len(res)>=top_k: break
    return res

# --- Aggregation functions (exact local) ---
def top_products_prior(prior: pd.DataFrame, products: pd.DataFrame, top_k: int = 10):
    counts = prior["product_id"].value_counts().head(top_k)
    df = counts.rename_axis("product_id").reset_index(name="count")
    if products is not None:
        prod_idx = products.set_index("product_id")
        df["product_name"] = df["product_id"].apply(lambda x: prod_idx.loc[x]["product_name"] if x in prod_idx.index else str(x))
    else:
        df["product_name"] = df["product_id"].astype(str)
    df = df[["product_name","count"]].rename(columns={"product_name":"x","count":"y"})
    return df

def least_products_prior(prior: pd.DataFrame, products: pd.DataFrame, top_k: int = 10):
    counts = prior["product_id"].value_counts()
    tail = counts[counts>0].tail(top_k)
    df = tail.rename_axis("product_id").reset_index(name="count")
    if products is not None:
        prod_idx = products.set_index("product_id")
        df["product_name"] = df["product_id"].apply(lambda x: prod_idx.loc[x]["product_name"] if x in prod_idx.index else str(x))
    else:
        df["product_name"] = df["product_id"].astype(str)
    df = df[["product_name","count"]].rename(columns={"product_name":"x","count":"y"})
    return df

def avg_orders_per_product(prior: pd.DataFrame):
    counts = prior["product_id"].value_counts()
    return float(counts.mean())

def avg_reorder_ratio(prior: pd.DataFrame):
    if "reordered" in prior.columns:
        return float(prior.groupby("product_id")["reordered"].mean().mean())
    return None

def orders_by_day_of_week(orders: pd.DataFrame):
    if orders is None or "order_dow" not in orders.columns:
        return None
    counts = orders["order_dow"].value_counts().sort_index()
    df = counts.rename_axis("day").reset_index(name="count")
    df["x"] = df["day"].astype("str"); df["y"] = df["count"]
    return df[["x","y"]]

# compute generic chart from a LLM-suggested "chart_spec"
def compute_chart_from_spec(chart_spec: dict, tables: dict):
    try:
        table_name = chart_spec.get("table")
        if table_name not in tables or tables[table_name] is None:
            return None
        df = tables[table_name].copy()
        # optional join
        js = chart_spec.get("join")
        if js:
            right_table = js.get("table")
            left_on = js.get("left_on")
            right_on = js.get("right_on")
            right_label = js.get("right_label")
            if right_table in tables and tables[right_table] is not None:
                right_df = tables[right_table][[right_on,right_label]].drop_duplicates()
                df = df.merge(right_df, left_on=left_on, right_on=right_on, how="left")
        xcol = chart_spec.get("x")
        ycol = chart_spec.get("y")
        agg = chart_spec.get("agg","count")
        top_k = int(chart_spec.get("top_k",10))
        if agg == "count":
            grouped = df.groupby(xcol).size().rename("y").reset_index().sort_values("y",ascending=False).head(top_k)
        elif agg == "sum":
            grouped = df.groupby(xcol)[ycol].sum().rename("y").reset_index().sort_values("y",ascending=False).head(top_k)
        elif agg == "avg":
            grouped = df.groupby(xcol)[ycol].mean().rename("y").reset_index().sort_values("y",ascending=False).head(top_k)
        else:
            return None
        grouped = grouped.rename(columns={xcol:"x"})[["x","y"]]
        return grouped
    except Exception as e:
        st.exception(e)
        return None

def render_chart(df: pd.DataFrame, chart_type: str="bar", title: str=""):
    if df is None or df.empty:
        st.info("No chart data to render.")
        return
    if chart_type=="bar":
        fig = px.bar(df, x="x", y="y", title=title)
    elif chart_type=="line":
        fig = px.line(df, x="x", y="y", title=title)
    elif chart_type=="pie":
        fig = px.pie(df, names="x", values="y", title=title)
    elif chart_type=="treemap":
        fig = px.treemap(df, path=["x"], values="y", title=title)
    else:
        fig = px.bar(df, x="x", y="y", title=title)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- LLM helpers (Hugging Face) ----------------
def call_hf_chat_completion(hf_token: str, model_id: str, user_question: str, short_context: Any=None, max_tokens: int=400):
    """
    Calls InferenceClient.chat_completion to return a JSON suggestion.
    Returns a Python dict. If LLM reply cannot be parsed as JSON, returns fallback dict with raw_text
    """
    if not HF_HUB_AVAILABLE:
        raise RuntimeError("huggingface_hub not installed")
    client = InferenceClient(token=hf_token)
    system = (
        "You are a helpful assistant. Use ONLY the provided short context (exact numbers/summaries) "
        "and do not hallucinate additional numeric facts. Return a SINGLE JSON object ONLY."
    )
    short_ctx_txt = json.dumps(short_context, ensure_ascii=False, indent=2) if short_context else "{}"
    user_msg = (
        f"Context (short):\n{short_ctx_txt}\n\nUser question:\n{user_question}\n\n"
        "Return JSON with keys:\n"
        "- answer_text: short 1-2 sentences\n"
        "- chart_type: one of [\"bar\",\"line\",\"pie\",\"treemap\",\"none\"]\n"
        "- chart_spec: object {table, x, y, agg: 'count'|'sum'|'avg', join: optional, top_k: optional}\n"
        "- followups: optional list of short follow-up Qs\n"
    )
    messages = [{"role":"system","content":system},{"role":"user","content":user_msg}]
    resp = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.0)
    # extract text
    try:
        content = resp.choices[0].message.content
    except Exception:
        content = str(resp)
    # try to extract json block
    m = re.search(r"\{[\s\S]*\}", content)
    if m:
        jtxt = m.group(0)
        try:
            jobj = json.loads(jtxt)
            return jobj
        except Exception:
            # parsing error -> return raw fallback
            return {"answer_text": content.strip()[:1000], "chart_type":"none", "chart_spec":{}, "followups":[]}
    else:
        return {"answer_text": content.strip()[:1000], "chart_type":"none", "chart_spec":{}, "followups":[]}

# ---------------- Streamlit UI ----------------
st.title("RAG-DB — Instacart explorer (local + optional LLM)")
st.sidebar.header("Options / Settings")

sample_nrows_prior = st.sidebar.number_input("Rows to load from order_products__prior (0 = full)", min_value=0, value=50000, step=10000)
mode = st.sidebar.selectbox("Mode", ["local-only","with-llm"])
hf_token_input = st.sidebar.text_input("Hugging Face API key (optional)", value=os.getenv("HF_API_KEY",""), type="password")
hf_model_input = st.sidebar.text_input("HF model id (chat-capable)", value="meta-llama/Meta-Llama-3-8B-Instruct")
st.sidebar.markdown("---")
st.sidebar.write("Notes: local-only always runs exact queries. with-llm asks HF to suggest friendly text & chart_spec; chart computed locally.")

q = st.text_area("Ask a question", value="Which products appear most frequently in prior orders?", height=140)
run = st.button("Run")

@st.cache_data(ttl=3600)
def load_tables(nrows_prior: Optional[int]=None):
    products = load_csv_if_exists("products")
    aisles = load_csv_if_exists("aisles")
    departments = load_csv_if_exists("departments")
    orders = load_csv_if_exists("orders")
    prior = load_csv_if_exists("order_products__prior", nrows=nrows_prior)
    return {"products":products,"aisles":aisles,"departments":departments,"orders":orders,"prior":prior}

if run:
    st.info("Loading CSVs...")
    nrows = None if sample_nrows_prior==0 else int(sample_nrows_prior)
    tbls = load_tables(nrows_prior=nrows)
    products = tbls["products"]; aisles = tbls["aisles"]; departments = tbls["departments"]; orders = tbls["orders"]; prior = tbls["prior"]
    if prior is None or products is None:
        st.error("Missing CSVs: ensure products.csv and order_products__prior.csv are in data/data/instacart/")
        st.stop()

    intent = detect_intent(q)
    st.markdown(f"### Detected intent: **{intent}**")

    # prepare short context (small exact summary)
    short_context = {}
    try:
        short_context["rows_in_prior"] = int(len(prior))
        short_context["top5"] = top_products_prior(prior, products, top_k=5).to_dict(orient="records")
    except Exception as e:
        short_context["error"] = str(e)

    use_llm = (mode=="with-llm") and hf_token_input.strip()!="" and HF_HUB_AVAILABLE
    tables_map = {"prior":prior,"products":products,"orders":orders,"aisles":aisles,"departments":departments}

    # --- Aggregation handlers ---
    if intent=="aggregation":
        ql = q.lower()
        # top products
        if any(x in ql for x in ["most frequently","most frequent","most frequently ordered","top products","most ordered","appear most frequently","most frequent product"]):
            df = top_products_prior(prior, products, top_k=15)
            st.subheader("Top products (exact counts)")
            st.dataframe(df.rename(columns={"x":"product_name","y":"count"}).head(15))
            render_chart(df, chart_type="bar", title="Top products (count)")
            # extra visualizations
            with st.expander("More views"):
                render_chart(df, chart_type="treemap", title="Top products (treemap)")
                render_chart(df, chart_type="pie", title="Top products (pie)")
        elif any(x in ql for x in ["least frequently","least ordered"]):
            df = least_products_prior(prior, products, top_k=15)
            st.subheader("Least frequently ordered products (sample)")
            st.dataframe(df.rename(columns={"x":"product_name","y":"count"}))
            render_chart(df, chart_type="bar", title="Least frequently ordered")
        elif "average number of orders per product" in ql or "average orders per product" in ql or "avg orders per product" in ql:
            avg = avg_orders_per_product(prior)
            st.write(f"**Average occurrences per product** (mean of counts): {avg:.2f}")
        elif "reorder ratio" in ql or "average reorder" in ql or "reorder rate" in ql:
            r = avg_reorder_ratio(prior)
            if r is not None:
                st.write(f"Average reorder ratio across products: **{r:.3f}**")
            else:
                st.info("No 'reordered' column present in this dataset sample.")
        elif "day" in ql or "day of week" in ql or "which days" in ql:
            df = orders_by_day_of_week(orders)
            if df is not None:
                st.subheader("Orders by day of week")
                st.dataframe(df.rename(columns={"x":"day","y":"count"}))
                render_chart(df, chart_type="bar", title="Orders by day (0=Sunday..6=Saturday)")
        else:
            # no exact handler → try LLM suggestion or fallback to top products
            local_df = top_products_prior(prior, products, top_k=10)
            local_answer = f"Top product: {local_df.iloc[0]['x']} with {int(local_df.iloc[0]['y'])} occurrences."
            if use_llm:
                with st.spinner("Asking LLM for a friendly summary and chart suggestion..."):
                    try:
                        jobj = call_hf_chat_completion(hf_token_input.strip(), hf_model_input.strip(), q, short_context)
                        st.subheader("LLM suggested answer and chart_spec (parsed)")
                        st.json(jobj)
                        st.write("LLM answer:")
                        st.write(jobj.get("answer_text","(no answer_text)"))
                        # compute chart_spec if present
                        cs = jobj.get("chart_spec",{})
                        ct = jobj.get("chart_type","bar")
                        if cs:
                            chart_df = compute_chart_from_spec(cs, tables_map)
                            if chart_df is not None:
                                render_chart(chart_df, chart_type=ct, title="LLM suggested chart (computed exactly)")
                            else:
                                st.warning("Could not compute chart_spec locally — check chart_spec fields (table/x/y).")
                        if jobj.get("followups"):
                            st.markdown("**Follow-up suggestions:**")
                            for f in jobj["followups"][:5]:
                                st.write("-", f)
                    except Exception as e:
                        st.error(f"LLM step failed: {e}")
                        st.info("Showing local fallback result:")
                        st.write(local_answer)
                        render_chart(local_df, chart_type="bar", title="Local fallback: top products")
            else:
                st.info("No handler matched and LLM disabled. Showing local fallback (top products).")
                st.write(local_answer)
                render_chart(local_df, chart_type="bar", title="Local fallback: top products")

    # --- Retrieval handlers ---
    else:
        st.subheader("Retrieval / Keyword search")
        proc = q.strip()
        prod_hits = fuzzy_search_products(products, proc, top_k=30)
        if prod_hits:
            st.write(f"Products matching query (top {len(prod_hits)}):")
            df = pd.DataFrame(prod_hits)[["product_id","product_name"]]
            st.dataframe(df)
            # optional LLM summary
            if use_llm:
                with st.spinner("Asking LLM to summarize results..."):
                    try:
                        jobj = call_hf_chat_completion(hf_token_input.strip(), hf_model_input.strip(), q, short_context, max_tokens=200)
                        st.subheader("LLM summary (optional)")
                        st.write(jobj.get("answer_text","(no LLM answer)"))
                    except Exception:
                        st.info("LLM summary failed; continuing.")
        else:
            st.info("No product fuzzy match found. Trying aisles substring match.")
            if aisles is not None:
                mask = aisles["aisle"].str.lower().str.contains(proc.lower(), na=False)
                if mask.any():
                    st.write("Matching aisles (examples):")
                    st.dataframe(aisles[mask].head(30))
                else:
                    st.warning("No matches found. Try simpler keywords like 'frozen', 'snacks', 'produce'.")
            else:
                st.warning("Aisles data not available.")

    st.success("Done.")