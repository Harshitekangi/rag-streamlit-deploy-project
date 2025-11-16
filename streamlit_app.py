# streamlit_app_experiment.py
"""
Experimental Streamlit app for RAG-DB (Instacart) — experiment branch.
- Local exact aggregation + retrieval
- Optional robust Hugging Face LLM for friendly summary & chart_spec suggestion
- Defensive: local facts always win over LLM contradictions
"""
import os, json, re, time
from pathlib import Path
from typing import Optional, Any, Dict, List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
import requests
from requests.exceptions import RequestException

# --- HF client availability ---
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except Exception:
    HF_HUB_AVAILABLE = False

# load .env
load_dotenv()

# Config
st.set_page_config(page_title="RAG-DB — Instacart (experiment)", layout="wide")
BASE = Path(".").resolve()
DATA_DIR = BASE / "data" / "data" / "instacart"

# Backend check (optional, keep for your app)
BACKEND_BASE = "http://127.0.0.1:8000"
def backend_is_healthy(timeout: float = 0.7) -> bool:
    try:
        resp = requests.get(f"{BACKEND_BASE}/ping", timeout=timeout)
        return resp.status_code == 200
    except RequestException:
        return False

# ---------- Helpers ----------
def load_csv_if_exists(name: str, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, nrows=nrows)

def detect_intent(q: str) -> str:
    ql = (q or "").lower()
    agg_keywords = ["count","top","most","frequent","total","sum","avg","average","mean","least","how many","per product","reorder","ratio","percentage","orders by","which days","day of week"]
    retrieval_keywords = ["show","list","example","what are","what is","which aisles","give me","find","show me","list items","list products","contain"]
    if any(k in ql for k in agg_keywords):
        return "aggregation"
    if any(k in ql for k in retrieval_keywords):
        return "retrieval"
    return "retrieval"

def tokenize(s: str):
    return [t for t in re.split(r"[^0-9a-z]+", (s or "").lower()) if t]

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
            out.append({"product_id": int(r["product_id"]), "product_name": name, "score": score})
    out = sorted(out, key=lambda x:-x["score"])
    seen=set(); res=[]
    for it in out:
        if it["product_id"] in seen: continue
        seen.add(it["product_id"]); res.append(it)
        if len(res)>=top_k: break
    return res

# --- Aggregation functions ---
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
_ALLOWED_TABLES = {"prior","products","orders","aisles","departments"}

def _validate_chart_spec(raw):
    if not isinstance(raw, dict):
        return {}
    table = raw.get("table")
    x = raw.get("x")
    y = raw.get("y")
    agg = raw.get("agg","count")
    top_k = int(raw.get("top_k",10)) if raw.get("top_k") is not None else 10
    join = raw.get("join")
    if table not in _ALLOWED_TABLES:
        return {}
    if not x:
        if table == "prior":
            x = "product_id"
        elif table == "orders":
            x = "order_dow"
        else:
            return {}
    if agg not in {"count","sum","avg"}:
        agg = "count"
    cleaned = {"table":table,"x":x,"y":y,"agg":agg,"top_k":top_k}
    if join and isinstance(join, dict):
        jt = join.get("table")
        if jt in _ALLOWED_TABLES:
            cleaned["join"] = {"table": jt, "left_on": join.get("left_on"), "right_on": join.get("right_on"), "right_label": join.get("right_label")}
    return cleaned

def compute_chart_from_spec(chart_spec: dict, tables: dict):
    try:
        cs = _validate_chart_spec(chart_spec)
        if not cs:
            return None
        table_name = cs["table"]
        if table_name not in tables or tables[table_name] is None:
            return None
        df = tables[table_name].copy()
        if "join" in cs:
            js = cs["join"]
            if js.get("table") in tables and tables[js.get("table")] is not None:
                right_df = tables[js["table"]][[js["right_on"], js["right_label"]]].drop_duplicates()
                df = df.merge(right_df, left_on=js["left_on"], right_on=js["right_on"], how="left")
        xcol = cs["x"]; ycol = cs.get("y"); agg = cs["agg"]; top_k = cs["top_k"]
        if agg == "count":
            grouped = df.groupby(xcol).size().rename("y").reset_index().sort_values("y", ascending=False).head(top_k)
        elif agg == "sum" and ycol:
            grouped = df.groupby(xcol)[ycol].sum().rename("y").reset_index().sort_values("y", ascending=False).head(top_k)
        elif agg == "avg" and ycol:
            grouped = df.groupby(xcol)[ycol].mean().rename("y").reset_index().sort_values("y", ascending=False).head(top_k)
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
    if chart_type == "bar":
        fig = px.bar(df, x="x", y="y", title=title)
    elif chart_type == "line":
        fig = px.line(df, x="x", y="y", title=title)
    elif chart_type == "pie":
        fig = px.pie(df, names="x", values="y", title=title)
    elif chart_type == "treemap":
        fig = px.treemap(df, path=["x"], values="y", title=title)
    else:
        fig = px.bar(df, x="x", y="y", title=title)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Minimal robust HF chat wrapper (use the working pattern) ----------
def hf_chat_wrapper(hf_token: str, model_id: str, user_question: str, short_context: Any = None, max_tokens: int = 400):
    SAFE = {"answer_text": "(LLM unavailable)", "chart_type":"none", "chart_spec":{}, "followups":[], "confidence":0.0, "raw_content":""}
    if not hf_token:
        return {**SAFE, "answer_text":"HF token missing — LLM disabled."}
    try:
        ctx = ""
        if short_context:
            try:
                ctx = "\n\nShort context:\n" + json.dumps(short_context, ensure_ascii=False, indent=2)
            except Exception:
                ctx = "\n\nShort context: (unserializable)"

        messages = [
            {"role":"system", "content": "You are a concise data assistant. Use ONLY the short context provided. Return a JSON object only with keys: answer_text, chart_type (bar|line|pie|none), chart_spec (object), followups (list), confidence (0.0-1.0)."},
            {"role":"user", "content": f"{user_question}{ctx}\n\nReturn JSON only."}
        ]

        client = InferenceClient(model=model_id, token=hf_token)
        resp = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.0)

        try:
            model_text = resp.choices[0].message.content
        except Exception:
            model_text = str(resp)

        # try parse JSON
        parsed = None
        try:
            parsed = json.loads(model_text)
        except Exception:
            m = re.search(r"\{(?:.|\n)*\}", model_text, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None

        if not parsed:
            # second chance: ask model to return strict JSON (assistant continuation)
            followup_req = "Return ONLY a JSON object now with keys answer_text, chart_type, chart_spec, followups, confidence."
            messages2 = messages + [{"role":"assistant","content": model_text}, {"role":"user","content": followup_req}]
            try:
                resp2 = client.chat_completion(messages=messages2, max_tokens=250, temperature=0.0)
                try:
                    model_text2 = resp2.choices[0].message.content
                except Exception:
                    model_text2 = str(resp2)
                m2 = re.search(r"\{(?:.|\n)*\}", model_text2, flags=re.DOTALL)
                if m2:
                    parsed = json.loads(m2.group(0))
                    model_text = model_text2
            except Exception:
                parsed = None

        if not parsed:
            return {**SAFE, "answer_text": model_text.strip()[:1500], "raw_content": model_text[:2000], "confidence":0.4}

        ans = parsed.get("answer_text") or parsed.get("answer") or parsed.get("text") or ""
        chart_type = parsed.get("chart_type", "none")
        chart_spec = parsed.get("chart_spec", {}) or parsed.get("chart", {}) or {}
        followups = parsed.get("followups", []) or []
        confidence = float(parsed.get("confidence", 0.9 if ans else 0.5))

        return {
            "answer_text": str(ans).strip(),
            "chart_type": chart_type if chart_type in {"bar","line","pie","none"} else "none",
            "chart_spec": chart_spec if isinstance(chart_spec, dict) else {},
            "followups": followups if isinstance(followups, list) else [],
            "confidence": max(0.0, min(1.0, float(confidence))),
            "raw_content": (model_text[:2000] if isinstance(model_text, str) else str(model_text)[:2000])
        }

    except Exception as e:
        return {**SAFE, "answer_text": f"LLM call failed: {repr(e)}", "raw_content": repr(e)[:2000]}

# ---------------- Streamlit UI ----------------
st.title("RAG-DB — Instacart experiment (local + optional LLM)")

st.sidebar.header("Options / Settings")
sample_nrows_prior = st.sidebar.number_input("Rows to load from order_products__prior (0 = full)", min_value=0, value=50000, step=10000)
mode = st.sidebar.selectbox("Mode", ["local-only","with-llm"])
hf_token_input = st.sidebar.text_input("Hugging Face API key (optional)", value=os.getenv("HF_API_KEY",""), type="password")
hf_model_input = st.sidebar.text_input("HF model id (chat-capable)", value="meta-llama/Meta-Llama-3-8B-Instruct")
st.sidebar.markdown("---")
st.sidebar.write("Notes: local-only runs exact queries. with-llm asks HF to suggest friendly text & chart_spec; chart computed locally.")

q = st.text_area("Ask a question", value="Which products appear most frequently in prior orders?", height=150)
run = st.button("Run")

@st.cache_data(ttl=3600)
def load_tables(nrows_prior: Optional[int] = None):
    products = load_csv_if_exists("products")
    aisles = load_csv_if_exists("aisles")
    departments = load_csv_if_exists("departments")
    orders = load_csv_if_exists("orders")
    prior = load_csv_if_exists("order_products__prior", nrows=nrows_prior)
    return {"products":products,"aisles":aisles,"departments":departments,"orders":orders,"prior":prior}

if run:
    # optional backend message
    if not backend_is_healthy():
        st.warning("Backend (FastAPI) not detected at :8000 — app will work in self-contained local mode.")
    st.info("Loading CSVs...")
    nrows = None if sample_nrows_prior == 0 else int(sample_nrows_prior)
    tables = load_tables(nrows_prior=nrows)
    products = tables["products"]; aisles = tables["aisles"]; departments = tables["departments"]; orders = tables["orders"]; prior = tables["prior"]
    if prior is None or products is None:
        st.error("Missing CSVs. Put products.csv and order_products__prior.csv in data/data/instacart/")
        st.stop()

    ql_raw = str(q).strip()
    intent = detect_intent(ql_raw)
    st.markdown(f"### Detected intent: **{intent}**")

    # small short context
    short_context = {}
    try:
        short_context["rows_in_prior"] = int(len(prior))
        sc_top5 = top_products_prior(prior, products, top_k=5)
        short_context["top5"] = sc_top5.to_dict(orient="records")
    except Exception as e:
        short_context["error"] = str(e)

    use_llm = (mode == "with-llm") and hf_token_input.strip() != "" and HF_HUB_AVAILABLE
    tables_map = {"prior":prior,"products":products,"orders":orders,"aisles":aisles,"departments":departments}

    # -------- Aggregation --------
    if intent == "aggregation":
        ql = ql_raw.lower()
        if any(x in ql for x in ["most frequently","most frequent","top products","most ordered","appear most frequently"]):
            df = top_products_prior(prior, products, top_k=15)
            st.subheader("Top products (exact counts)")
            st.dataframe(df.rename(columns={"x":"product_name","y":"count"}).head(15))
            render_chart(df, chart_type="bar", title="Top products (count)")
            with st.expander("More visualizations"):
                render_chart(df, chart_type="treemap", title="Top products (treemap)")
                render_chart(df, chart_type="pie", title="Top products (pie)")
        elif any(x in ql for x in ["least frequently","least ordered"]):
            df = least_products_prior(prior, products, top_k=15)
            st.subheader("Least frequently ordered (sample)")
            st.dataframe(df.rename(columns={"x":"product_name","y":"count"}))
            render_chart(df, chart_type="bar", title="Least frequently ordered")
        elif "average number of orders per product" in ql or "avg orders per product" in ql:
            avg = avg_orders_per_product(prior)
            st.write(f"**Average occurrences per product:** {avg:.2f}")
        elif "reorder ratio" in ql or "reorder rate" in ql:
            rr = avg_reorder_ratio(prior)
            if rr is not None:
                st.write(f"Average reorder ratio across products: **{rr:.3f}**")
            else:
                st.info("No 'reordered' column present in this dataset sample.")
        elif "day" in ql or "day of week" in ql:
            df = orders_by_day_of_week(orders)
            if df is not None:
                st.subheader("Orders by day of week")
                st.dataframe(df.rename(columns={"x":"day","y":"count"}))
                render_chart(df, chart_type="bar", title="Orders by day")
            else:
                st.info("orders.csv missing or 'order_dow' not present.")
        else:
            # fallback: local summary + optional LLM suggestion
            local_df = top_products_prior(prior, products, top_k=10)
            local_answer = f"Top product: {local_df.iloc[0]['x']} with {int(local_df.iloc[0]['y'])} occurrences."
            if use_llm:
                with st.spinner("Asking LLM for a friendly summary and suggested chart..."):
                    jobj = hf_chat_wrapper(hf_token_input.strip(), hf_model_input.strip(), ql_raw, short_context)
                    # Debug compact
                    st.markdown("**LLM (compact debug)**")
                    st.write("Confidence:", jobj.get("confidence"))
                    st.write("Raw preview:", jobj.get("raw_content","")[:600])
                    # If LLM returned parsed chart_spec+answer, show and compute
                    st.subheader("LLM answer (suggested)")
                    st.write(jobj.get("answer_text","(no LLM answer)"))
                    cs = jobj.get("chart_spec", {})
                    ct = jobj.get("chart_type", "bar")
                    if cs:
                        chart_df = compute_chart_from_spec(cs, tables_map)
                        if chart_df is not None:
                            render_chart(chart_df, chart_type=ct, title="LLM suggested chart (computed exactly)")
                        else:
                            st.warning("LLM suggested chart_spec cannot be computed locally; preview:")
                            st.json(cs)
                    # followups
                    if jobj.get("followups"):
                        st.markdown("**Follow-ups:**")
                        for f in jobj["followups"][:5]:
                            st.write("-", f)
            else:
                st.info("LLM disabled — showing local fallback.")
                st.write(local_answer)
                render_chart(local_df, chart_type="bar", title="Local fallback: top products")

    # -------- Retrieval --------
    else:
        st.subheader("Retrieval / Keyword search")
        proc = ql_raw
        prod_hits = fuzzy_search_products(products, proc, top_k=50)
        if prod_hits:
            st.write(f"Products matching query (top {len(prod_hits)}):")
            df = pd.DataFrame(prod_hits)[["product_id","product_name","score"]]
            st.dataframe(df)
            # Build short_context with retrieved examples to pass to LLM
            retrieved_names = [r["product_name"] for r in prod_hits[:20]]
            sc_for_llm = dict(short_context) if isinstance(short_context, dict) else {"rows_in_prior": short_context}
            sc_for_llm["retrieved_examples"] = retrieved_names[:10]

            # LLM summary (defensive)
            if use_llm:
                with st.spinner("Asking LLM to summarize retrieved results..."):
                    jobj = hf_chat_wrapper(hf_token_input.strip(), hf_model_input.strip(), f"Summarize these retrieved examples for the user: {proc}", sc_for_llm, max_tokens=220)
                    st.markdown("**LLM (compact debug)**")
                    st.write("Confidence:", jobj.get("confidence"))
                    st.write("Raw preview:", jobj.get("raw_content","")[:600])

                    llm_text = (jobj.get("answer_text") or "").strip()
                    # defensive override: if LLM says "no results" but we found items, prefer local
                    lower = llm_text.lower()
                    negative_phrases = ["no items", "no results", "nothing found", "no matches", "none found"]
                    contradicted = any(phrase in lower for phrase in negative_phrases) and len(prod_hits) > 0

                    if contradicted:
                        st.warning("LLM summary contradicted local retrieval — preferring local facts.")
                        top_names = retrieved_names[:8]
                        fallback = f"Found {len(prod_hits)} matching product(s). Top examples: {', '.join(top_names)}."
                        st.subheader("Accurate summary (from local retrieval)")
                        st.write(fallback)
                        st.dataframe(pd.DataFrame([{"product_name":n} for n in top_names]))
                    else:
                        st.subheader("LLM summary (optional)")
                        st.write(llm_text if llm_text else "(LLM returned no summary)")
                        if jobj.get("followups"):
                            st.markdown("**Follow-up suggestions:**")
                            for f in jobj["followups"][:5]:
                                st.write("-", f)
        else:
            st.info("No product fuzzy match found. Trying aisles substring...")
            if aisles is not None:
                mask = aisles["aisle"].str.lower().str.contains(proc.lower(), na=False)
                if mask.any():
                    st.write("Matching aisles (examples):")
                    st.dataframe(aisles[mask].head(30))
                else:
                    st.warning("No matches. Try simpler keywords like 'frozen', 'snacks', 'produce'.")
            else:
                st.warning("Aisles data not available.")

    st.success("Done.")