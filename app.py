import os
import time
import traceback

import numpy as np
import joblib
import streamlit as st

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


# =========================
# Load pattern analysis model
# =========================

@st.cache_resource
def load_pattern_model():
    """
    Load Naive Bayes fake-news model and TF-IDF vectorizer
    from ml/pattern_analysis.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        base_dir,
        "ml",
        "pattern_analysis",
        "fake_news_model_improved.pkl",
    )
    vec_path = os.path.join(
        base_dir,
        "ml",
        "pattern_analysis",
        "tfidf_vectorizer_naivebayes_clickbait.pkl",
    )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


# =========================
# Optional: load NLI model (only if ENABLE_NLI="true")
# =========================

@st.cache_resource
def load_nli_model():
    """
    Try to load NLI model, but only when ENABLE_NLI="true" in env.
    On local Windows this usually stays disabled.
    On Streamlit Cloud (Linux) you can enable it via secrets.
    """
    use_nli = os.getenv("ENABLE_NLI", "").lower() == "true"
    if not use_nli:
        # NLI disabled – app still works with pattern model only
        return None

    try:
        from transformers import pipeline

        nli = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            max_length=512,
            truncation=True,
        )
        return nli
    except Exception as e:
        # If anything goes wrong, log and disable NLI
        print("NLI disabled due to error:", e)
        traceback.print_exc()
        return None


# =========================
# Web scraping helper for NLI
# =========================

@st.cache_data(show_spinner=False)
def fetch_web_context(claim: str) -> str:
    """
    Search the web for the claim and return cleaned article text.
    Uses DuckDuckGo search + simple HTML <p> extraction.
    Returns empty string on failure.
    """
    try:
        # 1) Search DuckDuckGo
        url = None
        with DDGS() as ddgs:
            results = ddgs.text(claim, max_results=3)
            for r in results:
                url = r.get("href") or r.get("url")
                if url:
                    break

        if not url:
            return ""

        # 2) Download the page
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        # 3) Extract paragraph text
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if p.get_text(strip=True)
        ]
        if not paragraphs:
            return ""

        content = "\n".join(paragraphs)

        # Limit size for NLI
        return content[:4000]

    except Exception as e:
        print("Web fetch error:", e)
        traceback.print_exc()
        return ""


# =========================
# Scoring helpers
# =========================

def pattern_predict(model, vectorizer, text: str) -> np.ndarray:
    """
    Run the pattern-based classifier.
    If anything goes wrong (feature mismatch, etc.), return 50/50
    so the app does not crash.
    Returns np.array([real_prob, fake_prob]).
    """
    try:
        X = vectorizer.transform([text])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            return np.array(proba, dtype=float)

        # Fallback if no predict_proba (unlikely)
        pred = model.predict(X)[0]
        if pred in [0, 1]:
            return np.array([1.0 - pred, float(pred)])
        return np.array([0.5, 0.5])

    except Exception as e:
        print("Pattern model error:", e)
        traceback.print_exc()
        # Neutral fallback: 50% fake, 50% real
        return np.array([0.5, 0.5])


def nli_relation(nli, claim: str, context: str):
    """
    Run NLI if model is available.
    Returns (relation, score, label) or None on error.
    relation in {"support","contradict","neutral"}.
    """
    if nli is None:
        return None

    try:
        result = nli({"text": context[:1000], "text_pair": claim})
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        label = str(result["label"]).upper()
        score = float(result["score"])

        if "ENTAIL" in label:
            return "support", score, label
        elif "CONTRAD" in label:
            return "contradict", score, label
        else:
            return "neutral", score, label

    except Exception as e:
        print("NLI error:", e)
        traceback.print_exc()
        return None


# =========================
# Streamlit UI
# =========================

def main():
    st.set_page_config(
        page_title="FactGuru – Fake News Detection",
        page_icon="🕵️‍♂️",
        layout="wide",
    )

    st.title("🕵️‍♂️ Enhanced Fact Verification System")
    st.caption("Powered by: Pattern Analysis, Web Evidence & (optional) NLI Semantic Engine")
    st.markdown("---")

    claim = st.text_input(
        "📝 Enter claim to verify",
        placeholder="e.g., Russia is part of USA",
    )

    context = st.text_area(
        "📄 Optional: paste article/content (used for semantic NLI check)",
        placeholder="Paste news article text or evidence here (optional)...",
        height=180,
    )

    auto_fetch = st.checkbox(
        "🌐 Automatically fetch an article from the web for NLI",
        value=True,
    )

    col1, _ = st.columns([1, 3])
    with col1:
        analyze = st.button("Analyze")

    # Load models once (cached)
    with st.spinner("Loading models..."):
        pattern_model, vectorizer = load_pattern_model()
        nli_model = load_nli_model()

    if analyze:
        if not claim or len(claim.strip()) < 5:
            st.error("Claim must be at least 5 characters long.")
            return

        start = time.time()

        proba = pattern_predict(pattern_model, vectorizer, claim)
        if len(proba) == 2:
            real_prob = float(proba[0])
            fake_prob = float(proba[1])
        else:
            real_prob = fake_prob = 0.5

        if fake_prob >= 0.6:
            verdict = "Likely **FAKE** ❌"
        elif real_prob >= 0.6:
            verdict = "Likely **REAL** ✅"
        else:
            verdict = "**UNCERTAIN / MIXED** ⚠️"

        end = time.time()

        # ---------- Pattern result ----------
        st.markdown("### 🎯 Result (Pattern Analysis)")
        st.markdown(verdict)
        st.progress(min(max(fake_prob, 0.0), 1.0))
        st.write(f"Fake probability: `{fake_prob:.3f}`")
        st.write(f"Real probability: `{real_prob:.3f}`")
        st.write(f"Processing time: `{end - start:.3f}` seconds")

        st.markdown("---")
        st.markdown("### 🧠 Semantic NLI (optional)")

        if nli_model is None:
            st.info(
                "NLI model is **disabled** on this environment. "
                "Set `ENABLE_NLI=\"true\"` in Streamlit Cloud secrets to enable it."
            )
            return

        # Decide which context to use: user text or auto-fetched
        if context.strip():
            context_to_use = context
            st.write("Using **user-provided** article/content for NLI.")
        elif auto_fetch:
            with st.spinner("Fetching article from the web..."):
                auto_context = fetch_web_context(claim)

            if not auto_context:
                st.info(
                    "Could not automatically fetch a useful article. "
                    "You can paste news content manually above."
                )
                return

            context_to_use = auto_context
            st.markdown("#### 🌐 Auto-fetched article snippet")
            snippet = context_to_use[:1000]
            st.write(snippet + ("..." if len(context_to_use) > 1000 else ""))
        else:
            st.info("Provide article text above or enable auto-fetching to run NLI.")
            return

        # Run NLI
        with st.spinner("Running NLI semantic check..."):
            rel = nli_relation(nli_model, claim, context_to_use)

        if rel is None:
            st.warning("Failed to run NLI check.")
        else:
            relation, score, label = rel
            st.write(f"NLI label: `{label}` (score: `{score:.3f}`)")
            if relation == "support":
                st.success("The content **SUPPORTS** the claim.")
            elif relation == "contradict":
                st.error("The content **CONTRADICTS** the claim.")
            else:
                st.info("The content is **NEUTRAL / IRRELEVANT** to the claim.")


if __name__ == "__main__":
    main()
