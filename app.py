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
# Web scraping helper for NLI / evidence
# =========================

@st.cache_data(show_spinner=False)
def fetch_web_context(claim: str) -> str:
    """
    Search the web for the claim and return cleaned article text.
    Tries DuckDuckGo news search first, then text search.
    Falls back to search snippets if scraping the page fails.
    Returns '' on total failure.
    """
    try:
        url = None
        snippet_parts = []

        with DDGS() as ddgs:
            # 1) Try news search
            try:
                news_results = ddgs.news(claim, max_results=3)
                for r in news_results:
                    url = r.get("url") or r.get("href")
                    title = r.get("title") or ""
                    body = r.get("body") or ""
                    if title:
                        snippet_parts.append(title)
                    if body:
                        snippet_parts.append(body)
                    if url:
                        break
            except Exception as e:
                print("DDG news search error:", e)

            # 2) Fallback: normal web search
            if not url:
                try:
                    text_results = ddgs.text(claim, max_results=3)
                    for r in text_results:
                        url = r.get("href") or r.get("url")
                        title = r.get("title") or ""
                        body = r.get("body") or ""
                        if title:
                            snippet_parts.append(title)
                        if body:
                            snippet_parts.append(body)
                        if url:
                            break
                except Exception as e:
                    print("DDG text search error:", e)

        # If we still have no URL but have snippets, use snippets only
        if not url and snippet_parts:
            return ("\n".join(snippet_parts))[:4000]

        if not url:
            return ""

        # 3) Try to download and parse the page
        try:
            resp = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = [
                p.get_text(" ", strip=True)
                for p in soup.find_all("p")
                if p.get_text(strip=True)
            ]
            if paragraphs:
                content = "\n".join(paragraphs)
                return content[:4000]
        except Exception as e:
            print("Article scrape error:", e)

        # 4) If scraping failed but we have snippets, use them
        if snippet_parts:
            return ("\n".join(snippet_parts))[:4000]

        return ""

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
    Returns (relation, score, label, explanation) or None on error.
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
            relation = "support"
            explanation = "Evidence in the article semantically supports the claim."
        elif "CONTRAD" in label:
            relation = "contradict"
            explanation = "Evidence in the article semantically contradicts the claim."
        else:
            relation = "neutral"
            explanation = "Evidence is neutral or unrelated to the claim."

        return relation, score, label, explanation

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

    st.title("🕵️‍♂️ FactGuru – Fake News Detection")
    st.caption(
        "Combines pattern analysis, web evidence and optional NLI semantic reasoning."
    )
    st.markdown("---")

    # --------- Layout: left = input & logs, right = results ----------
    col_left, col_right = st.columns([1.2, 1.8])

    with col_left:
        st.subheader("Claim input")
        claim = st.text_area(
            "Enter claim to verify",
            placeholder="e.g., Russia is part of USA",
            height=100,
        )

        manual_context = st.text_area(
            "Optional: paste news article / content (if you don't want auto-fetch)",
            placeholder="Paste article text here, or leave blank to auto-fetch from web...",
            height=120,
        )

        auto_fetch = st.checkbox(
            "🌐 Automatically fetch an article from the web",
            value=True,
            help="If checked and no manual content is provided, the app will search the web "
                 "for an article related to the claim.",
        )

        nli_model = load_nli_model()
        nli_available = nli_model is not None

        enable_nli = st.checkbox(
            "🧠 Enable NLI (semantic check using article text)",
            value=True,
            disabled=not nli_available,
            help="NLI is more expensive but gives a deeper semantic judgement using article text.",
        )

        if not nli_available:
            st.info(
                "NLI model is not available on this environment "
                "(ENABLE_NLI env variable is not set to 'true' or model failed to load). "
                "Pattern analysis will still work."
            )

        analyze = st.button("Analyze")

        st.markdown("### 🔎 Analysis progress & logs")
        progress_bar = st.progress(0)
        log_box = st.empty()

    with col_right:
        st.subheader("🧾 Result overview")
        verdict_placeholder = st.empty()
        stats_placeholder = st.empty()
        evidence_placeholder = st.empty()
        nli_placeholder = st.empty()

    # Load pattern model once
    pattern_model, vectorizer = load_pattern_model()

    if analyze:
        logs = []

        def log(msg):
            logs.append(msg)
            # Render logs as a small console-style area
            log_text = "\n".join(f"- {m}" for m in logs[-12:])
            log_box.markdown(f"```text\n{log_text}\n```")

        if not claim or len(claim.strip()) < 5:
            st.error("Claim must be at least 5 characters long.")
            return

        start_time = time.time()
        progress_bar.progress(5)
        log("Starting analysis pipeline...")

        # ----- Step 1: Pattern analysis -----
        log("Running pattern-based classifier...")
        proba = pattern_predict(pattern_model, vectorizer, claim)
        progress_bar.progress(30)

        if len(proba) == 2:
            real_prob = float(proba[0])
            fake_prob = float(proba[1])
        else:
            real_prob = fake_prob = 0.5

        if fake_prob >= 0.6:
            verdict_text = "Likely FAKE ❌"
            verdict_color = "❌"
        elif real_prob >= 0.6:
            verdict_text = "Likely REAL ✅"
            verdict_color = "✅"
        else:
            verdict_text = "UNCERTAIN / MIXED ⚠️"
            verdict_color = "⚠️"

        log(f"Pattern model verdict: {verdict_text}")
        progress_bar.progress(45)

        # ----- Step 2: Get evidence text -----
        context_to_use = None
        evidence_source = ""

        if manual_context.strip():
            context_to_use = manual_context.strip()
            evidence_source = "User-provided article/content."
            log("Using user-provided article/content for evidence.")
        elif auto_fetch:
            log("Fetching article from the web for evidence...")
            context_to_use = fetch_web_context(claim)
            if context_to_use:
                evidence_source = "Auto-fetched using DuckDuckGo search."
                log("Successfully fetched web article/snippet for evidence.")
            else:
                evidence_source = ""
                log("Could not fetch a useful article. Evidence will be limited.")
        else:
            log("No article text provided and auto-fetch disabled.")
            context_to_use = ""
            evidence_source = ""

        progress_bar.progress(60)

        # ----- Step 3: Optional NLI -----
        nli_result = None
        if enable_nli and nli_available and context_to_use:
            log("Running NLI semantic check between claim and article...")
            nli_result = nli_relation(nli_model, claim, context_to_use)
            if nli_result is None:
                log("NLI failed or returned no result.")
            else:
                relation, score, label, explanation = nli_result
                log(f"NLI result: {label} (score {score:.3f}) – {relation}")
        else:
            if enable_nli and not nli_available:
                log("NLI requested, but NLI model is not available.")
            elif enable_nli and not context_to_use:
                log("NLI requested, but no article text available.")
            else:
                log("NLI disabled for this run.")

        progress_bar.progress(90)

        elapsed = time.time() - start_time
        log(f"Finished analysis in {elapsed:.2f} seconds.")
        progress_bar.progress(100)

        # ---------- Render right side cards ----------

        # Verdict + probabilities
        verdict_placeholder.markdown(
            f"### 🎯 Final Verdict\n"
            f"**{verdict_text}**"
        )

        stats_placeholder.markdown(
            f"""**Pattern model probabilities**  
- Fake probability: `{fake_prob:.3f}`  
- Real probability: `{real_prob:.3f}`  
- Processing time: `{elapsed:.3f}` seconds"""
        )

        # Evidence section
        if context_to_use:
            snippet = context_to_use[:600]
            evidence_md = (
                "### 📚 Evidence summary\n"
                f"**Source**: {evidence_source or 'N/A'}  \n\n"
                f"**Snippet:**\n\n"
                f"> {snippet.replace('\n', ' ')}{'...' if len(context_to_use) > 600 else ''}"
            )
        else:
            evidence_md = (
                "### 📚 Evidence summary\n"
                "_No article text available. You can paste content manually or enable auto-fetch._"
            )

        evidence_placeholder.markdown(evidence_md)

        # NLI section
        if nli_result:
            relation, score, label, explanation = nli_result

            if relation == "support":
                badge = "🟢 SUPPORT"
            elif relation == "contradict":
                badge = "🔴 CONTRADICT"
            else:
                badge = "🟡 NEUTRAL"

            nli_placeholder.markdown(
                f"""### 🧠 Semantic NLI (optional)

**NLI verdict:** {badge}  
**Raw label:** `{label}`  
**Confidence:** `{score:.3f}`  

{explanation}
"""
            )
        else:
            nli_placeholder.markdown(
                "### 🧠 Semantic NLI (optional)\n"
                "_NLI result not available for this run._"
            )


if __name__ == "__main__":
    main()
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
    Tries DuckDuckGo news search first, then text search.
    Falls back to search snippets if scraping the page fails.
    Returns '' on total failure.
    """
    try:
        url = None
        snippet_parts = []

        with DDGS() as ddgs:
            # 1) Try news search
            try:
                news_results = ddgs.news(claim, max_results=3)
                for r in news_results:
                    url = r.get("url") or r.get("href")
                    title = r.get("title") or ""
                    body = r.get("body") or ""
                    if title:
                        snippet_parts.append(title)
                    if body:
                        snippet_parts.append(body)
                    if url:
                        break
            except Exception as e:
                print("DDG news search error:", e)

            # 2) Fallback: normal web search
            if not url:
                try:
                    text_results = ddgs.text(claim, max_results=3)
                    for r in text_results:
                        url = r.get("href") or r.get("url")
                        title = r.get("title") or ""
                        body = r.get("body") or ""
                        if title:
                            snippet_parts.append(title)
                        if body:
                            snippet_parts.append(body)
                        if url:
                            break
                except Exception as e:
                    print("DDG text search error:", e)

        # If we still have no URL but have snippets, use snippets only
        if not url and snippet_parts:
            return ("\n".join(snippet_parts))[:4000]

        if not url:
            return ""

        # 3) Try to download and parse the page
        try:
            resp = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = [
                p.get_text(" ", strip=True)
                for p in soup.find_all("p")
                if p.get_text(strip=True)
            ]
            if paragraphs:
                content = "\n".join(paragraphs)
                return content[:4000]
        except Exception as e:
            print("Article scrape error:", e)

        # 4) If scraping failed but we have snippets, use them
        if snippet_parts:
            return ("\n".join(snippet_parts))[:4000]

        return ""

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
    st.caption(
        "Powered by: Pattern Analysis, Web Evidence & (optional) NLI Semantic Engine"
    )
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
