import os
import time
import traceback

import streamlit as st
import joblib
import numpy as np


# ============ LOAD PATTERN MODEL (your existing fake news model) ============

@st.cache_resource
def load_pattern_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "ml", "pattern_analysis", "fake_news_model_improved.pkl")
    vec_path = os.path.join(base_dir, "ml", "pattern_analysis", "tfidf_vectorizer_naivebayes_clickbait.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


# ============ OPTIONAL: NLI MODEL (transformers, may fail locally) ============

@st.cache_resource
def load_nli_model():
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
        # On your Windows, this will likely fail because of torch DLL
        print("NLI disabled:", e)
        return None


def pattern_predict(model, vectorizer, text: str):
    X = vectorizer.transform([text])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        pred = model.predict(X)[0]
        proba = np.array([1.0 - pred, pred]) if pred in [0, 1] else np.array([0.5, 0.5])
    return proba


def nli_relation(nli, claim: str, context: str):
    if nli is None:
        return None

    try:
        result = nli({"text": context[:1000], "text_pair": claim})
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        label = result["label"].upper()
        score = float(result["score"])

        if "ENTAIL" in label:
            return ("support", score, label)
        elif "CONTRAD" in label:
            return ("contradict", score, label)
        else:
            return ("neutral", score, label)
    except Exception:
        traceback.print_exc()
        return None


def main():
    st.set_page_config(
        page_title="FactGuru – Fake News Detection",
        page_icon="🕵️‍♂️",
        layout="wide",
    )

    st.title("🕵️‍♂️ Enhanced Fact Verification System")
    st.caption("Powered by: Pattern Analysis, (optional) NLI Semantic Engine")
    st.markdown("---")

    claim = st.text_input("📝 Enter claim to verify", placeholder="e.g., COVID-19 vaccines cause infertility")
    context = st.text_area(
        "📄 Optional: paste article/content (used for semantic NLI check)",
        placeholder="Paste news article text or evidence here (optional)...",
        height=200,
    )

    col1, _ = st.columns([1, 3])
    with col1:
        analyze = st.button("Analyze")

    with st.spinner("Loading models..."):
        pattern_model, vectorizer = load_pattern_model()
        nli_model = load_nli_model()  # may be None on your PC

    if analyze:
        if not claim or len(claim.strip()) < 5:
            st.error("Claim must be at least 5 characters long.")
            return

        start = time.time()

        proba = pattern_predict(pattern_model, vectorizer, claim)
        fake_prob = float(proba[1])
        real_prob = float(proba[0])

        if fake_prob >= 0.6:
            verdict = "Likely **FAKE** ❌"
        elif real_prob >= 0.6:
            verdict = "Likely **REAL** ✅"
        else:
            verdict = "**UNCERTAIN / MIXED** ⚠️"

        end = time.time()

        st.markdown("### 🎯 Result (Pattern Analysis)")
        st.markdown(verdict)
        st.progress(fake_prob)
        st.write(f"Fake probability: `{fake_prob:.3f}`")
        st.write(f"Real probability: `{real_prob:.3f}`")
        st.write(f"Processing time: `{end - start:.3f}` seconds")

        st.markdown("---")
        st.markdown("### 🧠 Semantic NLI (optional)")

        if nli_model is None:
            st.info(
                "NLI model is **disabled** on this machine (PyTorch DLL issue). "
                "On a Linux server / Streamlit Cloud it may work without changes."
            )
        else:
            if context.strip():
                with st.spinner("Running NLI semantic check..."):
                    rel = nli_relation(nli_model, claim, context)
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
            else:
                st.info("Provide article/content text above to run NLI-based semantic check.")


if __name__ == "__main__":
    main()
