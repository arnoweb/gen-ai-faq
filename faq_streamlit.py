import streamlit as st
import streamlit.components.v1 as components
from streamlit_searchbox import st_searchbox
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import os
import requests
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).resolve().parent / "docs"

st.markdown(
    """
    <style>
    .hero-blob { position: fixed; z-index: -1; border-radius: 50%; filter: blur(70px); opacity: 0.25; pointer-events: none; }
    .blob-a { width: 320px; height: 320px; background: #4F46E5; top: -80px; left: -60px; }
    .blob-b { width: 280px; height: 280px; background: #22D3EE; top: -40px; right: -60px; }

    h1 { color: #4F46E5 !important; font-weight: 800 !important; letter-spacing: -0.02em; }

    .trust-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.4rem 0 1rem; }
    .trust-pill {
        display: inline-flex; align-items: center; gap: 0.35rem;
        background: #F0FDF4; color: #14532D; border: 1px solid #4ADE80;
        border-radius: 100px; padding: 0.3rem 0.75rem; font-size: 0.82rem; font-weight: 600;
    }

    .stButton > button {
        background: #4F46E5 !important; color: #fff !important; border: none !important;
        border-radius: 100px !important; padding: 0.5rem 1.4rem !important; font-weight: 600 !important;
        transition: opacity 0.15s ease;
    }
    .stButton > button:hover { opacity: 0.88; }

    div[data-testid="stTextInput"] input {
        border: 2px solid #4F46E5 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1.05rem !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.08);
    }
    div[data-testid="stTextInput"] input:focus {
        box-shadow: 0 0 0 3px rgba(79,70,229,0.18) !important;
    }
    div[data-testid="stTextInput"] label { font-weight: 600 !important; }

    .result-tight {
        background: #FFFFFF; border: 1px solid #DDE1E8; border-radius: 12px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; box-shadow: 0 1px 2px rgba(15,23,42,0.06);
    }
    .result-tight p { margin: 0 0 4px 0; line-height: 1.25; }
    .result-tight .sim {
        display: inline-block; margin-top: 6px; color: #14532D; background: #F0FDF4;
        border: 1px solid #4ADE80; border-radius: 100px; padding: 2px 10px;
        font-size: 0.78rem; font-weight: 600;
    }
    hr { margin: 2px 0; }
    </style>
    <div class="hero-blob blob-a"></div>
    <div class="hero-blob blob-b"></div>
    """,
    unsafe_allow_html=True,
)

try:
    _secret_hf_key = st.secrets.get("HF_API_KEY", "")
except Exception:
    _secret_hf_key = ""
HF_API_KEY = os.getenv("HF_API_KEY") or _secret_hf_key
os.environ.setdefault("HF_TOKEN", HF_API_KEY or "")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"

model_path = "arnoweb/model-faq-sentence-autotrain"
reranker_path = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# File paths for each language
faq_data_paths = {
    "English": "data/faq_source_en.jsonl",
    "Français": "data/faq_source_fr.jsonl"
}

# ONNX Runtime backend: same weights, ~3-6x faster than PyTorch on CPU (verified locally),
# with cosine similarity to the PyTorch output ~1.0 (no meaningful precision loss).
ONNX_MODEL_KWARGS = {"provider": "CPUExecutionProvider"}

@st.cache_resource
def load_model():
    # int8-quantized export (pushed alongside the fp32 one) instead of the default
    # onnx/model.onnx: ~265MB / ~30% less RAM than the 1.0GB fp32 file, cosine similarity
    # to the fp32 embeddings ~0.96-0.997 on real FAQ questions — no meaningful ranking impact.
    return SentenceTransformer(
        model_path,
        backend="onnx",
        model_kwargs={**ONNX_MODEL_KWARGS, "file_name": "onnx/model_quantized.onnx"},
    )

@st.cache_resource
def load_reranker():
    # int8-quantized export (already published on the Hub) instead of the default fp32
    # onnx/model.onnx: ~113MB / ~4x less RAM than the 449MB fp32 file, same architecture.
    return CrossEncoder(
        reranker_path,
        backend="onnx",
        model_kwargs={**ONNX_MODEL_KWARGS, "file_name": "onnx/model_quint8_avx2.onnx"},
    )

@st.cache_data(ttl=300)
def load_faq_data(faq_data_path):
    faq_items = []
    with open(faq_data_path, "r") as f:
        for line in f:
            if line.strip():
                faq_items.append(json.loads(line))
    faq_questions = [item["query"] for item in faq_items]
    faq_answers = [item["answer"] for item in faq_items]
    return faq_questions, faq_answers

@st.cache_data(ttl=300)
def compute_question_embeddings(_model, faq_questions):
    return _model.encode(faq_questions, convert_to_tensor=True)

def build_rag_prompt(user_question, retrieved_faqs, retrieved_questions, language="fr"):
    if language == "Français":
        intro = "Voici des extraits de notre FAQ :\n"
        docs = "\n\n".join([
            f"Document {i+1} :\nQuestion : {q}\nRéponse : {a}" for i, (q, a) in enumerate(zip(retrieved_questions, retrieved_faqs))
        ])
        question = f"\n\nQuestion utilisateur : {user_question}\n"
        instruction = (
            "En t'appuyant strictement sur les documents ci-dessus, rédige une réponse claire et détaillée (3 à 5 phrases) à la question de l'utilisateur. "
            "Si l'information n'est pas présente dans les documents, réponds simplement : 'Je ne sais pas.' Ne fais pas d'hypothèses et n'invente pas de réponses."
        )
    else:
        intro = "Here are some excerpts from our FAQ:\n"
        docs = "\n\n".join([
            f"Document {i+1}:\nQuestion: {q}\nAnswer: {a}" for i, (q, a) in enumerate(zip(retrieved_questions, retrieved_faqs))
        ])
        question = f"\n\nUser question: {user_question}\n"
        instruction = (
            "Based strictly on the above documents, write a clear, moderately detailed answer (3–5 sentences) to the user's question. "
            "If the information is not present in the documents, simply answer: 'I don't know.' Do not make assumptions or invent answers."
        )
    return f"{intro}{docs}{question}{instruction}"


def query_local_llm(prompt, model="qwen/qwen3-vl-4b"):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }
    response = requests.post("http://localhost:1234/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def query_remote_llm(prompt, model: str = HF_MODEL_DEFAULT, temperature: float = 0.2, max_tokens: int = 800):
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY is not set in the environment.")

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Hugging Face API error: {response.text}") from exc

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response format from Hugging Face: {data}") from exc



st.title("Assistant FAQ IA")
st.caption("Un moteur de recherche sémantique qui comprend vraiment vos questions, pas seulement vos mots-clés.")

# Language switcher
language = st.radio("Select language / Choisissez la langue:", ["Français", "English"])

st.markdown(
    f"""
    <div class="trust-pills">
      <span class="trust-pill">✓ {"No data sent to a third party for search" if language == "English" else "Aucune donnée envoyée à un tiers pour la recherche"}</span>
      <span class="trust-pill">✓ {"Refuses to answer rather than invent" if language == "English" else "Refuse de répondre plutôt que d'inventer"}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

faq_data_path = faq_data_paths[language]
if not HF_API_KEY:
    st.error(
        "Missing HF_API_KEY: the fine-tuned model repo is private and no Hugging Face "
        "token was found (checked env var HF_API_KEY and Streamlit secrets). On Streamlit "
        "Cloud, add HF_API_KEY under this app's Settings → Secrets, then save to trigger a reboot."
    )
    st.stop()
model = load_model()
reranker = load_reranker()
faq_questions, faq_answers = load_faq_data(faq_data_path)
question_embeddings = compute_question_embeddings(model, faq_questions)

st.write(
    "Start typing your question — matching FAQ entries appear as you type."
    if language == "English"
    else "Commencez à taper votre question — les résultats apparaissent au fil de la frappe."
)

placeholder_text = (
    "e.g., I still haven't received my package, what should I do?"
    if language == "English"
    else "ex. Je n'ai toujours pas reçu mon colis, que faire ?"
)

TOP_K = 3
RERANK_POOL = 8  # bi-encoder candidates handed to the cross-encoder for reranking
RERANK_SKIP_THRESHOLD = 0.9  # top bi-encoder score above which reranking is skipped
SIMILARITY_THRESHOLD = 0.3  # You can adjust this value
MIN_QUERY_WORDS = 3  # avoid matching on ambiguous short fragments (e.g. "je n'ai")


@st.cache_data(ttl=300)
def retrieve_top_k(query: str, faq_data_path: str, k: int = TOP_K):
    # A FAQ sees the same questions over and over — caching on (query, dataset) skips
    # re-encoding and reranking entirely for a repeat, at essentially no cost to add.
    model = load_model()
    faq_questions, faq_answers = load_faq_data(faq_data_path)
    question_embeddings = compute_question_embeddings(model, faq_questions)

    query_embedding = model.encode(query, convert_to_tensor=True)
    # Pool selection is keyed on the FAQ questions only. Matching against answers too
    # (an earlier version did) lets an unrelated FAQ outrank the right one whenever its
    # answer happens to share a word with the query (e.g. "colis" appearing in an
    # assembly-instructions answer) even though its question has nothing to do with it.
    similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]

    pool_size = min(RERANK_POOL, len(faq_questions))
    pool_indices = similarities.topk(pool_size).indices.tolist()

    top_score = similarities[pool_indices[0]].item()
    if top_score >= RERANK_SKIP_THRESHOLD:
        # Already unambiguous (near-exact match) — the cross-encoder wouldn't change
        # the ranking, so skip its 8 forward passes.
        indices = pool_indices[:k]
    else:
        # The cross-encoder still sees question+answer for extra context when
        # reranking, but only among candidates whose question already looked relevant.
        reranker = load_reranker()
        pairs = [(query, f"{faq_questions[i]} {faq_answers[i]}") for i in pool_indices]
        rerank_scores = reranker.predict(pairs)
        reranked = sorted(zip(pool_indices, rerank_scores), key=lambda item: -item[1])
        indices = [idx for idx, _ in reranked[:k]]

    return indices, similarities


def search_faq(searchterm: str):
    st.session_state["faq_last_query"] = searchterm
    if not searchterm or len(searchterm.split()) < MIN_QUERY_WORDS:
        return []
    indices, similarities = retrieve_top_k(searchterm, faq_data_path)
    return [
        (f"{faq_questions[idx]}  ·  {similarities[idx].item():.2f}", idx)
        for idx in indices
        if similarities[idx].item() >= SIMILARITY_THRESHOLD
    ]


searchbox_key = f"faq_searchbox_{language}"

selected_idx = st_searchbox(
    search_faq,
    placeholder=placeholder_text,
    label="Ask your question:" if language == "English" else "Posez votre question :",
    key=searchbox_key,
    # The library's own "No options" message can't be reworded, only hidden — we show
    # our own, more specific message below instead.
    style_overrides={"searchbox": {"optionEmpty": "hidden"}},
)

if selected_idx is None:
    current_search = st.session_state.get(searchbox_key, {}).get("search", "")
    has_options = bool(st.session_state.get(searchbox_key, {}).get("options_py"))

    if not current_search:
        st.info(
            "Start typing a question above to see matching FAQ entries."
            if language == "English"
            else "Commencez à taper une question ci-dessus pour voir les réponses correspondantes."
        )
    elif len(current_search.split()) < MIN_QUERY_WORDS:
        st.info(
            "Keep typing — a few more words will help find the right match."
            if language == "English"
            else "Continuez à taper — quelques mots de plus aideront à trouver la bonne réponse."
        )
    elif not has_options:
        st.info(
            "No matching question yet — try rephrasing."
            if language == "English"
            else "Aucune question ne correspond pour l'instant — essayez de reformuler."
        )
    else:
        st.info(
            "Pick a matching question above to see the full answer."
            if language == "English"
            else "Choisissez une question ci-dessus pour voir la réponse complète."
        )
else:
    last_query = st.session_state.get("faq_last_query") or faq_questions[selected_idx]
    context_indices, similarities = retrieve_top_k(last_query, faq_data_path)
    selected_score = similarities[selected_idx].item()
    retrieved_faqs = [faq_answers[idx] for idx in context_indices]
    retrieved_questions = [faq_questions[idx] for idx in context_indices]

    st.markdown(
        f"""
        <div class="result-tight">
            <p><strong>Q:</strong> {faq_questions[selected_idx]} <span class="sim">{selected_score:.2f}</span></p>
            <p>{faq_answers[selected_idx]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # RAG section
    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
    if st.button("Générer une réponse avec le LLM" if language == "Français" else "Generate answer with LLM", key="llm_button", help=None):
        prompt = build_rag_prompt(last_query, retrieved_faqs, retrieved_questions, language=language)
        with st.spinner("Call of the LLM..."):
            try:
                llm_response = query_remote_llm(prompt)
            except Exception as exc:
                st.error(f"LLM call failed: {exc}")
                llm_response = None
        if llm_response:
            low_conf = llm_response.strip().lower()
            if ("i don't know" in low_conf) or ("je ne sais pas" in low_conf):
                st.warning(
                    "Aucune réponse trouvée. Réessayez en reformulant votre question."
                    if language == "Français"
                    else "No answer found. Try rephrasing your question."
                )
            else:
                st.markdown("### Réponse générée :" if language == "Français" else "### Generated answer:")
                st.write(llm_response)

st.divider()
st.markdown(
    "Curious how much fine-tuning actually improves results? "
    "[Compare base vs fine-tuned side by side](https://arnoweb-rag-faq-compare-basevsfinetuned-huggingface.streamlit.app/)."
    if language == "English"
    else "Curieux de savoir ce que le fine-tuning change vraiment ? "
    "[Comparez avant/après fine-tuning côte à côte](https://arnoweb-rag-faq-compare-basevsfinetuned-huggingface.streamlit.app/)."
)

with st.expander("Business value & use cases" if language == "English" else "Valeur métier & cas d'usage"):
    bv_language = st.radio("Language / Langue", ["Français", "English"], horizontal=True, key="bv_lang_main")
    bv_file = "business-value-en.html" if bv_language == "English" else "business-value.html"
    components.html((DOCS_DIR / bv_file).read_text(encoding="utf-8"), height=6000, scrolling=True)

with st.expander("Technical architecture" if language == "English" else "Architecture technique"):
    components.html((DOCS_DIR / "architecture.html").read_text(encoding="utf-8"), height=6000, scrolling=True)

with st.expander("About this project" if language == "English" else "À propos de ce projet"):
    st.markdown(
        "**Who this is for:** e-commerce and SaaS teams with a bilingual (FR/EN) customer base "
        "who want FAQ search that understands real questions, not just keywords — without adding "
        "a recurring per-query AI bill.\n\n"
        "**Value:** fine-tuned semantic retrieval that finds the right answer even when the "
        "question is phrased differently, with a visible confidence score and an explicit "
        "\"I don't know\" instead of a guess. The search itself runs locally, at no ongoing cost.\n\n"
        "**Next steps:** connecting live FAQ content from a CMS (WordPress, Contentful, Strapi) "
        "instead of static files, a lightweight embeddable widget for any website, and a hosted "
        "API/SDK for teams who want to build their own UI."
    )

st.markdown(
    'Made by <a href="https://www.linkedin.com/in/bretonarnaud/" target="_blank">Arnaud BRETON</a>',
    unsafe_allow_html=True,
)
