import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from openai import OpenAI

COMETAPI_BASE_URL = "https://api.cometapi.com/v1"

MODELS = [
    "gemini-3.1-pro-preview",
    "claude-opus-4-7",
    "gpt-5.5",
    "doubao-seed-2-0-pro-260215",
    "qwen3.6-plus",
    "kimi-k2.6",
]

SYSTEM_PROMPT = (
    "Think through the problem carefully inside <thinking>...</thinking> tags. "
    "Then write your final answer after the closing tag."
)


def query_model(api_key: str, model: str, prompt: str) -> str:
    client = OpenAI(api_key=api_key, base_url=COMETAPI_BASE_URL)
    is_claude = "claude" in model.lower()
    kwargs = (
        {"extra_body": {"system": SYSTEM_PROMPT}}
        if is_claude
        else {}
    )
    messages = [{"role": "user", "content": prompt}] if is_claude else [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return resp.choices[0].message.content


def parse_cot(text: str) -> tuple:
    m = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text.strip()


def render_result(result):
    if result.get("error"):
        st.error(result["error"])
        return
    thinking, answer = parse_cot(result["text"])
    if thinking:
        with st.expander("Chain of Thought"):
            st.markdown(thinking)
    st.markdown("**Answer**")
    st.markdown(answer)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Multi-Model Runner", page_icon="⚡", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = {}
if "models_run" not in st.session_state:
    st.session_state.models_run = []

_secret_key = st.secrets.get("COMETAPI_KEY", "")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    if not _secret_key:
        st.header("API Key")
        api_key = st.text_input(
            "CometAPI Key",
            type="password",
            placeholder="sk-...",
        )
        if not api_key:
            st.warning("Enter your CometAPI key to get started.")
    else:
        api_key = _secret_key
    st.header("Models")
    selected_models = [m for m in MODELS if st.checkbox(m, value=True)]

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("⚡ Multi-Model Runner")

prompt = st.text_area("Prompt", height=200, placeholder="Enter your prompt here...")

btn_col1, btn_col2 = st.columns([5, 1])
with btn_col1:
    run_btn = st.button(
        "Run",
        type="primary",
        disabled=not api_key or not selected_models or not prompt.strip(),
        use_container_width=True,
    )
with btn_col2:
    reset_btn = st.button("Reset", use_container_width=True)

if reset_btn:
    st.session_state.results = {}
    st.session_state.models_run = []
    st.rerun()

if run_btn:
    st.session_state.results = {}
    st.session_state.models_run = list(selected_models)

    st.divider()
    cols_per_row = min(len(selected_models), 3)
    cols = st.columns(cols_per_row)

    placeholders = {}
    for i, model in enumerate(selected_models):
        with cols[i % cols_per_row]:
            st.subheader(model)
            placeholders[model] = st.empty()
            placeholders[model].status("Running...", state="running")

    with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
        futures = {
            executor.submit(query_model, api_key, model, prompt.strip()): model
            for model in selected_models
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                text = future.result()
                st.session_state.results[model] = {"text": text}
            except Exception as exc:
                st.session_state.results[model] = {"error": str(exc)}

            with placeholders[model].container():
                render_result(st.session_state.results[model])

elif st.session_state.results:
    st.divider()
    models_run = st.session_state.models_run
    cols_per_row = min(len(models_run), 3)
    cols = st.columns(cols_per_row)

    for i, model in enumerate(models_run):
        with cols[i % cols_per_row]:
            st.subheader(model)
            if model in st.session_state.results:
                render_result(st.session_state.results[model])
