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
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def parse_cot(text: str) -> tuple:
    m = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text.strip()


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Multi-Model Runner", page_icon="⚡", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("API Key")
    api_key = st.text_input(
        "CometAPI Key",
        type="password",
        placeholder="sk-...",
    )
    st.header("Models")
    selected_models = [m for m in MODELS if st.checkbox(m, value=True)]
    if not api_key:
        st.warning("Enter your CometAPI key to get started.")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("⚡ Multi-Model Runner")

prompt = st.text_area("Prompt", height=200, placeholder="Enter your prompt here...")

run_btn = st.button(
    "Run",
    type="primary",
    disabled=not api_key or not selected_models or not prompt.strip(),
    use_container_width=True,
)

if run_btn:
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
                result = future.result()
            except Exception as exc:
                placeholders[model].error(f"Error: {exc}")
                continue

            thinking, answer = parse_cot(result)
            with placeholders[model].container():
                if thinking:
                    with st.expander("Chain of Thought"):
                        st.markdown(thinking)
                st.markdown("**Answer**")
                st.markdown(answer)
