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


def query_model(api_key: str, model: str, prompt: str, temperature: float | None = None) -> str:
    client = OpenAI(api_key=api_key, base_url=COMETAPI_BASE_URL)
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        **kwargs,
    )
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
if "jobs_run" not in st.session_state:
    st.session_state.jobs_run = []

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
    st.header("Runs")
    n_runs = st.slider("Parallel runs per model", min_value=1, max_value=8, value=1)
    st.header("Temperature")
    custom_temp = st.checkbox("Custom temperature")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01, disabled=not custom_temp) if custom_temp else None

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
    st.session_state.jobs_run = []
    st.rerun()

if run_btn:
    st.session_state.results = {}
    # jobs_run is a list of (model, run_index) tuples
    jobs = [(model, run_idx) for model in selected_models for run_idx in range(1, n_runs + 1)]
    st.session_state.jobs_run = jobs

    st.divider()
    cols_per_row = min(len(jobs), 3)
    cols = st.columns(cols_per_row)

    placeholders = {}
    for i, (model, run_idx) in enumerate(jobs):
        with cols[i % cols_per_row]:
            label = f"{model} — run {run_idx}" if n_runs > 1 else model
            st.subheader(label)
            key = (model, run_idx)
            placeholders[key] = st.empty()
            placeholders[key].status("Running...", state="running")

    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(query_model, api_key, model, prompt.strip(), temperature): (model, run_idx)
            for model, run_idx in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                text = future.result()
                st.session_state.results[key] = {"text": text}
            except Exception as exc:
                st.session_state.results[key] = {"error": str(exc)}

            with placeholders[key].container():
                render_result(st.session_state.results[key])

elif st.session_state.jobs_run:
    st.divider()
    jobs = st.session_state.jobs_run
    n_runs_stored = max(run_idx for _, run_idx in jobs)
    cols_per_row = min(len(jobs), 3)
    cols = st.columns(cols_per_row)

    for i, (model, run_idx) in enumerate(jobs):
        with cols[i % cols_per_row]:
            label = f"{model} — run {run_idx}" if n_runs_stored > 1 else model
            st.subheader(label)
            key = (model, run_idx)
            if key in st.session_state.results:
                render_result(st.session_state.results[key])
