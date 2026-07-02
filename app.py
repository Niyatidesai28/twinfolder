import streamlit as st
import pandas as pd
import traceback

st.set_page_config(
    page_title="FounderTwin",
    page_icon="🚀",
    layout="wide"
)

try:
    from foundertwin_backend import (
        analyze_startup_idea_v3,
        explain_idea_llm_grounded,
        build_project_steps_llm_grounded,
        ask_foundertwin_llm_grounded,
    )
except RuntimeError as e:
    st.error(str(e))
    st.stop()

st.title("🚀 FounderTwin")
st.caption("Startup idea validation, competitor discovery, MVP planning, and grounded chatbot")

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_idea" not in st.session_state:
    st.session_state.current_idea = ""

if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

if "current_explanation" not in st.session_state:
    st.session_state.current_explanation = ""

if "current_project_steps" not in st.session_state:
    st.session_state.current_project_steps = ""


# =========================
# HELPERS
# =========================
def safe_get(dct, key, default=""):
    if isinstance(dct, dict):
        return dct.get(key, default)
    return default


def reset_chat():
    st.session_state.messages = []
    st.session_state.current_idea = ""
    st.session_state.current_analysis = None
    st.session_state.current_explanation = ""
    st.session_state.current_project_steps = ""


def render_analysis_summary(analysis):
    if not isinstance(analysis, dict):
        st.info("No structured analysis available yet.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Cluster", str(safe_get(analysis, "startup_cluster_name", "N/A")))
        st.metric("Best Score", str(safe_get(analysis, "best_score", "N/A")))

    with col2:
        st.metric("Market Signal", str(safe_get(analysis, "market_signal", "N/A")))
        st.metric("Avg Top Score", str(safe_get(analysis, "avg_top_score", "N/A")))

    with col3:
        st.metric("Cluster Quality", str(safe_get(analysis, "startup_cluster_quality", "N/A")))
        st.metric("Cluster Size", str(safe_get(analysis, "cluster_size", "N/A")))

    tags = safe_get(analysis, "query_tags", [])
    if tags:
        st.markdown("**Query Tags:** " + ", ".join(map(str, tags)))

    summary = safe_get(analysis, "summary", "")
    if summary:
        st.markdown("**Summary:**")
        st.write(summary)

    detailed_results = safe_get(analysis, "detailed_results", [])
    if isinstance(detailed_results, list) and len(detailed_results) > 0:
        cleaned_rows = []
        for row in detailed_results:
            if isinstance(row, dict):
                cleaned_rows.append({
                    "company_name": row.get("company_name", ""),
                    "similarity_score": row.get("similarity_score", ""),
                    "startup_cluster_name": row.get("startup_cluster_name", ""),
                    "tags": ", ".join(row.get("tags", [])) if isinstance(row.get("tags", []), list) else row.get("tags", ""),
                    "description": row.get("description", "")
                })

        if cleaned_rows:
            st.markdown("**Top Matches:**")
            st.dataframe(pd.DataFrame(cleaned_rows), use_container_width=True)


def initial_analysis_flow(user_idea, top_n):
    analysis = analyze_startup_idea_v3(user_idea, top_n=top_n)
    explanation_result = explain_idea_llm_grounded(user_idea, top_n=top_n)
    steps_result = build_project_steps_llm_grounded(user_idea, top_n=top_n)

    st.session_state.current_idea = user_idea
    st.session_state.current_analysis = analysis
    st.session_state.current_explanation = explanation_result.get("llm_explanation", "")
    st.session_state.current_project_steps = steps_result.get("project_steps", "")

    st.session_state.messages.append({
        "role": "user",
        "content": user_idea
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"""
### Idea Explanation
{st.session_state.current_explanation}

---

### MVP Build Plan
{st.session_state.current_project_steps}
"""
    })


def followup_chat_flow(question, top_n):
    current_idea = st.session_state.current_idea

    answer_result = ask_foundertwin_llm_grounded(
        current_idea,
        question,
        top_n=top_n
    )

    llm_answer = answer_result.get("llm_answer", "")

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": llm_answer
    })


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("FounderTwin Controls")

    top_n = st.slider("Top matches", min_value=3, max_value=10, value=5, step=1)

    new_idea = st.text_area(
        "Enter startup idea",
        value=st.session_state.current_idea,
        height=140,
        placeholder="Example: AI copilot for startup idea validation"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Analyze Idea", use_container_width=True):
            if new_idea.strip():
                try:
                    initial_analysis_flow(new_idea.strip(), top_n=top_n)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error while analyzing idea: {e}")
                    st.code(traceback.format_exc())

    with col2:
        if st.button("Reset", use_container_width=True):
            reset_chat()
            st.rerun()

    st.divider()

    st.subheader("Current Idea")
    if st.session_state.current_idea:
        st.write(st.session_state.current_idea)
    else:
        st.caption("No idea loaded yet.")

    st.divider()

    st.subheader("Structured Analysis")
    render_analysis_summary(st.session_state.current_analysis)


# =========================
# MAIN CHAT AREA
# =========================
st.markdown("""
### How to use
1. Enter a startup idea in the sidebar and click **Analyze Idea**
2. Review the structured analysis and top matches
3. Use the chat below for follow-up questions
4. To switch ideas, either reset or enter a new one in the sidebar
""")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask follow-up questions about users, pricing, competitors, MVP, GTM, risks...")

if prompt:
    try:
        if not st.session_state.current_idea.strip():
            initial_analysis_flow(prompt.strip(), top_n=top_n)
            st.rerun()
        else:
            followup_chat_flow(prompt.strip(), top_n=top_n)
            st.rerun()
    except Exception as e:
        st.error(f"App error: {e}")
        st.code(traceback.format_exc())
