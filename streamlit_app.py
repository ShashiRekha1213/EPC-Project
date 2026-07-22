import streamlit as st

from supervisor import supervisor, VALID_AGENTS


# =====================================================
# Page Setup
# =====================================================

st.set_page_config(
    page_title="EPC Project Assistant",
    page_icon="🏗️",
    layout="centered",
)

AGENT_LABELS = {
    "knowledge": "🗂️ Knowledge Agent",
    "schedule": "📅 Schedule Risk Agent",
    "supply_chain": "🚚 Supply Chain Agent",
    "compliance": "✅ Compliance Agent",
    "commissioning": "🔧 Commissioning Agent",
}


# =====================================================
# Sidebar
# =====================================================

with st.sidebar:
    st.markdown("### EPC Project Assistant")
    st.caption("Multi-agent copilot for Data Centre construction projects.")

    st.markdown("---")
    st.markdown("**Agents**")
    for name, label in AGENT_LABELS.items():
        st.markdown(f"- {label}")

    st.markdown("---")

    override = st.selectbox(
        "Force a specific agent (optional)",
        options=["Auto (let the supervisor decide)"] + sorted(VALID_AGENTS),
        index=0,
        help="Normally the supervisor LLM picks the right agent for you. "
             "Override this only for testing a specific agent directly.",
    )

    selected_agent = None if override.startswith("Auto") else override

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# =====================================================
# Session State
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "agent": None,
            "content": (
                "Hi, I'm your EPC Project Assistant. Ask me about project "
                "status, schedule risk, procurement, compliance, or "
                "commissioning."
            ),
        }
    ]


# =====================================================
# Render Chat History
# =====================================================

st.title("🏗️ EPC Project Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("agent"):
            st.caption(AGENT_LABELS.get(msg["agent"], msg["agent"]))
        st.markdown(msg["content"])


# =====================================================
# Chat Input
# =====================================================

query = st.chat_input(
    "Ask about project status, schedule risk, procurement, compliance…"
)

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = supervisor(query, agent=selected_agent)

        agent_used = result.get("agent")
        answer = result.get("answer", "")

        if not result.get("success"):
            st.error(answer)
        else:
            st.caption(AGENT_LABELS.get(agent_used, agent_used))
            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "agent": agent_used,
            "content": answer,
        }
    )
