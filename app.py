import streamlit as st
import requests

st.set_page_config(layout="wide")
st.title("AI RAG Assistant")

API_URL = "http://127.0.0.1:8000"

# =========================
# Upload
# =========================
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        res = requests.post(
            f"{API_URL}/upload",
            files={"file": uploaded_file}
        )

        data = res.json()

        if "error" in data:
            st.error(data["error"])
        else:
            st.success(f"PDF Ready | Pages: {data.get('pages')} | Chunks: {data.get('chunks')}")


# =========================
# Ask
# =========================
question = st.text_input("Ask a question")

if st.button("Ask"):

    if not question:
        st.warning("Enter a question first")
        st.stop()

    with st.spinner("Thinking..."):

        res = requests.post(
            f"{API_URL}/ask",
            json={"question": question}  # 🔥 FIXED
        )

        data = res.json()

        # Debug (مهم لو فيه مشكلة)
        st.write("DEBUG:", data)

        if "error" in data:
            st.error(data["error"])
            st.stop()

        # =========================
        # Output
        # =========================
        st.subheader("Answer")
        st.write(data.get("answer"))

        st.subheader("Model")
        st.write(data.get("model"))

        # =========================
        # Performance
        # =========================
        st.subheader("Performance")

        perf = data.get("performance", {})

        st.write(f"Latency: {perf.get('latency')} sec")
        st.write(f"Hallucination: {perf.get('hallucination')}")
        st.write(f"Context Length: {perf.get('context_length')}")
        st.write(f"Docs Used: {perf.get('docs_used')}")

        # =========================
        # Agent
        # =========================
        st.subheader("Agent Reasoning")

        agent = data.get("agent", {})

        st.write("Plan:")
        st.write(agent.get("plan"))

        st.write("Verification:")
        st.write(agent.get("verification"))