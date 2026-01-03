import streamlit as st
from src.rag_service import answer_question

st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")

st.title("📊 Financial Reports Chatbot")
st.caption("Ask questions about financial documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask a question about a financial report...")

if user_input:
    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = answer_question(user_input)
            answer = result["answer"]
            st.markdown(answer)

    # Store assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    # Optional: show sources
    with st.expander("📄 Sources & Confidence"):
        st.write(f"**Overall confidence:** {result['confidence']:.4f}")
        for i, source in enumerate(result["sources"], 1):
            st.markdown(
                f"""
                **{i}. {source['source']}**  
                Page: {source['page']}  
                Confidence: {source['score']:.4f}  

                _{source['preview'][:300]}..._
                """
            )
