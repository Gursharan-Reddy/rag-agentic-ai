import streamlit as st
import os
from src.config import PINECONE_INDEX_NAME

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("Agentic AI RAG Chatbot")
st.markdown("Ask questions strictly based on the **Agentic AI For Executives** eBook.")

@st.cache_resource
def get_graph():
    from src.graph import build_rag_graph
    return build_rag_graph(PINECONE_INDEX_NAME)

try:
    with st.spinner("Initializing RAG graph and loading models..."):
        graph = get_graph()
    st.success("System ready!")
except Exception as e:
    st.error(f"Error loading RAG graph: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about Agentic AI..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating response..."):
            try:
                state = {"question": query, "context": [], "answer": "", "score": 0.0}
                result = graph.invoke(state)
                
                answer = result.get("answer", "No answer generated.")
                chunks = result.get("context", [])
                score = result.get("score", 0.0)

                st.markdown(answer)
                
                with st.expander("🔍 View Retrieved Context Chunks & Confidence Score"):
                    st.metric(label="Confidence Score", value=f"{score:.2f}")
                    st.markdown(f"**Retrieved Chunks ({len(chunks)}):**")
                    for i, chunk in enumerate(chunks, 1):
                        st.info(f"**Chunk {i}:**\n\n{chunk}")

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")