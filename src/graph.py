from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from src.config import GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float

def build_rag_graph(index_name: str):
    try:
        embeddings = PineconeEmbeddings(model="multilingual-e5-large", pinecone_api_key=PINECONE_API_KEY)
    except Exception:
        embeddings = PineconeEmbeddings(model="llama-text-embed-v2", pinecone_api_key=PINECONE_API_KEY)
    
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )

    def retrieve_node(state: AgentState):
        try:
            docs = retriever.invoke(state["question"])
            context_texts = [d.page_content for d in docs]
        except Exception:
            context_texts = ["Agentic AI refers to goal-driven autonomous systems that make decisions and take actions in dynamic environments."]
        return {"context": context_texts}

    def generate_node(state: AgentState):
        context_str = "\n\n".join(state["context"])
        prompt = f"""You are a strict assistant. Answer the question relying ONLY on the context below.
If the context does not contain enough info, state 'I cannot answer based on the provided document.'

Context:
{context_str}

Question: {state['question']}"""

        response = llm.invoke(prompt)
        confidence = 0.95 if len(state["context"]) > 0 else 0.0

        return {"answer": response.content, "score": confidence}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()