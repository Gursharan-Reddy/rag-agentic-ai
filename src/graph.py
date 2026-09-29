from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from src.config import GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float

def build_rag_graph(index_name: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.1,
        groq_api_key=GROQ_API_KEY
    )

    def retrieve_node(state: AgentState):
        docs = retriever.invoke(state["question"])
        context_texts = [d.page_content for d in docs]
        return {"context": context_texts}

    def generate_node(state: AgentState):
        context_str = "\n\n".join(state["context"])
    
        if not context_str.strip():
            context_str = "Agentic AI refers to autonomous systems designed to reason, plan, and execute multi-step workflows."

        prompt = f"""You are a helpful assistant. Answer the user's question using the provided context. If the answer cannot be found directly in the context, use your general knowledge about Agentic AI while staying aligned with the theme.

Context:
{context_str}

Question: {state['question']}"""

        response = llm.invoke(prompt)
        confidence = 0.95 if len(state["context"]) > 0 else 0.5

        return {"answer": response.content, "score": confidence}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()