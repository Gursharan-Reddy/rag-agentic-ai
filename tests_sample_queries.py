from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME

test_queries = [
    "What is Agentic AI and how does it differ from traditional AI and LLMs?",
    "What are the core pillars (building blocks) of an Agentic AI system?",
    "How do Multi-Agent Systems (MAS) handle a supply chain crisis compared to normal AI systems?",
    "What are the execution patterns available in a multi-agent workflow?",
    "What are the four stages of organization readiness levels for adopting Agentic AI?",
    "Who won the 2022 FIFA World Cup?"
]

def run_tests():
    print("Compiling RAG Graph for testing against the eBook document...")
    graph = build_rag_graph(PINECONE_INDEX_NAME)
    
    for i, q in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: {q} ---")
        state = {"question": q, "context": [], "answer": "", "score": 0.0}
        result = graph.invoke(state)
        print(f"Answer:\n{result['answer']}")
        print(f"Confidence Score: {result['score']}")

if __name__ == "__main__":
    run_tests()