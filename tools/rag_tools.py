from langchain_core.tools import tool
from rag_embedding_retrieval.rag_embedding_retrieval import ragRetriever

@tool
def search_query_using_rag(query: str) -> str:
    """Search the in-project documents using semantic similarity.

    Use this to find relevant information across the project's documentation using RagEmbeddingRetrieval class.

    Args:
        query: Natural language search query, e.g. 'What is the leave policy of the company?'

    Returns:
        Formatted list of matching menu items with details.
    """

    ragRetriever_t = ragRetriever
    result = ragRetriever_t.hybrid_search(query, top_k=3)
    return result

rag_tools = [search_query_using_rag]