from phi.agent import Agent
from phi.model.cohere import CohereChat
from phi.embedder.cohere import CohereEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from lancedb.embeddings import EmbeddingFunctionRegistry

cohere = CohereEmbedder(
    api_key='cohere_api_key',
    model="embed-english-v3.0"
)
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=cohere,  
    ),
)
try:
    knowledge_base.load()
except Exception as e:
    print(f"Knowledge base already loaded or encountered an error: {e}")

agent = Agent(
    model=CohereChat(
        id="command-r-08-2024",
        api_key="cohere_api_key"
    ),
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
)

agent.print_response(
    "How do I make chicken and galangal in coconut milk soup?", 
    stream=True
)
