import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

from src.core.interfaces.base import IRecommendationEngine
from src.core.entities.models import BuyerPreferences, RecommendationResult
from src.infrastructure.config import settings
from src.infrastructure.prompts import RECOMMENDATION_TEMPLATE


class LangChainRecommendationEngine(IRecommendationEngine):
    """RAG-based recommendation engine using LangChain + ChromaDB."""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )
        self._chain: ConversationalRetrievalChain | None = None

    def _build_chain(self, preferences: BuyerPreferences) -> ConversationalRetrievalChain:
        docs = CSVLoader(file_path=settings.data_path).load()
        split_docs = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(docs)
        embedding = OpenAIEmbeddings(api_key=settings.openai_api_key)
        retriever = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding,
            persist_directory=settings.db_path,
        ).as_retriever()

        history = ChatMessageHistory()
        history.add_user_message(
            "You are an AI sales assistant. Summarize buyer home preferences."
        )
        history.add_ai_message(preferences.to_query())

        memory = ConversationSummaryMemory(
            llm=self._llm,
            chat_memory=history,
            memory_key="chat_history",
            input_key="question",
            return_messages=True,
        )

        prompt = PromptTemplate(
            template=RECOMMENDATION_TEMPLATE,
            input_variables=["context", "chat_history", "question"],
        )
        return ConversationalRetrievalChain.from_llm(
            llm=self._llm,
            retriever=retriever,
            memory=memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": prompt},
        )

    async def recommend(self, preferences: BuyerPreferences) -> RecommendationResult:
        chain = await asyncio.to_thread(self._build_chain, preferences)
        query = "As a sales assistant, show the best matching home for this user in an appealing format."
        result = await asyncio.to_thread(chain, {"question": query})
        return RecommendationResult(answer=result["answer"], query=query)
