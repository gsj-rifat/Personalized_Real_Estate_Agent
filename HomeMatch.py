import os
import pandas as pd
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

# Initialize chat model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=2000)

# Step 1: Generate Home Listings using LLM
prompt_template = PromptTemplate(
    template="""
Generate a CSV table about {topic} with these columns: {attributes}.
Use real-world examples. Generate {rows} rows. Output only the CSV content.
""",
    input_variables=["topic", "attributes", "rows"]
)

generated_csv = llm.invoke(
    prompt_template.format(
        topic="Homes",
        attributes="Neighborhood, Location, Bedrooms, Bathrooms, House Size (sqft), Price (k$)",
        rows="20"
    )
)

# Save to CSV
with open("data/home.csv", "w") as f:
    f.write(generated_csv.content)

# Step 2: Load and embed documents
df = pd.read_csv("data/home.csv")

loader = CSVLoader(file_path="data/home.csv")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
db = Chroma.from_documents(documents=split_docs, embedding=embedding, persist_directory="db")
retriever = db.as_retriever()

# Step 3: Simulate buyer preferences
personal_questions = [
    "How big do you want your house to be?",
    "What are 3 most important things for you in choosing this property?",
    "Which amenities would you like?",
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?",
]

personal_answers = [
    "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
    "A quiet neighborhood, good local schools, and convenient shopping options.",
    "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
    "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
]

history = ChatMessageHistory()
history.add_user_message(
    f"You are an AI sales assistant. Ask the user {len(personal_questions)} questions to understand preferences."
)
for question, answer in zip(personal_questions, personal_answers):
    history.add_ai_message(question)
    history.add_user_message(answer)

history.add_ai_message("Now summarize the preferred home features.")

memory = ConversationSummaryMemory(
    llm=llm,
    chat_memory=history,
    memory_key="chat_history",
    input_key="question",
    buffer="Summarize user preferences based on their answers. Focus on location, budget, area, and features.",
    return_messages=True
)

# Step 4: Define prompt for retrieval
qa_prompt = PromptTemplate(
    template="""
You are a sales assistant helping a home buyer.
Use the retrieved context and customer preferences to suggest a suitable home.
Keep it attractive and concise. Max 5 sentences.

Context: {context}
Preferences: {chat_history}
Question: {question}

Answer:
""",
    input_variables=["context", "chat_history", "question"]
)

# Step 5: Build Conversational Retrieval Chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    chain_type="stuff",
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)

# Step 6: Run assistant
query = "As a sales assistant, show the best matching home for this user in an appealing format."
result = chain({"question": query})
print("\nPersonalized Recommendation:\n")
print(result["answer"])
