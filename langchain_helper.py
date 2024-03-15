from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import os

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Create Google Generative AI LLM model
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key="AIzaSyCmp4GRrVJSFDadvjdZowuBday4kYXQdcg", temperature=0.1)

# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='dreaddit/val.csv', source_column="post")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings, )

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb=FAISS.load_local("faiss_index", instructor_embeddings,allow_dangerous_deserialization=True)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold = 0.7,search_kwargs={"k": 2})

    prompt_template = """ You are a psychiatrist - a medical doctor (an M.D. or D.O.) who specializes in mental health, including substance use disorders, anxiety, and detecting depression, By using the provided context and your experties as psychiatrist, evaluate the content of social media post provided in post section and provide the answer to question.

    CONTEXT: ```{context}```

    POST: ```{question}```

    QUESTION: ```Does the poster suffer from stress?```"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    input_key="query",
                                    verbose=True,
                                    chain_type_kwargs = {"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    # create_vector_db()
    chain = get_qa_chain()
    print(chain("I think of the feeling alone and not finding anyone to convey my thought and suffering from not good health from last 4 weeks."))