import os
import time
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.callbacks.manager import get_openai_callback
from agents.logger import log_interaction


class KnowledgeAgent:
    def __init__(self, doc_path="docs/support_faq.txt"):
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        if os.path.exists("faiss_index"):
            # ✅ Load from existing index (uploaded files)
            self.vectorstore = FAISS.load_local(
                "faiss_index",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            # ⚠️ Fallback to default static doc
            loader = TextLoader(doc_path)
            docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            split_docs = splitter.split_documents(docs)
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)

        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )

    def run(self, query, channel="chat"):
        if not self.qa:
            return "No knowledge base available. Please upload a document."

        with get_openai_callback() as cb:
            start = time.time()
            result = self.qa.invoke(query)
            end = time.time()

            print(f"Response: {result['result']}")
            print(f"Tokens used: {cb.total_tokens}")
            print(f"Cost: ${cb.total_cost:.6f}")
            print(f"Latency: {end - start:.2f} seconds")

        log_interaction({
            "agent": "KnowledgeAgent",
            "channel": channel,
            "query": query,
            "response": result['result'],
            "tokens": cb.total_tokens,
            "cost": cb.total_cost,
            "latency": round(end - start, 2)
        })

        return result['result']

    @staticmethod
    def rebuild_index_from_folder(folder_path):
        print(f"📄 Rebuilding FAISS index from: {folder_path}")

        from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredMarkdownLoader
        from langchain_text_splitters import CharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS

        docs = []
        for fname in os.listdir(folder_path):
            file_path = os.path.join(folder_path, fname)

            if fname.endswith(".txt"):
                loader = TextLoader(file_path)
            elif fname.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
            elif fname.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                print(f"❌ Skipping unsupported file type: {fname}")
                continue

            docs.extend(loader.load())

        if not docs:
            raise ValueError("No supported documents found for indexing.")

        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        split_docs = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.from_documents(split_docs, embeddings)

        vectorstore.save_local("faiss_index")
        print("✅ FAISS index saved.")


