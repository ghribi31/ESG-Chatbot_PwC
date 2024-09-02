from langchain_community.document_loaders.csv_loader import CSVLoader  # Updated import
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import

class ESGRetriever:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.docs = self._load_and_split_documents()
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
        self.db = self._create_faiss_index()

    def _load_and_split_documents(self):
        loader = CSVLoader(file_path=self.dataset_path)
        data = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(data)

    def _create_faiss_index(self):
        return FAISS.from_documents(self.docs, self.embeddings)

    def get_retriever(self):
        return self.db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 4}
        )
