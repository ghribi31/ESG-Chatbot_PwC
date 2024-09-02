from ESGBOT.model import ESGModel  # Adjusted import to absolute path
from ESGBOT.retriever import ESGRetriever  # Adjusted import to absolute path
from langchain.chains import ConversationalRetrievalChain

class ESGChatbot:
    def __init__(self, model_path, dataset_path):
        self.model = ESGModel(model_path)
        self.retriever = ESGRetriever(dataset_path)
        self.qa_chain = self._create_conversational_chain()

    def _create_conversational_chain(self):
        retriever = self.retriever.get_retriever()
        llm = self.model.get_pipeline()
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

    def query(self, query, chat_history=[]):
        company_keywords = ["company", "performance", "ESG risk", "rating", "sector", "employees", 
                            "decarbonization", "target", "turnover", "Name", "Ticker", "Sector"]
        
        if any(keyword.lower() in query.lower() for keyword in company_keywords):
            result = self.qa_chain.invoke({'question': query, 'chat_history': chat_history})
            answer = result['answer']
        else:
            answer = self.model.generate_response(query)
        
        return answer
