import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import os

# Get the Hugging Face API token from the environment variable
huggingface_api_token = os.getenv('HUGGINGFACE_API_TOKEN')

import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

load_dotenv()

# Load Hugging Face API token from Streamlit secrets
huggingface_api_token = st.secrets["HUGGINGFACE_API_TOKEN"]

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature": 0.2, "max_new_tokens": 512, "return_full_text": False},
    huggingfacehub_api_token=huggingface_api_token
)

prompt_template = PromptTemplate(
    template="You are an AI assistant helping to answer FAQs. Use the following context to provide an answer. If the user asks an irrelevant question or greets, display appropriate message.\n\nContext:\n{context}\n\nQuery:\n{query}\nAnswer:",
    input_variables=["context", "query"]
)

class Config:
    # Load Google API key from Streamlit secrets
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    MODEL_NAME = "gemini-1.5-flash"
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    TEMPERATURE = 0.2
    MAX_RETRIES = 3
    TOP_K_SUGGESTIONS = 3

import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Optional
import streamlit as st

class FAQRetriever:
    def __init__(self, index_path: str, documents_path: str):
        self.index = faiss.read_index(index_path)
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
    
    def get_suggestions(self, query: str, top_k: int = Config.TOP_K_SUGGESTIONS) -> List[str]:
        if query.strip() == "":
            return []
        try:
            input_embedding = self.model.encode([query])
            distances, indices = self.index.search(np.array(input_embedding), top_k)
            return [self.documents[i]['question'] for i in indices[0]]
        except Exception as e:
            st.error(f"Error generating suggestions: {str(e)}")
            return []
    
    def get_relevant_context(self, query: str, top_k: int = Config.TOP_K_SUGGESTIONS) -> Tuple[str, float]:
        try:
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(np.array(query_embedding), top_k)
            best_distance = distances[0][0]
            results = [self.documents[i] for i in indices[0]]
            context = " ".join([doc['answer'] for doc in results])
            return context, best_distance
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
            return "", float('inf')

class LLMGenerator:
    def __init__(self):
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(
            Config.MODEL_NAME,
            generation_config={
                "temperature": Config.TEMPERATURE,
                "response_mime_type": "application/json"
            }
        )
    
    @staticmethod
    def format_prompt(context: str, query: str) -> str:
        return f"""You are an AI assistant designed to help answer FAQs for Saras AI Institute. Use the following context to provide accurate answers.

Contextual Responses: When asked a relevant question, respond based on the provided context.
Unavailable Information: If you cannot find a relevant answer in the context, respond politely by saying, "I'm sorry, but I don't have specific information about that."
Greetings: When greeted (e.g., "hi," "hello"), respond with an appropriate greeting such as "Hello! How can I assist you today?"
Farewells: When someone says goodbye (e.g., "bye," "thanks"), respond with a friendly farewell such as "Thank you! Have a great day!" 
Irrelevant Queries: For irrelevant or nonsensical questions (e.g., "Who is the PM?"), inform the user that you are a Saras AI FAQ bot and cannot assist with that by saying, "I'm here to help with FAQs related to Saras AI Institute. Please ask a relevant question."
Respond in JSON format with a single key "answer" containing your response.

Context:
{context}

Query:
{query}"""

    def generate_answer(self, query: str, context: str) -> str:
        try:
            prompt = self.format_prompt(context, query)
            response = self.model.generate_content(prompt)
            
            try:
                response_json = json.loads(response.text)
                answer = response_json.get("answer", "")
                if answer and isinstance(answer, str):
                    return answer.strip()
            except json.JSONDecodeError:
                if response.text and isinstance(response.text, str):
                    return response.text.strip()
            
            raise ValueError("Invalid response format")
        except Exception as e:
            st.error(f"LLM Error: {str(e)}")
            return ""

import streamlit as st
import json
import time

class FAQWizardApp:
    def __init__(self):
        self.retriever = FAQRetriever('./faiss_index.index', './documents.pkl')
        self.generator = LLMGenerator()
        self.initialize_session_state()
    
    @staticmethod
    def initialize_session_state():
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": """Welcome to the FAQ Wizard! 😊
Feel free to type your query below to receive some helpful question suggestions. You can click on any suggestion that catches your eye, or if you prefer, type your own question. Happy chatting!"""}
            ]
        if "show_suggestions" not in st.session_state:
            st.session_state.show_suggestions = False
        if "current_suggestions" not in st.session_state:
            st.session_state.current_suggestions = []
    
    def generate_response(self, query: str) -> str:
        context, _ = self.retriever.get_relevant_context(query)
        
        if not context:
            return "I apologize, but I'm having trouble accessing the information at the moment."
        
        for attempt in range(Config.MAX_RETRIES):
            response = self.generator.generate_answer(query, context)
            if response:
                return response
            time.sleep(1)
        
        # Fallback to context if LLM fails
        cleaned_context = context.strip().split('\n')[0]
        return f"Due to temporary issues, I'll provide the relevant information directly: {cleaned_context}"
    
    def render_chat(self):
        st.title("FAQ Wizard Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your question here"):
            self.handle_user_input(prompt)
    
    def render_suggestions_sidebar(self):
        with st.sidebar:
            st.header("Suggestions")
            if st.session_state.show_suggestions and st.session_state.current_suggestions:
                st.subheader("You might also want to ask:")
                for suggestion in st.session_state.current_suggestions:
                    if st.button(suggestion, key=suggestion):
                        self.handle_user_input(suggestion)
    
    def handle_user_input(self, user_input: str):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = self.generate_response(user_input)
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Update suggestions
        st.session_state.current_suggestions = self.retriever.get_suggestions(user_input)
        st.session_state.show_suggestions = True
        st.rerun()

def main():
    app = FAQWizardApp()
    app.render_chat()
    app.render_suggestions_sidebar()

if __name__ == "__main__":
    main()
