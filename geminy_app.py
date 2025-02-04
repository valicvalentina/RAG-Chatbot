import streamlit as st
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Postavi API ključ za Gemini (Google AI)
GOOGLE_API_KEY = "AIzaSyBA78X7vPPKGCxKFfCu9X5gU83JXIZzxeQ"
genai.configure(api_key=GOOGLE_API_KEY)

# Model za embeddinge (radi lokalno)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Python je popularan programski jezik.",
    "FAISS je biblioteka za pretraživanje vektora.",
    "LangChain olakšava integraciju LLM-a s alatima poput FAISS-a.",
    "Retrieval-Augmented Generation je tehnika kombiniranja dohvaćanja i generiranja."
]

def generate_embeddings(texts):
    return np.array(embedding_model.encode(texts))

doc_embeddings = generate_embeddings(documents)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

def get_relevant_documents(query, k=3):
    query_embedding = generate_embeddings([query])[0]
    distances, indices = index.search(np.array([query_embedding]), k)
    return [documents[i] for i in indices[0]]

class Chatbot:
    def generate_response(self, question, context):
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"""
        Kontekst: {context}
        Pitanje: {question}
        Odgovor:
        """
        response = model.generate_content(prompt)
        return response.text.strip()

bot = Chatbot()
st.set_page_config(page_title="Chat Assistant")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Pozdrav! Kako vam mogu pomoći?"}]

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_input := st.chat_input("Unesite pitanje..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Generiram odgovor..."):
            relevant_docs = get_relevant_documents(user_input)
            context = " ".join(relevant_docs)
            response = bot.generate_response(user_input, context)
            st.write(response)
    
    st.session_state["messages"].append({"role": "assistant", "content": response})
