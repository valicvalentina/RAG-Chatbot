import streamlit as st
import cohere
import faiss
import numpy as np

# Postavi Cohere API ključ
COHERE_API_KEY = "rD5U0m2YXuAqF2mFZrEpNprhZC9ZGq0Gh3R0di1Y"
cohere_client = cohere.Client(api_key=COHERE_API_KEY)

# Dokumenti za pretraživanje
documents = [
    "Python je popularan programski jezik.",
    "FAISS je biblioteka za pretraživanje vektora.",
    "LangChain olakšava integraciju LLM-a s alatima poput FAISS-a.",
    "Retrieval-Augmented Generation je tehnika kombiniranja dohvaćanja i generiranja."
]

# Funkcija za generiranje embeddinga pomoću Cohere
def generate_embeddings(texts, client):
    response = client.embed(model="embed-english-v2.0", texts=texts)
    return np.array(response.embeddings)

# Kreiraj FAISS bazu
doc_embeddings = generate_embeddings(documents, cohere_client)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])  # FAISS index
index.add(doc_embeddings)

# Funkcija za dohvaćanje relevantnih dokumenata
def get_relevant_documents(query, index, documents, client, k=3):
    query_embedding = generate_embeddings([query], client)[0]
    distances, indices = index.search(np.array([query_embedding]), k)
    return [documents[i] for i in indices[0]]

# Chatbot klasa
class Chatbot:
    def __init__(self):
        self.template = """
        You are a friendly assistant. Answer the user's questions as best as you can.

        Context: {context}
        Question: {question}
        Answer:
        """

    def generate_response(self, question, context):
        prompt = self.template.format(context=context, question=question)
        response = cohere_client.generate(
            model="command-xlarge-nightly",
            prompt=prompt,
            max_tokens=500,
            temperature=0.5,
        )
        return response.generations[0].text.strip()

# Inicijalizacija bota
bot = Chatbot()

# Streamlit sučelje
st.set_page_config(page_title="Chat Assistant")
with st.sidebar:
    st.title("Chat Assistant")

# Pohrana povijesti poruka
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Pozdrav! Kako vam mogu pomoći?"}]

# Prikaz chat poruka
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Unos korisničkog pitanja
if user_input := st.chat_input("Unesite pitanje..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generiranje odgovora
    if st.session_state["messages"][-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Generiram odgovor..."):
                relevant_docs = get_relevant_documents(user_input, index, documents, cohere_client)
                context = " ".join(relevant_docs)
                response = bot.generate_response(user_input, context)
                st.write(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
