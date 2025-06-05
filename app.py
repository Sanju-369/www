import streamlit as st
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load Excel data
df = pd.read_excel("Book1.xlsx")
df.columns = [col.strip() for col in df.columns]
df.fillna("Not Available", inplace=True)

# Prepare text
texts1 = [
    " | ".join([f"{col}: {row[col]}" for col in df.columns])
    for _, row in df.iterrows()
]

# Initialize ChromaDB and embeddings
client = chromadb.Client()
collection = client.get_or_create_collection("odisha_hospitals")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Store to ChromaDB
for i, text in enumerate(texts1):
    embedding = model.encode(text).tolist()
    collection.add(documents=[text], embeddings=[embedding], ids=[f"doc_{i}"])

# Groq LLM Setup
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def ask_llm(question, context):
    prompt = f"Answer the question based on this context:\n{context}\n\nQuestion: {question}"
    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
        top_p=1,
    )
    return completion.choices[0].message.content

def query_hospitals(question):
    query_embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    if not results["documents"][0]:
        return "No relevant hospital or blood bank information found in the dataset."

    context = "\n".join(results["documents"][0])
    return ask_llm(question, context)

# Streamlit UI
st.title("🏥 Odisha Hospital Info Assistant")
question = st.text_input("Ask something about hospitals, blood banks, contacts...")

if question:
    with st.spinner("Thinking..."):
        response = query_hospitals(question)
        st.markdown("### 🤖 Answer")
        st.write(response)
