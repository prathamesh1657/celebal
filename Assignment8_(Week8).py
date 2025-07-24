import pandas as pd
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer
import streamlit as st




openai.api_key = "your_openai_api_key"


try:
    data = pd.read_csv("Training Dataset.csv")
except FileNotFoundError:
    st.error("Dataset file not found.")
    st.stop()


data = data.fillna("N/A")

records = []
for i in range(len(data)):
    row_text = " | ".join(str(x) for x in data.iloc[i])
    records.append(row_text)


model = SentenceTransformer('all-MiniLM-L6-v2')


document_embeddings = model.encode(records, show_progress_bar=True)


embedding_dim = document_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(np.array(document_embeddings))




def find_similar_documents(query_text, top_k=5):
    query_vector = model.encode([query_text])
    distances, indices = faiss_index.search(query_vector, top_k)
    similar_docs = [records[idx] for idx in indices[0]]
    return similar_docs




def get_generated_response(user_question):
    retrieved_docs = find_similar_documents(user_question)
    context_block = "\n".join(retrieved_docs)

    full_prompt = (
        f"You are helping a user understand loan-related data.\n"
        f"Refer to the records below to provide an answer.\n\n"
        f"---\n{context_block}\n---\n"
        f"Question: {user_question}\n"
        f"Reply clearly:"
    )

    try:
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use free-tier model if credits available
            messages=[
                {"role": "system", "content": "You answer questions about loan applications using provided data."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.4,
            max_tokens=300
        )
        reply = result['choices'][0]['message']['content'].strip()
        return reply
    except Exception as e:
        return f"Error generating response: {e}"




st.set_page_config(page_title="Loan Assistant Q&A", layout="centered")
st.title("🏦 Loan Approval Q&A System")
st.markdown("Ask anything related to the loan application records provided.")

user_input = st.text_input("🔍 Your question:")

if user_input:
    with st.spinner("Processing your question..."):
        output = get_generated_response(user_input)
    st.markdown("### ✅ Answer")
    st.write(output)
