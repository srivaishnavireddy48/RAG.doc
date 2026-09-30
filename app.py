import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# Page settings
st.set_page_config(
    page_title="PDF RAG Q&A",
    page_icon="📚"
)

st.title("📚 PDF RAG Q&A")
st.write("Upload a PDF and ask questions about it.")


# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ChromaDB
client = chromadb.Client()

collection = client.get_or_create_collection(
    name="documents"
)


# PDF input
uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


# Add PDF
if st.button("➕ Add PDF"):

    if uploaded_file is None:

        st.warning("Please upload a PDF.")

    else:

        # Read PDF
        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        if not text.strip():

            st.warning(
                "Could not extract text from this PDF."
            )

        else:

            # Split text into chunks
            chunks = [
                text[i:i + 500]
                for i in range(0, len(text), 500)
            ]

            # Create embeddings
            embeddings = model.encode(chunks)

            # Add to ChromaDB
            collection.add(
                ids=[
                    f"chunk_{i}"
                    for i in range(len(chunks))
                ],
                documents=chunks,
                embeddings=embeddings.tolist()
            )

            st.success(
                f"Added {len(chunks)} PDF chunks!"
            )


# Question
question = st.text_input(
    "❓ Ask a question"
)


# Ask AI
if st.button("🔍 Ask AI"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif collection.count() == 0:

        st.warning(
            "Please add a PDF first."
        )

    else:

        # Create embedding for question
        question_embedding = model.encode(
            [question]
        )[0]

        # Search ChromaDB
        results = collection.query(
            query_embeddings=[
                question_embedding.tolist()
            ],
            n_results=min(
                3,
                collection.count()
            )
        )

        # Get relevant chunks
        retrieved_chunks = results[
            "documents"
        ][0]

        # Combine chunks
        context = "\n\n".join(
            retrieved_chunks
        )


        # Create prompt
        prompt = f"""
You are a helpful AI assistant.

Answer the question only using the
information from the PDF.

PDF CONTEXT:
{context}

QUESTION:
{question}

If the answer is not in the PDF,
say:

"I don't know based on the provided PDF."
"""


        # Ask Ollama
        try:

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            st.subheader("🤖 Answer")

            st.write(
                response["message"]["content"]
            )

        except Exception as e:

            st.error(
                "Ollama is not running or "
                "llama3.2 is not installed."
            )

            st.code(str(e))


st.divider()

st.caption(
    "Python + PyPDF + Sentence Transformers "
    "+ ChromaDB + Ollama + Streamlit"
)