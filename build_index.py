import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def build_index():
    try:
        # --- 1. Check if PDF exists ---
        pdf_path = "chore_schedule.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ PDF file '{pdf_path}' not found! Make sure it's in the same directory.")

        print("📄 Loading PDF...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        if not docs:
            raise ValueError("❌ PDF loaded but no content found!")

        print(f"✅ Loaded {len(docs)} pages from PDF")

        # --- 2. Split into chunks ---
        print("✂️ Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Larger to preserve table structure
            chunk_overlap=300,  # More overlap for context
            separators=["\n\n", "\nWeek", "\nChore", "\n•", "\n", " "]
        )
        splits = text_splitter.split_documents(docs)

        print(f"✅ Created {len(splits)} chunks")

        # Print first few chunks for debugging
        print("\n📝 Sample chunks:")
        for i, chunk in enumerate(splits[:2]):
            print(f"Chunk {i + 1}: {chunk.page_content[:200]}...")

        # --- 3. Use HuggingFace embeddings ---
        print("🧠 Generating embeddings with HuggingFace...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # --- 4. Build FAISS index ---
        print("📦 Building FAISS index...")
        vectorstore = FAISS.from_documents(splits, embeddings)

        # --- 5. Save FAISS index to disk with error handling ---
        print("💾 Saving FAISS index...")
        try:
            with open("faiss_index.pkl", "wb") as f:
                pickle.dump(vectorstore, f)
            print("✅ FAISS index saved successfully!")
        except Exception as e:
            print(f"❌ Error saving FAISS index: {e}")
            # Try alternative serialization
            vectorstore.save_local("faiss_db")
            print("✅ FAISS index saved using save_local method")

        # --- 6. Test loading the index ---
        print("🧪 Testing index loading...")
        try:
            with open("faiss_index.pkl", "rb") as f:
                test_vectorstore = pickle.load(f)
            print("✅ Index loading test successful!")
        except:
            print("⚠️ Pickle loading failed, but save_local files should work")

        print("🎉 Index building complete!")

    except Exception as e:
        print(f"❌ Error building index: {e}")
        raise


if __name__ == "__main__":
    build_index()