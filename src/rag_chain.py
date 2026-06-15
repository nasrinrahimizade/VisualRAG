#  Generating Answers with Citations
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from image_captioner import caption_image, load_captioner
# from langchain_classic.memory import ConversationBufferMemory
# from langchain_classic.chains import ConversationalRetrievalChain

load_dotenv(Path(__file__).parent.parent / ".env")

PROMPT_TEMPLATE = """
You are a research assistant specializing in computer vision and robotics.
Use ONLY the following context from research papers to answer the question.
Always mention which paper (source filename) supports each claim you make.
If the answer is not in the context, say:
"I don't have enough information in my paper collection to answer this."

Context:
{context}

Question: {question}

Answer:
"""


def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        api_key=os.environ.get("GROQ_API_KEY"),
    )


def format_docs(docs):
    """Combine retrieved chunks into a single context string."""
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(vector_store):
    """Build the RAG chain using the modern LangChain approach."""
    llm = load_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    retriever = vector_store.as_retriever(search_type='mmr' , search_kwargs={"k": 4, "fetch_k": 15})

    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True, 
    #     output_key = "answer"
    # )

    # chain = ConversationalRetrievalChain.from_llm( 
    #     llm = llm,
    #     retriever = retriever,
    #     memory = memory ,
    #     return_source_documents=True,
    #     combine_docs_chain_kwargs={"prompt": ChatPromptTemplate.from_template(PROMPT_TEMPLATE)}
    # )
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def ask(rag_chain, retriever, question: str) -> dict:
    """Ask a question and return answer with sources."""
    # Get the answer
    answer = rag_chain.invoke(question)

    # Get source documents separately
    source_docs = retriever.invoke(question)
    # source_docs = result.get("source_documents", [])

    sources = []
    seen = set()
    for doc in source_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": source,
                "page": page,
                "preview": doc.page_content[:200]
            })

    return {
        "answer": answer,
        "sources": sources
    }


def print_answer(result: dict):
    """Pretty-print the answer and citations."""
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(result["answer"])
    print("\n" + "-"*60)
    print("SOURCES:")
    print("-"*60)
    for i, src in enumerate(result["sources"], 1):
        print(f"\n[{i}] {src['source']} — Page {src['page']}")
        print(f"    {src['preview']}...")

def ask_from_image(image_path, chain, retriever, processor, model):
    """Ask a question based on an uploaded image."""
    caption_text = caption_image(image_path, processor, model)
    query = f"Computer vision and robotics research related to: {caption_text}"
    result = ask(chain, retriever, query)
    result["caption"] = caption_text
    return result

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from retrieval import get_or_build_vector_store

    print("Loading vector store...")
    vs = get_or_build_vector_store()

    print("Building RAG chain...")
    chain, retriever = build_rag_chain(vs)

    question = "What tracking method does PhysTwin use?"
    print(f"\nQuestion: {question}")
    result = ask(chain, retriever, question)
    print_answer(result)