
# Part 2: Mini-Project - Groundwork Coffee Co. Q&A Assistant

# Step 1: Setup

from pathlib import Path
from dotenv import load_dotenv
import os

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


# Find the homework repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load the API key from the repository root .env file.
load_dotenv(PROJECT_ROOT / ".env")

# Verify that the API key was loaded.
if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    raise ValueError("API key not found. Check your .env file.")

# Locate the Groundwork documents.
docs_dir = (
    PROJECT_ROOT.parent
    / "python-200-v1"
    / "lessons"
    / "06_AI_augmentation"
    / "resources"
    / "groundwork_docs"
)

# Verify that the document directory exists.
assert docs_dir.is_dir(), f"Document directory not found: {docs_dir}"

print("Groundwork document directory found successfully.")



# Step 2: Load the Documents

documents = SimpleDirectoryReader(
    input_dir=str(docs_dir)
).load_data()

print("\nDocuments loaded successfully!")
print("Total documents loaded:", len(documents))

print("\nDocument filenames:")

for document in documents:
    print("-", document.metadata["file_name"])


# Step 3: Build the Index and Query Engine

# Set the embedding model.
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

# Set the language model.
llm = OpenAI(model="gpt-4o-mini")
Settings.llm = llm

# Build the vector index from the documents.
index = VectorStoreIndex.from_documents(documents)

# Create the query engine.
query_engine = index.as_query_engine(
    similarity_top_k=3,
    llm=llm
)

print("\nIndex built successfully. Ready to answer questions.")



# Step 4: Query the Assistant

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

print("\n" + "=" * 60)
print("GROUNDWORK COFFEE Q&A ASSISTANT")
print("=" * 60)

for question in questions:

    # Ask the query engine.
    response = query_engine.query(question)

    print("\nQuestion:", question)
    print("\nAnswer:")
    print(response)

    # Display the top retrieved source node.
    if response.source_nodes:
        source = response.source_nodes[0]

        print("\nTop Retrieved Source Node:")
        print("Document:", source.node.metadata.get("file_name"))
        print("Similarity Score:", source.score)
        print("First 200 characters:")
        print(source.node.get_content()[:200])

    print("\n" + "-" * 60)

# Reflection:
# The assistant provided answers to all five questions using the retrieved Groundwork Coffee documents. The answers appeared confident and relevant to the questions asked. The retrieved
# source documents helped connect the answers to the original information. I was interested to see how the assistant could
# answer questions about different topics, including business hours, dairy-free options, the loyalty program, company history, and catering services. Overall, the results showed that the
# RAG pipeline can retrieve relevant information and generate useful answers based on the provided documents.
# I will review the five answers to determine whether the assistant provided accurate information from the Groundwork documents.
# I will also check whether the retrieved source documents match
# the questions being asked.


# Step 5: Find a Failure

challenge_question = (
    "What is the exact date Groundwork will open its fourth "
    "location, and what city will it be in?"
)

print("\n" + "=" * 60)
print("STEP 5: CHALLENGE QUESTION")
print("=" * 60)

# Ask the assistant the challenge question.
challenge_response = query_engine.query(challenge_question)

print("\nQuestion:", challenge_question)
print("\nFull Response:")
print(challenge_response)

# Display all three retrieved source nodes.
print("\nAll Retrieved Source Nodes:")

for i, source in enumerate(challenge_response.source_nodes, start=1):
    print(f"\nSource Node {i}")
    print("Document:", source.node.metadata.get("file_name"))
    print("Similarity Score:", source.score)
    print("First 200 characters:")
    print(source.node.get_content()[:200])

print("\n" + "=" * 60)

# Reflection:
# I asked about the opening date and city of a fourth location.
# I chose this question because the provided documents only describe
# Groundwork's three existing locations.
#
# I will evaluate whether the assistant admits that the information
# is missing or generates an unsupported answer.
#
# I will also consider how retrieval quality and the language model's
# response affect the reliability of a RAG system.


# Step 5 Reflection:
#
# I asked when and where Groundwork would open its fourth location.
# I expected this question to be difficult because the documents only
# describe the company's three existing locations.
#
# The assistant correctly stated that the exact opening date and city
# were not provided in the documents. It did not invent information.
#
# The first two retrieved documents contained some background about
# the company and its existing locations. However, the third document
# about wholesale and catering was not directly relevant.
#
# This shows that retrieval does not always return perfectly relevant
# information. The assistant also needs to recognize when the retrieved
# context does not contain the answer.
#
# To improve the system, I could add instructions telling the model
# to answer only from the provided documents, use a similarity
# threshold to filter weak results, and return a message when the
# required information is unavailable.


# Step 6: Reflection

# 1. How much code did LlamaIndex save?
#
# My LlamaIndex implementation used approximately 12 core lines
# to configure the embedding model and language model, build the
# vector index, and create the query engine.
#
# Without LlamaIndex, I would need to write more code to split
# documents into chunks, convert the chunks into embeddings,
# store and organize the vectors, calculate similarity scores,
# and retrieve the most relevant chunks.
#
# The keyword-based RAG approach also required manually processing
# query words, removing stopwords, comparing keywords, and ranking
# documents.
#
# This shows that a framework like LlamaIndex saves development
# time and makes it easier to build a RAG application without
# implementing every step from scratch.


# 2. Another business use case for RAG
#
# A company could build an HR assistant using its employee handbook,
# benefits information, vacation policies, and workplace guidelines.
#
# Employees could ask questions about company policies and receive
# answers based on the official documents instead of searching
# through multiple files manually.
#
# This could save employees time and help HR teams answer common
# questions more efficiently.


# 3. One failure mode that RAG cannot fully prevent
#
# RAG cannot completely prevent the language model from misunderstanding
# retrieved information or generating an incorrect answer.
#
# Even when the retrieval process finds relevant documents, the model
# might misinterpret the context or combine information incorrectly.
#
# Therefore, RAG can reduce unsupported answers, but it cannot
# guarantee that every answer will be accurate.
