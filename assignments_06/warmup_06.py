 
# Part 1: Warmup Exercises
# --- Environment Setup ---

from pathlib import Path
from dotenv import load_dotenv
import os
import string

# Load the .env file from the repository root.
# This assumes the script is executed from assignments_06.


#load_dotenv("../.env")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")



# RAG Concepts
# --- Concepts Question 1 ---

# Scenario A:
# Best approach: RAG (Retrieval-Augmented Generation).
# The legal team needs answers based on hundreds of internal PDFs
# that change every quarter. RAG retrieves relevant information
# from the latest documents without retraining the model.

# Scenario B:
# Best approach: Fine-tuning.
# The startup has 3,000 examples of its desired brand voice.
# Fine-tuning can teach the model to consistently reproduce
# this specific writing style.

# Scenario C:
# Best approach: Prompt Engineering.
# The analyst only needs help with one short report.
# She can provide the report directly in the prompt along with
# her question, without building a separate retrieval system.


# --- Concepts Question 2 ---

# A confidently wrong answer is more harmful than an answer that admits uncertainty because people may trust and act on information that sounds authoritative.

# For example, if an AI assistant gives incorrect medication instructions in a confident tone, someone might follow them without checking with a qualified healthcare professional.
# An answer that says "I am not sure" encourages the user to verify the information. The tone matters because confident language can make incorrect information appear trustworthy.


# --- Concepts Question 3 ---

# Original unordered RAG pipeline steps:
#
# 1. Generate a response from the LLM
# 2. Extract text from source documents
# 3. Receive the user's query
# 4. Retrieve the most relevant chunks
# 5. Convert text chunks into embeddings
# 6. Inject retrieved chunks into the prompt
# 7. Split text into chunks
# 8. Embed the user's query
#
# Correct RAG pipeline order:
#
# 1. Extract text from source documents.
#    Read the source files and obtain their text content.
#
# 2. Split text into chunks.
#    Divide large documents into smaller sections for retrieval.
#
# 3. Convert text chunks into embeddings.
#    Transform each chunk into a numerical vector representing
#    its semantic meaning.
#
# 4. Receive the user's query.
#    Accept the question that the user wants answered.
#
# 5. Embed the user's query.
#    Convert the question into a numerical vector using
#    the same embedding model.
#
# 6. Retrieve the most relevant chunks.
#    Compare the query embedding with stored chunk embeddings
#    and select the most relevant results.
#
# 7. Inject retrieved chunks into the prompt.
#    Add the retrieved information to the LLM prompt as context.
#
# 8. Generate a response from the LLM.
#    Use the question and retrieved context to generate an answer.


# ==========================================
# Keyword RAG
# --- Keyword Retrieval Function ---

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# --- Keyword Question 1 ---

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

print("\n========== Keyword Question 1 ==========")

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", results[0][0])

## Explanation:
# The result was unexpected because hours.txt contains the
# relevant information about weekend hours.
# However, hours.txt, hiring.txt, and loyalty.txt all received
# an overlap score of 1.
# The function sorts tied results in reverse tuple order,
# so loyalty.txt was selected alphabetically.
# This demonstrates a limitation of simple keyword retrieval.


# --- Keyword Question 2 ---

query = "Do you have anything without caffeine?"

print("\n========== Keyword Question 2 ==========")

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", results[0][0])


## Explanation:
# No document was selected because none of the query words matched the document words after punctuation removal.
# Keyword RAG did not identify the menu because it does not understand the meaning of words or related concepts.
# Semantic RAG would perform better because it compares the meanings of text using vector embeddings.

# --- Keyword Question 3 ---

# Prediction BEFORE running the function:
# I predict loyalty.txt will be selected because it explains how customers can join the loyalty program and earn points.
# However, keyword retrieval may not recognize that "rewards" and "loyalty" have similar meanings.

query = "How do I sign up for rewards?"

print("\n========== Keyword Question 3 ==========")

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", results[0][0])


# # Actual result:
# My prediction was incorrect. No document was selected.
# The loyalty.txt document describes the rewards program
# using the phrase "loyalty program" instead of "rewards".
# Keyword retrieval cannot recognize that these expressions
# have similar meanings because it only checks word overlap.



# ==========================================
# Semantic RAG Concepts
# ==========================================

# --- Semantic Question 1 ---

# 1. What is a vector embedding?
#
# A vector embedding is a list of numbers that represents the meaning of a piece of text.
# It allows computers to compare texts based on their meanings rather than only matching exact words.


# 2. Two text chunks have cosine similarity scores of 0.85 and 0.30. Which is more relevant?
# The chunk with a cosine similarity score of 0.85 is more relevant to the query.
# A score closer to 1 indicates that the two text embeddings have a stronger similarity in their direction and meaning.
# A score of 0.30 indicates a weaker similarity.

# 3. Why can semantic search find relevant chunks even when exact query words do not appear?
# Semantic search compares the meanings of text using vector embeddings instead of looking only for matching words.
# For example, a search for "rewards" may find a document containing "loyalty program" because the concepts are related.


# --- Semantic Question 2 ---

# Keyword RAG vs. Semantic RAG

# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | Vector embeddings and meaning |
# | What is retrieved?         | Full document                     | Relevant text chunks |
# | Can it handle synonyms?    | No                                | Yes, generally |
# | Storage format             | Plain text dictionary             | Vector store/index with embeddings |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity score |
#
# ==========================================
# LlamaIndex
# ==========================================

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

# --- LlamaIndex Setup ---

print("\n========== LlamaIndex Setup ==========")

# Configure the embedding model.
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

# Configure the LLM.
llm = OpenAI(model="gpt-4o-mini")
Settings.llm = llm

# Locate the BrightLeaf Solar PDF directory.
PDF_DIRECTORY = (
    PROJECT_ROOT.parent
    / "python-200-v1"
    / "lessons"
    / "06_AI_augmentation"
    / "resources"
    / "brightleaf_pdfs"
)

# Load PDF documents.
documents = SimpleDirectoryReader(
    input_dir=str(PDF_DIRECTORY)
).load_data()

print("Number of PDF pages loaded:", len(documents))

# Build the in-memory vector index.
index = VectorStoreIndex.from_documents(documents)

print("Vector index created successfully!")

# --- LlamaIndex Question 1 ---

print("\n========== LlamaIndex Question 1 ==========")

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

# Create query engine with top 3 retrieved chunks.
query_engine = index.as_query_engine(
    similarity_top_k=3,
    llm=llm
)

q1_responses = {}

for question in questions:

    print("\nQuestion:", question)

    response = query_engine.query(question)

    q1_responses[question] = response

    print("\nModel Answer:")
    print(response)

    print("\nRetrieved Source Nodes:")

    for i, source in enumerate(response.source_nodes, start=1):

        print(f"\nSource Node {i}")
        print("Similarity Score:", source.score)

        chunk_text = source.node.get_content()

        print("First 150 characters:")
        print(chunk_text[:150])

# Question 1: Employee benefits
# The first retrieved chunk was relevant and came from the employee
# benefits document. However, the second and third chunks were about
# partnerships and the company mission, which were less relevant.
# The model gave a confident and detailed answer. I was surprised that
# unrelated documents were also retrieved.

# Question 2: Security policies
# The first retrieved chunk was relevant and came from the security
# policy document. However, the other chunks were about employee
# benefits and partnerships, which were less relevant.
# The model gave a confident and specific answer. I was surprised that
# the retrieval included documents unrelated to security.

# --- LlamaIndex Question 2 ---

print("\n========== LlamaIndex Question 2 ==========")

question = "What employee benefits does BrightLeaf offer?"

# First run: Retrieve only 1 chunk.
query_engine_top1 = index.as_query_engine(
    similarity_top_k=1,
    llm=llm
)

response_top1 = query_engine_top1.query(question)

print("\n--- TOP 1 RESULT ---")

print("\nResponse:")
print(response_top1)

print("\nSource Node Scores:")

for source in response_top1.source_nodes:
    print(source.score)


# Second run: Retrieve 5 chunks.
query_engine_top5 = index.as_query_engine(
    similarity_top_k=5,
    llm=llm
)

response_top5 = query_engine_top5.query(question)

print("\n--- TOP 5 RESULT ---")

print("\nResponse:")
print(response_top5)

print("\nSource Node Scores:")

for source in response_top5.source_nodes:
    print(source.score)

# Observation:
# Top 1 and Top 5 produced very similar answers.
# Top 5 retrieved more source nodes, but the first source node had the
# highest similarity score in both cases. More retrieved context does
# not always mean a better answer.


# --- LlamaIndex Question 3 ---

print("\n========== LlamaIndex Question 3 ==========")

difficult_question = (
    "What is the exact date BrightLeaf plans to launch "
    "its next product in Europe?"
)

print("\nQuestion:", difficult_question)

response = query_engine.query(difficult_question)

print("\nModel Response:")
print(response)

print("\nAll Retrieved Chunks:")

for i, source in enumerate(response.source_nodes, start=1):

    print(f"\nSource Node {i}")

    print("Similarity Score:", source.score)

    print("Full Chunk Text:")
    print(source.node.get_content())

# # Question 3: Challenge query
#
# Expectation:
# I expected this question to be difficult because I thought the exact
# date of BrightLeaf's next European product launch might not be
# available in the provided documents.

# Actual result:
# The system retrieved partnership, earnings, and mission statement
# information, but none of the retrieved chunks provided the exact
# European product launch date. The model correctly stated that the
# date was not specified instead of inventing one.

# Proposed improvement:
# I would add a similarity score threshold and a fallback message.
# If the retrieved chunks do not meet the threshold, the system could
# explain that it does not have enough relevant information to answer.
# This could help reduce unsupported answers.
# --- LlamaIndex Question 4 ---

print("\n========== LlamaIndex Question 4 ==========")

# Create evaluators using gpt-4o-mini.
faithfulness_evaluator = FaithfulnessEvaluator(
    llm=llm
)

relevancy_evaluator = RelevancyEvaluator(
    llm=llm
)

# Observation:
# The model correctly stated that the exact European product launch date
# was not available in the provided documents. This demonstrates why
# retrieving relevant context and avoiding unsupported answers are important.


# ------------------------------------------
# Evaluation 1: Employee Benefits
# ------------------------------------------

q = "What employee benefits does BrightLeaf offer?"

print("\n--- Employee Benefits Evaluation ---")

response = query_engine.query(q)

faithfulness_result = faithfulness_evaluator.evaluate_response(
    query=q,
    response=response
)

relevancy_result = relevancy_evaluator.evaluate_response(
    query=q,
    response=response
)

print("\nFaithfulness Score:")
print(faithfulness_result.score)

print("\nFaithfulness Feedback:")
print(faithfulness_result.feedback)

print("\nRelevancy Score:")
print(relevancy_result.score)

print("\nRelevancy Feedback:")
print(relevancy_result.feedback)


# ------------------------------------------
# Evaluation 2: Question not covered
# ------------------------------------------

bad_question = (
    "What is BrightLeaf's policy for reimbursing "
    "employees' pet-care expenses?"
)

print("\n--- Pet-Care Reimbursement Evaluation ---")

bad_response = query_engine.query(bad_question)

bad_faithfulness = faithfulness_evaluator.evaluate_response(
    query=bad_question,
    response=bad_response
)

bad_relevancy = relevancy_evaluator.evaluate_response(
    query=bad_question,
    response=bad_response
)

print("\nModel Response:")
print(bad_response)

print("\nFaithfulness Score:")
print(bad_faithfulness.score)

print("\nFaithfulness Feedback:")
print(bad_faithfulness.feedback)

print("\nRelevancy Score:")
print(bad_relevancy.score)

print("\nRelevancy Feedback:")
print(bad_relevancy.feedback)

# QUESTION 4 OBSERVATIONS
#
# 1. Faithfulness:
# A score of 1.0 indicates that the response is fully supported
# by the retrieved context.
# A score of 0.0 indicates that the response is not supported
# by the retrieved context.
#
# 2. Relevancy:
# Relevancy measures how well the response addresses the user's question. Faithfulness measures whether the response is supported by the retrieved context.
#
# 3. Score comparison:
# Compare the employee benefits question with the pet-care reimbursement question. Explain why their scores differ,using the actual evaluator feedback.
#
# 4. LLM-as-a-judge:
# LLM-as-a-judge uses another language model to evaluate generated responses based on specific criteria.
# It is useful for RAG evaluation because answers can be correct in meaning even when their wording differs.
# A simple exact-match accuracy metric cannot capture this.

# QUESTION 4 OBSERVATIONS
#
## Observation:
# The employee benefits query received faithfulness and relevancy scores
# of 1.0, with YES feedback for both.
# The pet-care reimbursement query received scores of 0.0, with NO feedback for both. The documents did not contain a pet-care reimbursement policy.
# LLM evaluators help assess whether an answer is supported by the
# retrieved context and whether it addresses the user's question.