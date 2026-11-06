
# Part 1: Warmup Exercises

from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env file
load_dotenv()

# Create OpenAI client
client = OpenAI()


# --- Completions API ---

# API Q1

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?"
        }
    ]
)

# Print the model's response
print("Model Response:")
print(response.choices[0].message.content)

# Print model name
print("Model Name:", response.model)

# Print total tokens used
print("Total Tokens Used:", response.usage.total_tokens)



# API Q2 - Temperature

prompt = "Suggest a creative name for a data engineering consultancy."

temperatures = [0, 0.7, 1.5]

for temperature in temperatures:

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )

    print(f"\nTemperature: {temperature}")
    print("Response:", response.choices[0].message.content)


# Observation:
# Temperature 0 usually produces more consistent and predictable
# responses. Higher temperatures can produce more varied and creative
# responses. I would use temperature 0 when I need consistent,
# reproducible output.


# API Q3 - Multiple Completions

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
        }
    ],
    n=3,
    temperature=1.0
)

# Print all three completions
for index, choice in enumerate(response.choices, start=1):
    print(f"\nCompletion {index}:")
    print(choice.message.content)


# API Q4 - Maximum Tokens

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work."
        }
    ],
    max_tokens=15
)

print("\nAPI Q4 - Maximum Tokens")
print("Response:", response.choices[0].message.content)


# Observation:
# The response is limited because max_tokens sets a maximum
# number of tokens the model can generate. This is useful for
# controlling response length, reducing costs, and managing output.



# --- System Messages and Personas ---

# System Q1 - Different Personalities

# First personality: Patient Python tutor

messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q1 - Patient Python Tutor")
print(response.choices[0].message.content)


# Second personality: Funny comedian

messages = [
    {
        "role": "system",
        "content": "You are a funny comedian who explains programming concepts using simple jokes and humor."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q1 - Funny Comedian")
print(response.choices[0].message.content)


# Observation:
# Observation:
# The patient tutor used a supportive and simple teaching style.
# The comedian used jokes, humor, and funny comparisons.
# The system message changed the personality, tone, and style of the model's response, even though the user question stayed the same.



# System Q2 - Conversation History

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q2 - Conversation History")
print("Response:", response.choices[0].message.content)


# Observation:
# The API is stateless, meaning it does not remember previous calls.
# The model knows Jordan's name because we included the previous
# conversation messages in the current API request.


# --- Prompt Engineering ---

# Prompt Q1 - Zero-Shot Prompting

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Reviews:
{reviews}

Return each review number with its sentiment.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nPrompt Q1 - Zero-Shot Sentiment Analysis")
print(response.choices[0].message.content)


# Prompt Q2 - One-Shot Prompting

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Here is an example:

Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify these reviews:

{reviews}

Return each review number with its sentiment.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nPrompt Q2 - One-Shot Sentiment Analysis")
print(response.choices[0].message.content)


# Observation:
# Adding one example helped the model understand the expected format.
# Compared to zero-shot prompting, one-shot prompting can provide
# more consistent and organized results.


# Prompt Q3 - Few-Shot Prompting

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify each review as positive, negative, or mixed.

Here are three examples:

Review: "The product works perfectly and is easy to use."
Sentiment: positive

Review: "The application is slow and full of errors."
Sentiment: negative

Review: "The price is good, but the customer service is poor."
Sentiment: mixed

Now classify these reviews:

{reviews}

Return each review number with its sentiment.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nPrompt Q3 - Few-Shot Sentiment Analysis")
print(response.choices[0].message.content)


# Observation:
# Zero-shot: I would use it for simple tasks that do not need examples.
# One-shot: I would use it when I want to show the model the expected format.
# Few-shot: I would use it when a task needs more examples to improve consistency and accuracy.


# Prompt Q4 - Chain of Thought

prompt = """
Solve this salary problem. Show the calculation steps clearly,
then provide a clearly labeled final answer.

A data engineer earns $85,000 per year. She gets a 12% raise,
then 6 months later takes a new job that pays $7,500 more per year
than her post-raise salary.

What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nPrompt Q4 - Chain of Thought")
print(response.choices[0].message.content)


# Observation:
# Asking the model to break a problem into calculation steps
# helps it organize the information and avoid calculation mistakes.
# It makes the answer easier to check.


# Prompt Q5 - Structured Output

import json

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f"""
Analyze the sentiment of this review.

Return ONLY valid JSON with these keys:
- sentiment
- confidence (a float from 0 to 1)
- reason (one sentence)

Review:
{review}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

raw_response = response.choices[0].message.content

print("\nPrompt Q5 - Structured Output")
print("Raw Response:")
print(raw_response)

try:
    #result = json.loads(raw_response)
    cleaned_response = raw_response.strip()

    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response.split("\n", 1)[1]
        cleaned_response = cleaned_response.rsplit("```", 1)[0].strip()

    result = json.loads(cleaned_response)

    print("\nSentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
    print("\nError: Could not parse the JSON response.")
    print("Raw response for debugging:", raw_response)


# Prompt Q6 - Delimiters

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nPrompt Q6 - Delimiters (Instructions)")
print(response.choices[0].message.content)


# Test with regular text (not instructions)

plain_text = "The city park is full of colorful flowers. Many people visit it on sunny afternoons to relax and enjoy nature."

second_prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{plain_text}```
"""

second_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": second_prompt}
    ]
)

print("\nPrompt Q6 - Delimiters (Regular Prose)")
print(second_response.choices[0].message.content)


# Observation:
# Delimiters separate the text supplied by the user from the
# instructions given to the model. This helps the model understand
# which text it should process and which text tells it what to do.



# Ollama Question 1 - Local Models

# Ollama output:
"""
A large language model is a type of AI model that can understand
and generate human-like text, trained on massive datasets to
process and analyze vast amounts of information. It enables tasks
like writing, answering questions, or creating content by learning
patterns in unstructured text.
"""
# OpenAI Q1 - Same Prompt

local_prompt = "Explain what a large language model is in two sentences."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": local_prompt}
    ]
)

openai_output = response.choices[0].message.content

print("\nOpenAI Q1 - Response")
print(openai_output)

# Differences:
# Both models explained that large language models understand
# and generate human language. The Ollama response mentioned
# learning patterns in unstructured text and gave examples of
# tasks such as writing and answering questions. The OpenAI
# response provided more detail about machine learning,
# deep learning, grammar, context, and meaning.
#
# Advantage of running a model locally:
# A local model can run on your computer without sending
# prompts to an external cloud API.
#
# Disadvantage of running a model locally:
# A local model requires computer resources, such as memory
# and processing power, and may be less capable than a
# larger hosted model.