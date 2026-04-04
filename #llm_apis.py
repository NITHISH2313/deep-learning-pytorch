#llm_apis.py

import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ===== SETUP API KEY =====
api_key = "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"  # paste your key!

# ===== SIMPLE GROQ TEST =====
print("===== DIRECT GROQ API =====")
client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": "You are a helpful AI assistant."},
        {"role": "user",
         "content": "What is machine learning? explain in 3 lines"}
    ]
)

print(f"Q: What is machine learning?")
print(f"A: {response.choices[0].message.content}")
print()

# ===== LANGCHAIN + GROQ =====
print("===== LANGCHAIN + GROQ =====")
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant",
    temperature=0.7
)

template = """You are an expert AI teacher.
Explain the following topic clearly for a beginner.

Topic: {topic}

Explanation:"""

prompt = PromptTemplate(
    input_variables=["topic"],
    template=template
)

chain = prompt | llm | StrOutputParser()

topics = [
    "neural networks",
    "deep learning",
    "RAG in AI"
]

for topic in topics:
    print(f"\nTopic: {topic}")
    response = chain.invoke({"topic": topic})
    print(f"Explanation: {response[:300]}")
    print("-" * 40)

# ===== CHAT CONVERSATION =====
print("\n===== CHAT CONVERSATION =====")
messages = [
    {"role": "system",
     "content": "You are an AIML tutor helping a student named Nithish from Hyderabad learn GenAI."},
    {"role": "user",
     "content": "I am learning AIML for 23 days. What should I focus on next?"}
]

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=messages
)

print(f"Nithish: I am learning AIML for 23 days. What should I focus on next?")
print(f"AI Tutor: {response.choices[0].message.content}")