#langchain_basics

from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline


#=====SETUP LLM======

print("loading LLM..")
pipe = pipeline("text-generation" , model = "gpt2" , max_new_tokens = 100 , temperature = 0.7 , do_sample = True)
llm = HuggingFacePipeline(pipeline = pipe)
print("LLM loaded")

#=====PROMPT TEMPLATE=======
template = """
you are a helpful AI assistant.
answer the following question clearly and concisely

question : {question}

answer :"""

prompt = PromptTemplate(input_variables = ["question"] , template = template)

#=====CREATE CHAIN======
chain = prompt | llm | StrOutputParser()

#=======ASK QUESTIONS======

questions = [
    "what is machine learning.?",
    "what is python is used for.?",
    "what is artifical intelligence.?",
]
print("\n=====AI ANSWERS====")
for question in questions:
    print(f"\nQ:{question}")
    response = chain.invoke({"question" : question})
    print(f"A: {response[:300]}")
    print("-" *40)