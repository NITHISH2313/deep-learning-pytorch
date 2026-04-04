#huggingface_basics
from transformers import pipeline

#===== 1.sentiment analysis=====

print("=====SENTIMENT ANALYSIS=====")
sentiment = pipeline("sentiment-analysis")

texts = [
    "I love learning AIML!",
    "This is the worst day ever",
    "Hyderabad is an amazing city",
    "I am struggling with this concept"

]
for text in texts:
    result = sentiment(text)[0]
    print(f"text :{text}")
    print(f"sentiment ;{result['label']} ({result['score']*100:.1f}%)")
    print()
#======2.text generation=====
print("=====TEST GENERATION======")
generator = pipeline("text-generation",model="gpt2", max_new_tokens=50)
prompt = "Artificial Intelligence will"
result = generator(prompt)[0]
print(f"generated :{result['generated_text']}")


#=====3.zero shot classification======
print("=====ZERO SHOT CLASSIFICATION======")
classifier = pipeline("zero-shot-classification")

text ="i want to learn python and build AI apps"
labels = ["education","sports" , "technology","cooking"]

result = classifier(text, candidate_labels = labels)
print(f"text :{text}")
for label,score in zip(result['labels'],result['scores']):
    print(f"{label}:{score*100:.1f}%")

# ===== 4. FILL MASK =====
print("===== FILL MASK =====")
fill_mask = pipeline(
    "fill-mask",
    model="bert-base-uncased"
)

sentences = [
    "Nithish wants to become an [MASK]to become GENAI engineer",
    "Hyderabad is a great [MASK] for tech jobs",
    "Python is used for [MASK] learning"
]

for sentence in sentences:
    result = fill_mask(sentence)[0]
    print(f"Input:  {sentence}")
    print(f"Output: {result['sequence']}")
    print(f"Score:  {result['score']*100:.1f}%")
    print()
