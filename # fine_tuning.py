# fine_tuning.py

import torch 
from transformers import (BertTokenizer , BertForSequenceClassification)
from torch.optim import AdamW
from torch.utils.data import Dataset , DataLoader

#===== CUSTOM DATASET======

texts =[
    "I love learning AI and machine learning!",
    "This is an amazing course!",
    "Python is the best programming language!",
    "Hyderabad is a great city for tech jobs!",
    "I am so excited about GenAI!",
    "Deep learning is fascinating!",
    "I hate boring lectures!",
    "This concept is too difficult to understand",
    "I am frustrated with this error",
    "This is the worst code I have ever seen",
    "I don't understand anything!",
    "This is so confusing and complicated!"
]

labels = [1,1,1,1,1,1,  # positive = 1
          0,0,0,0,0,0]  # negative = 0


#====tokenizer=====
print("loading BERT tokenizer..")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#tokenize all texts
encodings = tokenizer(texts , truncation =True ,padding = True , max_length = 64 ,return_tensors ='pt')
print(f"Input IDs shape :{encodings['input_ids'].shape}")
print(f"samples tokens :{encodings['input_ids'][0][:0]}")

#====dataset class=====
class sentimentdataset(Dataset):
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self , idx):
        item = {key : val[idx]
                for key , val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

dataset = sentimentdataset(encodings , labels)
loader = DataLoader(dataset , batch_size = 4, shuffle = True)

print(f"\n dataset size :{len(dataset)}")
print(f"batches :{len(loader)}")


#=====LOAD BERT======
print("\n Loading BERT model...")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels = 2)

#==== FINE TUNE =====

optimizer = AdamW(model.parameters() , lr =2e-5)

print("\n Fine tuning Bert")
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(input_ids = batch['input_ids'], attention_mask =batch['attention_mask'],labels = batch['labels'])
        loss =outputs.loss
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"epoch {epoch+1}/3 -"
          f"loss : {total_loss/len(loader):.4f}")
    
#====test=====
print("\n Testing fine tuned model")
model.eval()
test_texts = [
    "I love building AI projects!",
    "This is really difficult and frustrating",
    "Nithish will become a great GenAI engineer!",
    "I cannot understand this at all"
]

with  torch.no_grad():
    for text in test_texts:
        encoding = tokenizer(text ,return_tensors ='pt', truncation = True,padding =True,max_length = 64)
        output = model(**encoding)
        prediction = torch.argmax(output.logits)
        sentiment = "positive" if prediction == 1 else  "negative"
        print(f"text:{text}")
        print(f"sentiment :{sentiment}")
        print()


