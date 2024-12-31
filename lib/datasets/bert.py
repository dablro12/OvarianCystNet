from transformers import BertTokenizer, BertModel
BERT_MODEL_LIST = [
    "madatnlp/km-bert",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
]


#%% Med BERT 이용해서 문장을 임베딩해 라벨을 지정
class bert_labeler:
    def __init__(self, model_name:str, device= 'cuda:0'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
    def encode(self, text:str):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        text_embedding = last_hidden_states[:, 0, :] # CLS 토큰 임베딩 추출
        return text_embedding.squeeze(0).cpu().detach()

# if __name__ == "__main__":
#     %time    
#     embed_model = bert_labeler(model_name = "suayptalha/medBERT-base")
#     print(embed_model.encode("Normal").shape)
