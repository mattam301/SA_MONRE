from transformers import AutoModel, AutoTokenizer
import re
import torch
import numpy

class BERT_FEATURES:
    def __init__(self):
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    def standardize_data(self, row):
        # Xóa dấu chấm, phẩy, hỏi ở cuối câu
        row = re.sub(r"[\.,\?]+$-", "", row)
        # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
        row = row.replace(",", " ").replace(".", " ") \
            .replace(";", " ").replace("“", " ") \
            .replace(":", " ").replace("”", " ") \
            .replace('"', " ").replace("'", " ") \
            .replace("!", " ").replace("?", " ") \
            .replace("-", " ").replace("?", " ")
        return row

    def make_bert_features(self, v_text):
        v_tokenized = []

        max_len = 256  # Mỗi câu dài tối đa 256 từ
        for i_text in v_text:
            i_text = self.standardize_data(i_text)
            line = self.tokenizer.encode(i_text, max_length=256)
            v_tokenized.append(line)

        # Chèn thêm số 1 vào cuối câu nếu như không đủ 256 từ
        padded = numpy.array([i + [1] * (max_len - len(i)) for i in v_tokenized])
        print('padded:', padded[0])
        print('len padded:', padded.shape)

        # Đánh dấu các từ thêm vào = 0 để không tính vào quá trình lấy features
        attention_mask = numpy.where(padded == 1, 0, 1)
        print('attention mask:', attention_mask[0])

        # Chuyển thành tensor
        padded = torch.tensor(padded).to(torch.long)
        print("Padd = ", padded.size())
        attention_mask = torch.tensor(attention_mask)

        # Lấy features dầu ra từ BERT
        with torch.no_grad():
            last_hidden_states = self.phobert(input_ids=padded, attention_mask=attention_mask)

        v_features = last_hidden_states[0][:, 0, :].numpy()
        print(v_features.shape)
        return v_features
