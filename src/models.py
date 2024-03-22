import numpy as np
from src.utils import format_text_pred
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm



class AllAsTextModel:
    def __init__(self, model1, cols_to_str_fn, tokenizer):
        self.tokenizer = tokenizer
        self.model1 = model1
        self.cols_to_str_fn = cols_to_str_fn
        # self.cols = cols
    
    def prepare_text(self, text):
        text = [str(item) for item in text]
        encoded = self.tokenizer(text,
                           add_special_tokens=True,
                           padding='max_length',
                           truncation=True,
                           max_length=256,
                           return_token_type_ids=False,
                           return_tensors='tf')
        return {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask']}

    def predict(self, examples):
        # examples_as_strings = np.apply_along_axis(
        #     lambda x: array_to_string(x, self.cols), 1, examples
        # )
        
        examples_as_strings = np.array(list(map(self.cols_to_str_fn, examples)))
        # print(examples_as_strings)
        prepared_input = self.prepare_text(examples_as_strings)
        predictions = self.model1.predict(prepared_input)
        # Since your final layer uses a sigmoid activation, this will be a probability
        # You may want to round the prediction for classification (0 or 1)
        predicted_class = np.round(predictions).astype(int)  # Convert probabilities to 0 or 1

        # for out in self.text_pipeline(KeyDataset(Dataset.from_dict({"text": examples_as_strings}), "text"), batch_size=64):
        #     print('out', out)
        #     break
        
        # preds = [
        #     out
        #     for out in self.text_pipeline(
        #         KeyDataset(Dataset.from_dict({"text": examples_as_strings}), "text"),
        #         batch_size=64,
        #     )
        # # ]
        # np.array(
        # [scores[i] for i in sorted(range(len(scores)), key=lambda x: order[x])]
        preds = np.array([format_text_pred(predictions, predicted_class)])
        preds = preds.flatten().tolist()
        # preds = np.array([[lab["score"] for lab in pred] for pred in preds])

        return preds
