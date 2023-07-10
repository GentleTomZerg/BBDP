# !pip install datasets transformers evaluate accelerate
# !pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER
sentimentAnalyser = SentimentIntensityAnalyzer()

import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt

import itertools
from datasets import Dataset

import torch
from torch import nn
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np
from dataclasses import dataclass

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from typing import List, Optional, Tuple, Union
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

root_dir = './'

train_data = pd.read_csv(f'{root_dir}/drugsComTrain_raw.csv')
test_data = pd.read_csv(f'{root_dir}/drugsComTest_raw.csv')

def calculate_sentiment(text):
    # Run VADER on the text
    scores = sentimentAnalyser.polarity_scores(text)
    # Extract the compound score
    compound_score = scores['compound']
    # Return compound score
    # original -1 ~ 1, we change to 0 ~ 1
    return (compound_score + 1) / 2

train_data['sentiment_score'] = train_data['review'].apply(calculate_sentiment)
test_data['sentiment_score'] = test_data['review'].apply(calculate_sentiment)

train_data['useful_cnt_log'] = train_data['usefulCount'].apply(lambda x: np.log2(x+1))
test_data['useful_cnt_log'] = test_data['usefulCount'].apply(lambda x: np.log2(x+1))

# train_data

data = pd.concat([train_data, test_data])
data = data.dropna(how='any')
data[['drugName','condition','review']] = data[['drugName','condition','review']].applymap(lambda x: x.lower())
data = data[['drugName','condition','review','rating','sentiment_score','useful_cnt_log']]

data

# data['drugName'] = data['drugName'].apply(lambda x: [y.strip() for y in x.split(' / ')])
# drug_cnt = Counter(itertools.chain(*data['drugName'].tolist()))
# data['drug_cnt'] = data['drugName'].apply(lambda x: [drug_cnt[y] for y in x])
# data['drug_cnt_min'] = data['drug_cnt'].apply(lambda x: min(x))
# data = data.loc[data['drug_cnt_min']>50]

drug_cnt = Counter(data['drugName'])
data['drug_cnt'] = data['drugName'].apply(lambda x: drug_cnt[x])
data = data.loc[data['drug_cnt']>100]

new_data = []
for drugName in set(data['drugName']):
    sub_data = data.loc[data['drugName']==drugName]
    if sub_data.shape[0]>100:
        sub_data = sub_data.sample(n=100)
    new_data.append(sub_data)
data = pd.concat(new_data,axis=0)

# data.isna().sum()

data['rating'] = data['rating']/10
data = data[['drugName','condition','rating','sentiment_score','useful_cnt_log']].reset_index(drop=True)
# data = data[:10]

len(set(data['drugName']))

label2id = {drug:i for i, drug in enumerate(set(data['drugName']))}
id2label = {v:k for k,v in label2id.items()}

data['drugLabel'] = data['drugName'].apply(lambda x: label2id[x])
data = data.rename(columns = {'drugLabel':'labels'})
# data['']

# id2label

dataset = Dataset.from_pandas(data)
dataset = dataset.train_test_split(test_size=0.3)

dataset

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
def preprocess_function(examples):
    return tokenizer(examples["condition"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

class RatedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # print(list(inputs.keys()))
        labels = inputs.get("labels")
        rating = inputs.get("rating")
        sentiment_score = inputs.get("sentiment_score")
        useful_cnt_log = inputs.get("useful_cnt_log")
        # print(useful_cnt_log)
        # forward pass
        outputs = model(**inputs)
        # print(outputs)

        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(reduction = 'none')

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        # print(inputs)


        loss = (loss * (rating.view(-1)) * sentiment_score.view(-1) * useful_cnt_log.view(-1)).mean()


        return (loss, outputs) if return_outputs else loss

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    # print(eval_pred)
    predictions, labels = eval_pred
    predictions = predictions[0]
    # print(predictions)
    # print(labels)
    predictions = np.argmax(predictions, axis=1)
    # print(predictions.shape, labels.shape)
    return accuracy.compute(predictions=predictions, references=labels)

@dataclass
class RateSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rating: Optional[Tuple[torch.FloatTensor]] = None

class RateModel(BertForSequenceClassification):

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,

        rating: Optional[torch.Tensor] = None,
        sentiment_score: Optional[torch.Tensor] = None,
        useful_cnt_log: Optional[torch.Tensor] = None,

        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        # if labels is not None:
        #     if self.config.problem_type is None:
        #         if self.num_labels == 1:
        #             self.config.problem_type = "regression"
        #         elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        #             self.config.problem_type = "single_label_classification"
        #         else:
        #             self.config.problem_type = "multi_label_classification"

        #     if self.config.problem_type == "regression":
        #         loss_fct = MSELoss()
        #         if self.num_labels == 1:
        #             loss = loss_fct(logits.squeeze(), labels.squeeze())
        #         else:
        #             loss = loss_fct(logits, labels)
        #     elif self.config.problem_type == "single_label_classification":
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     elif self.config.problem_type == "multi_label_classification":
        #         loss_fct = BCEWithLogitsLoss()
        #         loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RateSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rating = rating
        )

model = RateModel.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT", num_labels=len(label2id), id2label=id2label, label2id=label2id
)
# model = RateModel().load_state

training_args = TrainingArguments(
    output_dir=f'{root_dir}/modelv2',
    overwrite_output_dir = True,
    learning_rate=2e-5,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    num_train_epochs=30,
    weight_decay=0.,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy = "steps",
    save_steps = 100,
    eval_steps = 100,
    logging_steps = 100,
    load_best_model_at_end=True,
)

trainer = RatedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model()

text = ['trigeminal neuralgia', 'constipation']
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
inputs = tokenizer(text, return_tensors="pt",padding=True,truncation=True)
model = RateModel.from_pretrained(
    f"{root_dir}/modelv2",
).eval()
# model = RateModel().load_state
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax(dim=-1)
for id_ in predicted_class_id:
    # print(id_.item())
    print(model.config.id2label[id_.item()])

# torch.load(f"{root_dir}/model/checkpoint-3500/pytorch_model.bin",map_location=torch.device('cpu'))