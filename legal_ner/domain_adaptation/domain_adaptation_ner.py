import os
import json
import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, RobertaTokenizerFast
from transformers import AutoModelForTokenClassification


############################################################
#                                                          #
#                        NER EXTRACTOR                     #
#                                                          #
############################################################
class NERExtractor:
    def __init__(self, ner_model_path, tokenizer, original_label_list):
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            ner_model_path
        )
        self.ner_model
        self.ner_model.eval()
        self.tokenizer = tokenizer

        labels_list = ["B-" + l for l in original_label_list]
        labels_list += ["I-" + l for l in original_label_list]
        labels_list = sorted(labels_list + ["O"])[::-1]
        self.labels_to_idx = dict(
            zip(sorted(labels_list)[::-1], range(len(labels_list)))
        )
        print(self.labels_to_idx)
        self.idx_to_labels = {v[1]: v[0] for v in self.labels_to_idx.items()}

    ## Extract NER from text
    def extract_ner(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            verbose=False, 
            return_offsets_mapping=True
        )
        offset_mapping = inputs['offset_mapping'].squeeze(0).tolist()[1:-1]
        
        del inputs['offset_mapping']

        with torch.no_grad():
            logits = self.ner_model(**inputs).logits

        predicted_token_class_ids = logits.argmax(-1).squeeze(0).cpu().numpy().tolist()[1:-1]
        
        predictions = []
        for i, (offset, prediction) in enumerate(zip(offset_mapping, predicted_token_class_ids)):

            prediction = self.idx_to_labels[prediction].split('-')[-1]

            if prediction != "O":

                if i > 0:
                  prec_prediction = self.idx_to_labels[predicted_token_class_ids[i-1]].split('-')[-1]

                  if prediction == prec_prediction:
                      predictions[-1]['end'] = offset[1]
                  else:
                      predictions.append(
                          {
                              'label': prediction,
                              'start': offset[0],
                              'end': offset[1],
                          }
                      )
                else:
                  predictions.append(
                    {
                        'label': prediction,
                        'start': offset[0],
                        'end': offset[1],
                      }
                  )
        
        return predictions


############################################################
#                                                          #
#                        INFERENCE                         #
#                                                          #
############################################################                    

## Define the models to use with the corresponding checkpoint and tokenizer
base_dir = "results"
all_model_path = [
    (f'{base_dir}/bert-large-NER/checkpoint-65970',
    'dslim/bert-large-NER'),                    # ft on NER
    (f'{base_dir}/roberta-large-ner-english/checkpoint-65970',
    'Jean-Baptiste/roberta-large-ner-english'), # ft on NER
    (f'{base_dir}/nlpaueb/legal-bert-base-uncased/checkpoint-65970',
    'nlpaueb/legal-bert-base-uncased'),         # ft on Legal Domain
    (f'{base_dir}/saibo/legal-roberta-base/checkpoint-65970',
    'saibo/legal-roberta-base'),                # ft on Legal Domain
    (f'{base_dir}/nlpaueb/bert-base-uncased-eurlex/checkpoint-65970',
    'nlpaueb/bert-base-uncased-eurlex'),        # ft on Eurlex
    (f'{base_dir}/nlpaueb/bert-base-uncased-echr/checkpoint-65970',
    'nlpaueb/bert-base-uncased-echr'),          # ft on ECHR
    (f'{base_dir}/studio-ousia/luke-base/checkpoint-65970',
    'studio-ousia/luke-base'),                  # LUKE base
    (f'{base_dir}/studio-ousia/luke-large/checkpoint-65970',
    'studio-ousia/luke-large'),                 # LUKE large
]

## Loop over the models
for model_path in sorted(all_model_path):

    ## Load the test data
    test_data = 'data/NER_TEST/NER_TEST_DATA_FS.json'
    data = json.load(open(test_data)) 

    ## Load the tokenizer
    tokenizer_path = model_path[1]
    if 'luke' in model_path[0]: 
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) 
    
    ## Define the labels list
    ll = [
            "COURT",
            "PETITIONER",
            "RESPONDENT",
            "JUDGE",
            "DATE",
            "ORG",
            "GPE",
            "STATUTE",
            "PROVISION",
            "PRECEDENT",
            "CASE_NUMBER",
            "WITNESS",
            "OTHER_PERSON",
            "LAWYER"
    ]

    ## Initialize the NER extractor
    ner_extr = NERExtractor(
        ner_model_path = model_path[0], 
        tokenizer = tokenizer, 
        original_label_list=ll)
    
    print(model_path)
    print(tokenizer)

    ## Extract NER from the test data
    for i in tqdm(range(len(data))):

        text = data[i]['data']['text']
        source = data[i]['meta']['source']
        
        results = ner_extr.extract_ner(text)
        
        results_output = []
        for j, r in enumerate(results):
            o = {
                "value": {
                    "start": r['start'],
                    "end": r['end'],
                    "text": text[r['start']:r['end']],
                    "labels": [r['label']]
                },
                "id": f"{i}-{j}",
                "from_name": "label",
                "to_name": "text",
                "type": "labels"
            }
            results_output.append(o)
        data[i]['annotations'][0]['result'] = results_output
    
    ## Save the results
    json.dump(data, open(f'{base_dir}/all/{model_path[0].split("/")[-2]}_predictions.json', 'w'))import os
import json
import numpy as np
from argparse import ArgumentParser
from nervaluate import Evaluator

from transformers import AutoModelForTokenClassification
from transformers import Trainer, DefaultDataCollator, TrainingArguments

from utils.dataset import LegalNERTokenDataset

import spacy
nlp = spacy.load("en_core_web_sm")


############################################################
#                                                          #
#                           MAIN                           #
#                                                          #
############################################################ 
if __name__ == "__main__":

    parser = ArgumentParser(description="Training of LUKE model")
    parser.add_argument(
        "--ds_train_path",
        help="Path of train dataset file",
        default="/content/NLP-NER-Project/legal_ner/NER_TRAIN/NER_TRAIN_JUDGEMENT.json",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--ds_valid_path",
        help="Path of validation dataset file",
        default="/content/NLP-NER-Project/legal_ner/NER_DEV/NER_DEV_JUDGEMENT.json",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--output_folder",
        help="Output folder",
        default="results/",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--batch",
        help="Batch size",
        default=1,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--num_epochs",
        help="Number of training epochs",
        default=5,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--lr",
        help="Learning rate",
        default=1e-5,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        help="Weight decay",
        default=0.01,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--warmup_ratio",
        help="Warmup ratio",
        default=0.06,
        required=False,
        type=float,
    )

    args = parser.parse_args()

    ## Parameters
    ds_train_path = args.ds_train_path  # e.g., 'data/NER_TRAIN/NER_TRAIN_ALL.json'
    ds_valid_path = args.ds_valid_path  # e.g., 'data/NER_DEV/NER_DEV_ALL.json'
    output_folder = args.output_folder  # e.g., 'results/'
    batch_size = args.batch             # e.g., 256 for luke-based, 1 for bert-based
    num_epochs = args.num_epochs        # e.g., 5
    lr = args.lr                        # e.g., 1e-4 for luke-based, 1e-5 for bert-based
    weight_decay = args.weight_decay    # e.g., 0.01
    warmup_ratio = args.warmup_ratio    # e.g., 0.06

    ## Define the labels
    original_label_list = [
        "COURT",
        "PETITIONER",
        "RESPONDENT",
        "JUDGE",
        "DATE",
        "ORG",
        "GPE",
        "STATUTE",
        "PROVISION",
        "PRECEDENT",
        "CASE_NUMBER",
        "WITNESS",
        "OTHER_PERSON",
        "LAWYER"
    ]
    labels_list = ["B-" + l for l in original_label_list]
    labels_list += ["I-" + l for l in original_label_list]
    num_labels = len(labels_list) + 1

    ## Compute metrics
    def compute_metrics(pred):

        # Preds
        predictions = np.argmax(pred.predictions, axis=-1)
        predictions = np.concatenate(predictions, axis=0)
        prediction_ids = [[idx_to_labels[p] if p != -100 else "O" for p in predictions]]

        # Labels
        labels = pred.label_ids
        labels = np.concatenate(labels, axis=0)
        labels_ids = [[idx_to_labels[p] if p != -100 else "O" for p in labels]]
        unique_labels = list(set([l.split("-")[-1] for l in list(set(labels_ids[0]))]))
        unique_labels.remove("O")

        # Evaluator
        evaluator = Evaluator(
            labels_ids, prediction_ids, tags=unique_labels, loader="list"
        )
        results, results_per_tag = evaluator.evaluate()

        return {
            "f1-type-match": 2
            * results["ent_type"]["precision"]
            * results["ent_type"]["recall"]
            / (results["ent_type"]["precision"] + results["ent_type"]["recall"] + 1e-9),
            "f1-partial": 2
            * results["partial"]["precision"]
            * results["partial"]["recall"]
            / (results["partial"]["precision"] + results["partial"]["recall"] + 1e-9),
            "f1-strict": 2
            * results["strict"]["precision"]
            * results["strict"]["recall"]
            / (results["strict"]["precision"] + results["strict"]["recall"] + 1e-9),
            "f1-exact": 2
            * results["exact"]["precision"]
            * results["exact"]["recall"]
            / (results["exact"]["precision"] + results["exact"]["recall"] + 1e-9),
        }

    ## Define the models
    model_paths = [
        "dslim/bert-large-NER",                     # ft on NER
        "Jean-Baptiste/roberta-large-ner-english",  # ft on NER
        "nlpaueb/legal-bert-base-uncased",          # ft on Legal Domain
        "saibo/legal-roberta-base",                 # ft on Legal Domain
        "nlpaueb/bert-base-uncased-eurlex",         # ft on Eurlex
        "nlpaueb/bert-base-uncased-echr",           # ft on ECHR
        "studio-ousia/luke-base",                   # LUKE base
        "studio-ousia/luke-large",                  # LUKE large
    ]

    for model_path in model_paths:

        print("MODEL: ", model_path)

        ## Define the train and test datasets
        use_roberta = False
        if "luke" in model_path or "roberta" in model_path:
            use_roberta = True

        train_ds = LegalNERTokenDataset(
            ds_train_path, 
            model_path, 
            labels_list=labels_list, 
            split="train", 
            use_roberta=use_roberta
        )

        val_ds = LegalNERTokenDataset(
            ds_valid_path, 
            model_path, 
            labels_list=labels_list, 
            split="val", 
            use_roberta=use_roberta
        )

        ## Define the model
        model = AutoModelForTokenClassification.from_pretrained(
            model_path, 
            num_labels=num_labels, 
            ignore_mismatched_sizes=True
        )

        ## Map the labels
        idx_to_labels = {v[1]: v[0] for v in train_ds.labels_to_idx.items()}

        ## Output folder
        new_output_folder = os.path.join(output_folder, 'all')
        new_output_folder = os.path.join(new_output_folder, model_path)
        if not os.path.exists(new_output_folder):
            os.makedirs(new_output_folder)

        ## Training Arguments
        training_args = TrainingArguments(
            output_dir=new_output_folder,
            num_train_epochs=num_epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=False,
            save_total_limit=2,
            fp16=False,
            fp16_full_eval=False,
            metric_for_best_model="f1-strict",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )

        ## Collator
        data_collator = DefaultDataCollator()

        ## Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

        ## Train the model and save it
        trainer.train()
        trainer.save_model(output_folder)
        trainer.evaluate()



"""python 3.10
Example of usage:
python main.py \
    --ds_train_path data/NER_TRAIN/NER_TRAIN_ALL.json \
    --ds_valid_path data/NER_DEV/NER_DEV_ALL.json \
    --output_folder results/ \
    --batch 256 \
    --num_epochs 5 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.06
"""# Legal Named Entity Recognition
This folder contains the code and the data for the Legal NER task (Task 6.B).


## Requirements
The code is written in Python 3.10. The required packages are listed in `requirements.txt`. To install the required packages, run:

    pip install -r requirements.txt

## Code 
The main code for the L-NER task allowing to fine-tune the models is available in the `main.py` script.  
The `inference.py` script allows instead to predict the labels for the test set.

The `utils` folder contains the code for the data loading and the evaluation.

The data are not included in this repository as they are not yet publicly available.
More information are provided in the [official SemEval-2023 Task 6 website](https://sites.google.com/view/legaleval/home).


## Usage
To fine-tune the models on the train data and evaluate them on the dev set, run:

    python main.py

To predict the labels for the test set, run:

    python inference.py
nervaluate==0.1.8 
numpy==1.23.5 
scikit_learn==1.2.2
spacy==3.6.0
torch==2.1.0
tqdm==4.64.0 
transformers==4.26.0
typing-extensions==4.5.0
