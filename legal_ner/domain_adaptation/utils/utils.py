from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from nervaluate import Evaluator
import torch
from tqdm import tqdm



############################################################
#                                                          #
#                  LABELS MATCHING FUNCTION                #
#                                                          #
############################################################ 
def match_labels(tokenized_input, annotations):

    # Make a list to store our labels the same length as our tokens
    aligned_labels = ["O"] * len(
        tokenized_input["input_ids"][0]
    )  

    # Loop through the annotations
    for anno in annotations:

        previous_tokens = None

        # Loop through the characters in the annotation
        for char_ix in range(anno["start"], anno["end"]):

            token_ix = tokenized_input.char_to_token(char_ix)

            # White spaces have no token and will return None
            if token_ix is not None:  

                # If the token is a continuation of the previous token, we label it as "I"
                if previous_tokens is not None:
                    aligned_labels[token_ix] = (
                        "I-" + anno["labels"]
                        if aligned_labels[token_ix] == "O"
                        else aligned_labels[token_ix]
                    )

                # If the token is not a continuation of the previous token, we label it as "B"
                else:
                    aligned_labels[token_ix] = "B-" + anno["labels"]
                    previous_tokens = token_ix
                    
    return aligned_labels

def extract_embeddings(model, dataloader, save_path, save_path_labels): 
    model.eval() 
    embeddings = [] 
    labels = [] 
    print("Saving embeddings...") 
    with torch.no_grad(): 
        for batch in tqdm(dataloader): 
            ls = batch['labels'] 
            input_ids = batch['input_ids'].to(model.device) 
            attention_mask = batch['attention_mask'].to(model.device) 
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True) 
            embeddings.append(torch.sum(torch.cat(outputs.hidden_states[-4:], dim=0), dim=0)) 
            labels.append(ls) 
    embeddings = torch.cat(embeddings, dim=0) 
    labels_t = torch.tensor(torch.cat(labels, dim=1)).flatten() 
    print(embeddings.shape) 
    print(labels_t.shape) 
    torch.save(embeddings, save_path) 
    torch.save(labels_t, save_path_labels) 
    return embeddings