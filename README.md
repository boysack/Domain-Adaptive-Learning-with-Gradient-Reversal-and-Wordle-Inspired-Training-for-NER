# Domain-Adaptive Learning with Gradient Reversal and Wordle-Inspired Training for Named Entity Recognition
Project crafted by Alessio Cappello, Giulia Di Fede, Alberto Foresti and Claudio Macaluso for the Deep Natural Language Processing course at Politecnico di Torino (2023/2024).

This work introduces the following findings:
- A domain-adaptive learning approach for legal named entity recognition (L-NER), which aims to identify and classify specific legal entities in unstructured texts.
- A new training task inspired by the game 'Wordle', where the model is given hints based on the token position and the entity type to improve its learning.
- The work also experiments with a gradient reversal layer to reduce the domain shift between the legal data and the defence data, and compares the performance of domain classifiers at both the token level and the window level.
- We propose an approach on two datasets: the Indian Legal Document Corpus (ILDC) and the R3ad dataset, which are related to the legal and defence domains respectively.
- We report that the proposed extensions did not achieve state-of-the-art results, but gained useful insights about the two domains and the BERT contextualised embeddings, which can be used for future work in this field.

## Requirements installation
- Run the following line to install the requirements:
```bash
pip install -r NLP-NER-Project/legal_ner/requirements.txt
```

## Embeddings extraction
- Change directory to legal_ner:
``` bash
cd legal_ner
```
- Run main.py with the extract embeddings flag activated
```bash
python main.py --extract_embedding=True
```
Other parameters, such as the checkpoint folder and the datasets folder, can be set using the parser inside main.py.

## How to launch the model
- Change directory to legal_ner:
``` bash
cd legal_ner
```
- Launch the run.sh file:
``` bash
./run.sh
```
In the run.sh file it is possible to set the paths for the embeddings and the configuration for the run, which is a yaml file. It is possible to choose which experiment to run:
- gridsearch.yaml: it contains the values to look at for the betas
- ablation.yaml: it contains boolean values for the components to keep/remove 
It is also possible to specify the modality by setting the `action` parameter to:
- train: train the model with the specified parameters
- validate: perform inference on both source and target domain
- gridsearch: perform the randomised grid search taking the combinations from the yaml file

