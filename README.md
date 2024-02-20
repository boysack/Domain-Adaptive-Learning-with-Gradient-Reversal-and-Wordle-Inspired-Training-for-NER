# Domain-Adaptive Learning with Gradient Reversal and Wordle-Inspired Training for Named Entity Recognition
Project crafted by Alessio Cappello, Giulia Di Fede, Alberto Foresti and Claudio Macaluso for the Deep Natural Language Processing course at Politecnico di Torino (2023/2024).

This work introduces the following findings:
- A domain-adaptive learning approach for legal named entity recognition (L-NER), which aims to identify and classify specific legal entities in unstructured texts.
- A new training task inspired by the game 'Wordle', where the model is given hints based on the token position and the entity type to improve its learning.
- The work also experiments with a gradient reversal layer to reduce the domain shift between the legal data and the defence data, and compares the performance of domain classifiers at both the token level and the window level.
- We propose an approach on two datasets: the Indian Legal Document Corpus (ILDC) and the R3ad dataset, which are related to the legal and defence domains respectively.
- We report that the proposed extensions did not achieve state-of-the-art results, but gained useful insights about the two domains and the BERT contextualised embeddings, which can be used for future work in this field.
