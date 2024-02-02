path_source_embeddings="C:/Users/alber/Desktop/uni/NLP/NLP-NER-Project/all_embeddings/embeddings_legal1.pt"
path_source_labels="C:/Users/alber/Desktop/uni/NLP/NLP-NER-Project/all_embeddings/labels_legal1.pt"
path_source_val_embeddings="C:/Users/alber/Desktop/uni/NLP/NLP-NER-Project/all_embeddings/embeddings_legal1_val.pt"
path_source_val_labels="C:/Users/alber/Desktop/uni/NLP/NLP-NER-Project/all_embeddings/labels_legal1_val.pt"
path_target_embeddings="C:/Users/alber/Desktop/uni/NLP/NLP-NER-Project/all_embeddings/embeddings_def_train.pt"
path_target_labels="C:/Users/alber/Desktop/uni/NLP/NLP-NER-Project/all_embeddings/labels_def_train.pt"
path_target_val_embeddings="C:/Users/alber/Desktop/uni/NLP/NLP-NER-Project/all_embeddings/embeddings_def_val.pt"
path_target_val_labels="C:/Users/alber/Desktop/uni/NLP/NLP-NER-Project/all_embeddings/labels_def_val.pt"

C:/Users/alber/AppData/Local/Programs/Python/Python312/python.exe c:/Users/alber/Desktop/uni/NLP/NLP-NER-Project/legal_ner/domain_adaptation/train.py --path_source_embeddings "$path_source_embeddings" --path_source_labels "$path_source_labels" --path_target_embeddings "$path_target_embeddings" --path_target_labels "$path_target_labels" --path_target_val_embeddings "$path_target_val_embeddings" --path_target_val_labels "$path_target_val_labels" --path_source_val_embeddings "$path_source_val_embeddings" --path_source_val_labels "$path_source_val_labels" --remove_window_domain_classifier --remove_token_domain_classifier --remove_wordle_game_module