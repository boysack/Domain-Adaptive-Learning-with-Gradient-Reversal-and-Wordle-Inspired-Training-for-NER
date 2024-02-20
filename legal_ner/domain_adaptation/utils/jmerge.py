import json

# Leggi i dati dai json
documents_file = open('documents.json', 'r', encoding="utf-8")
e_json = documents_file.read().strip()
objects_json = [s.strip() for s in e_json.split("}")]
objects_json.pop()
objects_json = [f"{obj} }}" if not obj.endswith("}") else obj for obj in objects_json]
documents = [json.loads(obj) for obj in objects_json]

entities_file = open('entities.json', 'r', encoding="utf-8")
e_json = entities_file.read().strip()
objects_json = [s.strip() for s in e_json.split("}")]
objects_json.pop()
objects_json = [f"{obj} }}" if not obj.endswith("}") else obj for obj in objects_json]
entities = [json.loads(obj) for obj in objects_json]

# Creare un dizionario per mappare documentId agli oggetti di entities.json
document_id_mapping = {}

for obj_b in entities:
    document_id = obj_b.get('documentId')
    if document_id:
        if document_id not in document_id_mapping:
            document_id_mapping[document_id] = []
        document_id_mapping[document_id].append(obj_b)

# Creare una lista di risultati con lo schema desiderato
result_list = []

for obj_a in documents:
    document_id_a = obj_a.get('_id')
    if document_id_a:
        joined_objects = document_id_mapping.get(document_id_a, [])
        annotations = []

        for joined_obj in joined_objects:
            annotation = {
                "value": {
                    "start": joined_obj.get('begin', 0),
                    "end": joined_obj.get('end', 0),
                    "text": joined_obj.get('value', ''),
                    "labels": [joined_obj.get('type', '')]
                },
                "id": joined_obj.get('_id', ''),
                "from_name": "label",
                "to_name": "text",
                "type": "labels"
            }
            annotations.append(annotation)

        result_item = {
            "id": document_id_a,
            "annotations": [{"result": annotations}],
            "data": {"text": obj_a.get('text', '')},
            "meta": {"source": obj_a.get('sourceUrl', '')}
        }

        result_list.append(result_item)

# Converti il risultato in JSON e scrivilo su un file
with open('result.json', 'w') as result_file:
    json.dump(result_list, result_file, indent=4)