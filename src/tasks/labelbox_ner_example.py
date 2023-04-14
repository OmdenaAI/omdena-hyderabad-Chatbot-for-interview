import binascii
from uuid import uuid4
import random
import time
import tqdm
import os

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer

import labelbox as lb
from labelbox.schema.data_row import DataRow
from labelbox.schema.data_row_metadata import DataRowMetadataField, DataRowMetadataOntology
from labelbox import Ontology, OntologyBuilder, Tool, Project
from labelbox.schema.media_type import MediaType
from labelbox.data.annotation_types import Label, LabelList, ObjectAnnotation, TextEntity, TextData
from labelbox.data.serialization import NDJsonConverter
from labelbox.schema.annotation_import import LabelImport

# TODO: FIXME: Use Omdena's Labelbox API key here
LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGZ0eXM5dXYwNHdoMDcxZ2hsNm4wbWsyIiwib3JnYW5pemF0aW9uSWQiOiJjbGZ0eXM5dWUwNHdmMDcxZzJmNG9jeHg1IiwiYXBpS2V5SWQiOiJjbGZ1ZGN6cm8wa2RkMDcxdGMwdGowOXpqIiwic2VjcmV0IjoiMDIxNjE5OGI3MzMwOGZmMjY5ZGVlZDg2MTUwYjVlZDUiLCJpYXQiOjE2ODAxMzU2MDcsImV4cCI6MjMxMTI4NzYwN30.ajva1WRlKdGf-22s4oYb8khVoNBWJ0VDIHfI13rLgPE"

## Set batch size for batching data rows and annotation bulk import. 500-1000 is recommended size.
BATCH_SIZE = 500

## Set max number of data rows to import. WikiNeural dataset has ~1.1M data rows
MAX_DATA_ROW_LIMIT = 2000

def create_ner_objects(class_name, st, en):
  named_enity = TextEntity(start=st,end=en)
  named_enity_annotation = ObjectAnnotation(value=named_enity, name=class_name)
  return named_enity_annotation

def generate_predictions(datarow, dataset):  
  external_id = datarow["external_id"]
  dataset_name = external_id.split("_")[0] + "_" + external_id.split("_")[1]
  datarow_index = int(external_id.split("_")[2].split(".")[0])
  uid = datarow['id']
  text_data_row =  dataset[dataset_name][datarow_index]
  tokens = text_data_row["tokens"]
  tokenized_input = tokenizer(tokens, is_split_into_words=True)
  sentence = tokenizer.decode(tokenized_input["input_ids"], skip_special_tokens=True)
  annotations = []

  ## Generate prediction
  predictions = nlp(sentence)

  ## process predictions and compute text entities
  try:
    for item in predictions:
      score = item['score']
      if score > 0.99:
        entity = item['entity']
        start = item['start']
        end = item['end']
        index = predictions.index(item)

        if entity =="B-PER":
          for next_item in predictions[index+1:]:
            if next_item['entity']=="I-PER":
              end = next_item['end']
            else:
              break
          annotations.append(create_ner_objects("PER", start, end-1))
          
        if entity =="B-ORG":
          for next_item in predictions[index+1:]:
            if next_item['entity']=="I-ORG":
              end = next_item['end']
            else:
              break
          annotations.append(create_ner_objects("ORG", start, end-1))

        if entity =="B-LOC":
          for next_item in predictions[index+1:]:
            if next_item['entity']=="I-LOC":
              end = next_item['end']
            else:
              break
          annotations.append(create_ner_objects("LOC", start, end-1))

        if entity =="B-MISC":
          for next_item in predictions[index+1:]:
            if next_item['entity']=="I-MISC":
              end = next_item['end']
            else:
              break
          annotations.append(create_ner_objects("MISC", start, end-1))

  except Exception as e:
    print(e)

  text_data = TextData(uid=uid)
  return text_data, annotations

def create_data_rows_payload(payload, language_field: any, embedding_field: any):
  data_row_content = None
  label_content = None

  try:
    h, text_data_row, lang = payload
    file_name = lang + "_" + str(h) +'.txt'
    
    tokens = text_data_row["tokens"]
    tokenized_input = tokenizer(tokens, is_split_into_words=True)
    sentence = tokenizer.decode(tokenized_input["input_ids"], skip_special_tokens=True)

    embeddings = embedding_model.encode(sentence)
    embeddings_metadata = DataRowMetadataField(
          schema_id=embedding_field.uid,
          ## Labelbox currently only supports custom embedding of 128 max length
          value=embeddings[:128].tolist(),
        )
    
    language_metadata = DataRowMetadataField(
          schema_id=language_field.uid,
            value=lang,
        )
    metadata_payload = [language_metadata, embeddings_metadata]
  
    data_row_content = {DataRow.row_data: "gs://labelbox-datasets/wiki_neural_text_ner/"+file_name, DataRow.external_id: file_name, DataRow.metadata_fields: metadata_payload}

  except Exception as e:
    print(e)
  
  return data_row_content, label_content

def create_lb_client() -> lb.Client :
    print(f'{create_lb_client.__name__}')
    # Create Labelbox client
    client = lb.Client(api_key=LB_API_KEY)
    return client

def get_metadata_ontology(client: lb.Client) -> DataRowMetadataOntology:
    print(f'{get_metadata_ontology.__name__}')
    metadata_ontology = client.get_data_row_metadata_ontology()
    return metadata_ontology

def create_ontology(client: lb.Client) -> Ontology:
    print(f'{create_ontology.__name__}')
    ontology = OntologyBuilder()
    PER = Tool(tool = Tool.Type.NER, name = "PER")             
    ontology.add_tool(PER)
    ORG = Tool(tool = Tool.Type.NER, name = "ORG")             
    ontology.add_tool(ORG)
    LOC = Tool(tool = Tool.Type.NER, name = "LOC")             
    ontology.add_tool(LOC)
    MISC = Tool(tool = Tool.Type.NER, name = "MISC")             
    ontology.add_tool(MISC)
  
    ontology = client.create_ontology("WikiNEuRal Text NER", ontology.asdict())
    return ontology

def create_project(ontology: Ontology) -> Project: 
    print(f'{create_project.__name__}')
    project = client.create_project(name = "WikiNEuRal Text NER", media_type=MediaType.Text)
    project.setup_editor(ontology)
    # ontology_from_project = OntologyBuilder.from_project(project)

    return project

def create_data_payload(datasets: DatasetDict) -> list():
    """Process and create data row payload in batches"""
    print(f'{create_data_payload.__name__}')
    tuples = []
    print(f'{datasets}')
    for item in datasets:
    # if item == "train_en":
        for h, text_data_row in enumerate(datasets[item]):
            tuples.append((h, text_data_row, item))

    if MAX_DATA_ROW_LIMIT !=None:
        tuples = random.sample(tuples, MAX_DATA_ROW_LIMIT)

    chunked_tuples = list()
    for i in range(0, len(tuples), BATCH_SIZE):
        chunked_tuples.append(tuples[i:i+BATCH_SIZE])

    return chunked_tuples

def import_data(chunked_tuples: list(), project: OntologyBuilder, labelbox_dataset: Dataset, language_field: any, embedding_field: any) -> None:
    """Main iterator loop to import data"""
    print(f'{import_data.__name__}')
    print(f'language_field: {language_field} embedding_field: {embedding_field}')
    for chunk in chunked_tuples:
        start_time = time.time()
        current_index = chunked_tuples.index(chunk)
        print("Executing {} of {} iteration".format(current_index, len(chunked_tuples)))

        ## Generate data row payload
        data_rows = []
        for item in tqdm.tqdm(chunk):
            datarow,label = create_data_rows_payload(item, language_field, embedding_field)
            data_rows.append(datarow)

        ## Create data rows in Labelbox
        task = labelbox_dataset.create_data_rows(data_rows)
        task.wait_till_done()
        print(task)

        ## Submit a batch of the recently created data rows
        batch_datarows = []
        for item in task.result:
            batch_datarows.append(item['id'])

        batch = project.create_batch(
            str(current_index) + "_" + str(binascii.b2a_hex(os.urandom(5))), # name of the batch
            batch_datarows, # list of Data Rows
            1 # priority between 1-5
        )

        ## Generate model predictions
        ground_truth_list = LabelList()
        results = []

        for item in tqdm.tqdm(task.result):
            result = generate_predictions(item)
            ground_truth_list.append(Label(
                data=result[0],
                annotations = result[1]
            ))

        ## Convert model predictions to NDJSON format
        ground_truth_list.assign_feature_schema_ids(OntologyBuilder.from_project(project))
        ground_truth_ndjson = list(NDJsonConverter.serialize(ground_truth_list))

        ## Upload model predictions as ground truth
        upload_task = LabelImport.create_from_objects(client, project.uid, f"upload-job-{uuid4()}", ground_truth_ndjson)
        upload_task.wait_until_done()
        print(upload_task.errors)

        print(str((time.time() - start_time))+" seconds")

if __name__ == "__main__":
    print(f'loading dataset')
    datasets = load_dataset("Babelscape/wikineural")
    chunked_tuples = create_data_payload(datasets)

    print(f'tokenize dataset')
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")

    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

    print(f'create NER pipeline')
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    embedding_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

    client = create_lb_client()
    labelbox_dataset = client.create_dataset(name="WikiNEuRal Text NER")

    ## Create a custom metadata schema called language of string kind: https://docs.labelbox.com/docs/datarow-metadata#custom-fields    
    metadata_ontology = get_metadata_ontology(client)
    print(f'reserved fields: {metadata_ontology.reserved_fields} custom_fields: {metadata_ontology.custom_fields}')

    language_field = metadata_ontology.custom_by_name["language"]
    embedding_field = metadata_ontology.reserved_by_name["embedding"]

    ontology = create_ontology(client)
    project = create_project(ontology)

    import_data(chunked_tuples, project, labelbox_dataset, language_field, embedding_field)
