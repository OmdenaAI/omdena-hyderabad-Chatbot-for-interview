from dotenv import load_dotenv, find_dotenv
import pandas as pd
import os
import chromadb
from chromadb.utils import embedding_functions


def generate_qa_vector_db(vdb_path: str, df: pd.DataFrame) -> None:
    """This function processes the dataframe into the required format, and then creates the following collections in a ChromaDB instance
    1. question_collection - Contains question embeddings, and the metadata as 'position' and 'interview_phase'
    2. answer_collection - Contains the answer embeddings. No metadata (yet).

    Args:
        vdb_path (str): Relative path of the location of the ChromaDB instance.
        df (pd.DataFrame): Question/answer dataset.
    """
    chroma_client = chromadb.PersistentClient(path=vdb_path)

    huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
        api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    print("q_collection will be added")
    q_collection = chroma_client.create_collection(
        name="question_collection",
        metadata={"hnsw:space": "cosine"},
        embedding_function=huggingface_ef,
    )

    # Keep only question-related columns
    df_questions = df[
        ["Position/Role", "Question", "Interview Phase"]
    ].drop_duplicates()

    # df_questions = df_questions.drop_duplicates().reset_index(drop=True)
    df_questions.columns = [
        x.replace(" ", "_").lower().replace("/", "_or_") for x in df_questions.columns
    ]

    q_documents = [row.question for row in df_questions.itertuples()]
    q_metadata = [
        {"position": row.position_or_role, "interview_phase": row.interview_phase}
        for row in df_questions.itertuples()
    ]
    q_ids = ["q_id" + str(row.Index) for row in df_questions.itertuples()]

    q_collection.add(documents=q_documents, metadatas=q_metadata, ids=q_ids)
    print("q_collection added")

    print("a_collection will be added")
    a_collection = chroma_client.create_collection(
        name="answer_collection",
        metadata={"hnsw:space": "cosine"},
        embedding_function=huggingface_ef,
    )

    df_answers = df[["Answer", "Answer Quality"]]
    df_answers.columns = [
        x.replace(" ", "_").lower().replace("/", "_or_") for x in df_answers.columns
    ]

    a_documents = [row.answer for row in df_answers.itertuples()]
    a_metadata = [
        {"answer_quality": row.answer_quality} for row in df_answers.itertuples()
    ]
    a_ids = ["a_id" + str(row.Index) for row in df_answers.itertuples()]

    a_collection.add(documents=a_documents, ids=a_ids, metadatas=a_metadata)
    print("a_collection added")
    return None


def delete_collection_from_vector_db(vdb_path: str, collection_name: str) -> None:
    """Deletes a particular collection from the persistent ChromaDB instance.

    Args:
        vdb_path (str): Path of the persistent ChromaDB instance.
        collection_name (str): Name of the collection to be deleted.
    """
    chroma_client = chromadb.PersistentClient(path=vdb_path)
    chroma_client.delete_collection(collection_name)
    return None


def list_collections_from_vector_db(vdb_path: str) -> None:
    """Lists all the available collections from the persistent ChromaDB instance.

    Args:
        vdb_path (str): Path of the persistent ChromaDB instance.
    """
    chroma_client = chromadb.PersistentClient(path=vdb_path)
    print(chroma_client.list_collections())


def get_collection_from_vector_db(
    vdb_path: str, collection_name: str
) -> chromadb.Collection:
    """Fetches a particular ChromaDB collection object from the persistent ChromaDB instance.

    Args:
        vdb_path (str): Path of the persistent ChromaDB instance.
        collection_name (str): Name of the collection which needs to be retrieved.
    """
    load_dotenv(find_dotenv())
    chroma_client = chromadb.PersistentClient(path=vdb_path)

    huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
        api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    collection = chroma_client.get_collection(
        name=collection_name, embedding_function=huggingface_ef
    )

    return collection
