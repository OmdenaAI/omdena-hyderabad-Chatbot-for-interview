import pandas as pd
import numpy as np
import chromadb
from chatbot_functionalities.llms import llm_inference
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from typing import List
from langchain.prompts import PromptTemplate
from langchain import FewShotPromptTemplate
from pathlib import Path

def evaluate_answer(
    question: str,
    answer: str,
    position: str, 
    questions_collection: chromadb.Collection, 
    ):
    """Call HuggingFace/OpenAI model for inference

    Given a question,answer, and position , this function calls the relevant
    API to fetch LLM inference results.

    Args:
        question: The generated question from our database
        answer: answer given by the candidate
        position: job position that the candidate applying for


    Returns:
        Rating: rating for candidate's answer .
        qualitative_feedback : based on the candidate's answer and the given rating.

    HuggingFace repo_id example:
        - mistralai/Mistral-7B-Instruct-v0.1

    """
    # read the collected data from excel file
    excel_file_path = (Path.cwd() / "src" / "data" / "processed" / "combined_dataset.xlsx").__str__()
    collected_q_a_df = pd.read_excel(excel_file_path, sheet_name='combined')
    collected_q_a_df.columns = [
        x.replace(" ", "_").lower().replace("/", "_or_") for x in collected_q_a_df.columns
    ]

    # fetch good, average, poor examples for the given question and pass to llm (few shot learning)
    matching_questions = \
        questions_collection.query(
            query_texts=[question],
            where={"position": {"$eq": position}},
            n_results=3,
        )

    # fetch examples from collected data
    examples = []
    ratings_scope = ['Good', 'Average', 'Poor']
    for rating in ratings_scope:
        matching_rows = \
            collected_q_a_df\
                .query(f"position_or_role == '{position}'")\
                .query(f"question.isin({matching_questions['documents'][0]})")\
                .query(f"answer_quality == '{rating}'")\
                [['question', 'answer']]
        if matching_rows.shape[0] > 0:
            examples.append(
                {
                    'position': position, 
                    'question': question, 
                    'answer': matching_rows.answer.iloc[0], 
                    'Rating': rating, 
                }
            )

    #set up example_template
    example_template = """
        position: {position} .\
        question: {question} \
        answer: {answer}.\
        Rating:{Rating}.\
        """

    #set up example_prompt
    example_prompt = \
        PromptTemplate(
            input_variables=["position", "question", "answer","Rating"], 
            template=example_template, 
            )

    # Set up prefix prompt
    prefix = """
        ### instruction: you are an experienced interviewer.\
        You are interviewing a candidate for the position of {position} .\
        You are tasked to rate an answer provided by the candidate. You should provide a categorical Rating and qualitative feedback.\
        The categorical rating should be one of the following values: Good, average, or  Poor.\
        the qualitative feedback should provide sufficient details to justify the categorical rating.\
        The position and the question asked to the candidate and the answer given by the candidate are  given below.\
        also some examples are given below.\
        """
    suffix = """
        position : {position} .\
        question : {question} \
        answer : {answer}.\
        qualitative_feedback:
    """

    few_shot_prompt_template = \
        FewShotPromptTemplate(
            examples=examples, 
            example_prompt=example_prompt, 
            prefix=prefix, 
            suffix=suffix, 
            input_variables=["position", "question", "answer"], 
            example_separator="\\\n\\\n", 
            )

    # send prompt to LLM using the common function
    response = \
        llm_inference(
            model_type="huggingface",
            input_variables_list=[ position, question, answer],
            prompt_template=few_shot_prompt_template,
            hf_repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            inference_type = "evaluation",
            temperature=0.1,
            max_length=32000, 
            )

    return 'None', response

def evaluate_answer_obsolete(
    question: str,
    answer: str,
    position: str,
):
    """Call HuggingFace/OpenAI model for inference

    Given a question,answer, and position , this function calls the relevant
    API to fetch LLM inference results.

    Args:
        question: The generated question from our database
        answer: answer given by the candidate
        position: job position that the candidate applying for
        
    Returns:
        Rating: rating for candidate's answer .
        qualitative_feedback : based on the candidate's answer and the given rating.

    HuggingFace repo_id example:
        - mistralai/Mistral-7B-Instruct-v0.1

    """
    # Set up prompt and chain
    prompt = (
        """### instruction: you are an experienced interviewer.\
         You are interviewing a candidate for the position of {position} .\
         You are tasked to rate an answer provided by the candidate. You should provide a categorical rating and qualitative_feedback.\
          The categorical rating should be one of the following values: Good, average, or  Poor.\
            the qualitative_feedback should provide sufficient details to justify the categorical rating.\
            the format instructions of the output and the question asked to the candidate and the answer given by the candidate are  given below.\
            ### format instruction: {format_instructions}.\
            ### question:{question}.\
            ### answer:{answer}.\
            ### Rating:
            """
    )

    # Define Rating and feedback schema
    Rating_schema = ResponseSchema(name="Rating",
                                   description="it was the categorical value for the answer given by the candidate and this value could be poor, average or good. \
                                       ,the categorical value given by you as an experienced interviewer. \
                                      after asking a candidate a question related to the position he is applying for")
    
      #defining feedback schema
    qualitative_feedback_schema = ResponseSchema(name="qualitative_feedback",
                                                  description="the qualitative feedback is the sufficient details  which is given by you as an Experienced interviewer. \
                                                      the qualitative feedback is given after asking the candidate a question related to the position he is applying for, \
                                                       and the candidate provided his answer. \
                                                        the qualitative feedback should provide sufficient details to justify the categorical rating ")
    # Stack the two schemas
    response_schemas = [Rating_schema, qualitative_feedback_schema]

    # Parsing the output
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Extracting format instructions
    format_instructions = output_parser.get_format_instructions()

    # apply evaluation using hugging inference API
    response = llm_inference(
                model_type="huggingface",
                input_variables_list=[position, format_instructions, question, answer],
                prompt_template=prompt,
                hf_repo_id="mistralai/Mistral-7B-Instruct-v0.1",
                inference_type = "evaluation",
                temperature=0.1,
                max_length=2024,
            )

    # Output dictionary having two keys "Rating" and "qualitative_feedback"
    output_dict = output_parser.parse(response)

    return output_dict["Rating"] , output_dict["qualitative_feedback"]

def evaluate_all_answers(
    interview_history: pd.DataFrame, 
    questions_collection: chromadb.Collection, 
    ):
    """Evaluates all answers from interview history and obtains categorical rating 
    as well as qualitative feedback.
    """
    # interview history contains all the questions asked in the mock interview 
    # and the answers provided by the candidate
    # process each pair (question & answer) one by one and do evaluation
    # columns=["question", "interview_phase", "position", "answer", "ratings", "feedback"]
    for index, row in interview_history.iterrows():
        # get rating and qualitative feedback for a single question - answer pair
        rating, feedback = \
            evaluate_answer(
                question=row.question, 
                answer=row.answer, 
                position=row.position, 
                questions_collection=questions_collection,
                )
        
        # update the rating and feedback obtained from llm into the data frame
        interview_history.loc[index, ['ratings', 'feedback']] = [rating, feedback]

def get_ratings_for_answers(df: pd.DataFrame):
    arr_random = np.random.default_rng().uniform(low=0,high=1,size=[df.shape[0],1])
    df.loc[:, 'ratings'] = arr_random

def get_feedback_for_answers(df: pd.DataFrame):
    df.loc[:, 'feedback'] = 'Some Random Feedback'

def get_overall_feedback():
    return 'Some Overall Feedback'