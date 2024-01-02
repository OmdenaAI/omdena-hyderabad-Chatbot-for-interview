import pandas as pd
import chromadb
import re
from chatbot_functionalities.llms import llm_inference


def generate_questions(
    position: str, candidate_profile: str, question_collection: chromadb.Collection,
    question_order_df: pd.DataFrame
) -> pd.DataFrame:
    """This function will generate a set of relevant questions, given the candidate's position of choosing and their profile.

    Under the hood, it uses semantic search to extract the relevant questions from a vector database containing the
    embeddings of the question bank gathered as part of the project.

    If a semantic search match is not found based on the position or candidate profile, then an LLM will be used
    to generate a question for that particular interview phase.

    Args:
        position (str): Position of the candidate for which the interview is taking place.
        candidate_profile (str): Description of the profile of the candidate.

    Returns:
        pd.DataFrame: Pandas dataframe containing a list of all relevant questions generated, along with the interview phase and candidate profile.
    """
    
    # Trimming column names
    question_order_df.columns = question_order_df.columns.str.strip()

    # Instantiate an empty pandas DataFrame.
    question_df = pd.DataFrame(columns=["question", "interview_phase", "position", "answer", "ratings", "feedback"])

    # Instantiate empty lists for questions and interview phases. These will become columns in the dataframe at the end.
    questions_list = []
    interview_phase_list = []
    
    for _index, question_obj in question_order_df.iterrows():
        position = question_obj['Job Position']
        interview_phase = question_obj['Interview Phase']
        
        n_llm_results = int(question_obj['Number of LLM Generation questions'])
        n_results = (int(question_obj['Number of questions for this interview phase']) - n_llm_results)
        result_doc = []
        
        if n_results > 0:
            result = question_collection.query(
                query_texts=[candidate_profile],
                where={
                    "$and": [
                        {"position": {"$eq": position.strip()}},
                        {"interview_phase": {"$eq": interview_phase.strip()}},
                    ]
                },
                n_results=n_results,
            )
            result_doc = result['documents'][0]
            result_doc_count = len(result_doc)
            
            if result_doc_count > 0:
                questions_list.extend(result_doc)
                interview_phase_list.extend([interview_phase] * result_doc_count)
        
        n_results = n_llm_results - (n_results - len(result_doc))
        
        if n_results > 0:
            intro_template = """Assume you are an expert interviewer, interviewing a candidate. You have the following information:
            Position applying for : {position}
            Candidate profile summary : {candidate_profile}.
            Using the above information, generate {n_results} {interview_phase} question/questions which can help start off the interview. Please provide questions that are highly relevant for the job position only. Don't ask irrelevant questions."""

            intro_ques_llm = llm_inference(
                model_type="huggingface",
                input_variables_list=[position, candidate_profile, n_results, interview_phase],
                prompt_template=intro_template,
                hf_repo_id="tiiuae/falcon-7b-instruct",
                temperature=0.1,
                max_length=64,
                interview_phase=interview_phase
            )
            
            # Using list comprehension to filter out empty strings
            intro_ques_llm_list = [x for x in intro_ques_llm.split("\n") if x != ""]
            # Replace pattern: number followed by a period and space
            pattern = re.compile(r"^\d+\.\s")
            # Replace the specified pattern with an empty string for each element in the list
            intro_ques_llm_list = [re.sub(pattern, "", x) for x in intro_ques_llm_list]

            questions_list.extend(intro_ques_llm_list)
            interview_phase_list.extend([interview_phase] * len(intro_ques_llm_list))
            

    # Add lists as columns to the Dataframe.
    question_df["question"] = questions_list
    question_df["interview_phase"] = interview_phase_list
    question_df["position"] = [position] * len(questions_list)
    
    print("Question generation complete...\n")
    print(question_df)
    
    return question_df
