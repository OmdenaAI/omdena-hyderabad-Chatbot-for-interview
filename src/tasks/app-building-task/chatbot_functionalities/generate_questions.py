import pandas as pd
import chromadb
import re
from chatbot_functionalities.llms import llm_inference


def generate_questions(
    position: str, candidate_profile: str, question_collection: chromadb.Collection
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

    # Instantiate an empty pandas DataFrame.
    question_df = pd.DataFrame(columns=["question", "interview_phase", "position"])

    # Instantiate empty lists for questions and interview phases. These will become columns in the dataframe at the end.
    questions_list = []
    interview_phase_list = []

    # Uncomment the below 2 lines if you want to test with custom values.
    # position = "Nurse"
    # candidate_profile = "Dedicated and compassionate Registered Nurse with a diverse background in healthcare. Holds a [Degree or Certification] in Nursing from [Institution]. Proven expertise in providing patient-centered care, managing medical records, and collaborating with interdisciplinary teams. Skilled in administering medications, monitoring vital signs, and implementing nursing care plans. Demonstrates strong communication and interpersonal skills, fostering positive relationships with patients, families, and healthcare professionals. Upholds a commitment to continuous learning and professional development. Adept at maintaining a calm and focused demeanor in high-pressure situations. Excited about contributing clinical skills and compassionate care to a dynamic healthcare environment. [Optional: Specify any specializations, such as critical care, pediatrics, or other relevant areas of expertise.]"

    # ------------------------------- #
    # -------INTRODUCTION PHASE------ #
    # ------------------------------- #

    print("Generating questions for introduction phase...\n")
    # Fetch introduction questions using semantic search
    intro_ques_semantic_search = question_collection.query(
        query_texts=[candidate_profile],
        where={
            "$and": [
                {"position": {"$eq": position}},
                {"interview_phase": {"$eq": "Introduction"}},
            ]
        },
        n_results=2,
    )

    # Check if sufficient(2) introduction questions returned by semantic search.
    if len(intro_ques_semantic_search["documents"][0]) != 2:
        num_ques_to_gen = 2 - len(intro_ques_semantic_search["documents"][0])
        intro_template = """Assume you are an expert interviewer, interviewing a candidate. You have the following information:
        Position applying for : {position}
        Candidate profile summary : {candidate_profile}.
        Using the above information, generate {num_ques_to_gen} introductory question/questions which can help start off the interview. Please provide questions that are highly relevant for the job position only. Don't ask irrelevant questions."""

        intro_ques_llm = llm_inference(
            model_type="huggingface",
            input_variables_list=[position, candidate_profile, num_ques_to_gen],
            prompt_template=intro_template,
            hf_repo_id="tiiuae/falcon-7b-instruct",
            temperature=0.1,
            max_length=64,
        )
        # Using list comprehension to filter out empty strings
        intro_ques_llm_list = [x for x in intro_ques_llm.split("\n") if x != ""]
        # Replace pattern: number followed by a period and space
        pattern = re.compile(r"^\d+\.\s")
        # Replace the specified pattern with an empty string for each element in the list
        intro_ques_llm_list = [re.sub(pattern, "", x) for x in intro_ques_llm_list]

        questions_list.extend(intro_ques_llm_list)
        questions_list.extend(intro_ques_semantic_search["documents"][0])
        interview_phase_list.extend(["Introduction"] * 2)
    else:
        questions_list.extend(intro_ques_semantic_search["documents"][0])
        interview_phase_list.extend(["Introduction"] * 2)

    print("Introduction phase question generation complete...\n")

    # ------------------------------- #
    # -----------CORE PHASE---------- #
    # ------------------------------- #

    print("Generating questions for core phase...\n")

    # Fetch core questions using semantic search
    core_ques_semantic_search = question_collection.query(
        query_texts=[candidate_profile],
        where={
            "$and": [
                {"position": {"$eq": position}},
                {"interview_phase": {"$nin": ["Introduction", "Conclusion"]}},
            ]
        },
        n_results=4,
    )

    # Check if sufficient(4) core questions returned by semantic search.
    if len(core_ques_semantic_search["documents"][0]) != 4:
        num_ques_to_gen = 4 - len(core_ques_semantic_search["documents"][0])
        core_template = """Assume you are an expert interviewer, interviewing a candidate. You have the following information:
        Position applying for : {position}
        Candidate profile summary : {candidate_profile}.
        Using the above information, generate {num_ques_to_gen} position specific question/questions which can help start off the interview. Please provide questions that are highly relevant for the job position only. Don't ask irrelevant questions."""

        core_ques_llm = llm_inference(
            model_type="huggingface",
            input_variables_list=[position, candidate_profile, num_ques_to_gen],
            prompt_template=core_template,
            hf_repo_id="tiiuae/falcon-7b-instruct",
            temperature=0.1,
            max_length=64,
        )
        # Using list comprehension to filter out empty strings
        core_ques_llm_list = [x for x in core_ques_llm.split("\n") if x != ""]
        # Replace pattern: number followed by a period and space
        pattern = re.compile(r"^\d+\.\s")
        # Replace the specified pattern with an empty string for each element in the list
        core_ques_llm_list = [re.sub(pattern, "", x) for x in core_ques_llm_list]

        questions_list.extend(core_ques_llm_list)
        interview_phase_list.extend(["Core"] * num_ques_to_gen)
        questions_list.extend(core_ques_semantic_search["documents"][0])
        interview_phase_list.extend(
            [d["interview_phase"] for d in core_ques_semantic_search["metadatas"][0]]
        )
    else:
        questions_list.extend(core_ques_semantic_search["documents"][0])
        interview_phase_list.extend(
            [d["interview_phase"] for d in core_ques_semantic_search["metadatas"][0]]
        )

    print("Core phase question generation complete...\n")

    # Add lists as columns to the Dataframe.
    question_df["question"] = questions_list
    question_df["interview_phase"] = interview_phase_list
    question_df["position"] = [position] * len(questions_list)

    return question_df
