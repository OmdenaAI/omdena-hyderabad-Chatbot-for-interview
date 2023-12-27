from llms import llm_inference
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser


def evaluate_answer(
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