from langchain import FewShotPromptTemplate
from chatbot_functionalities.llms import llm_inference


def evaluate_answer(
    question: str,
    answer: str,
    position: str,
) -> str:
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
     if position == "Customer Service Representative":
    #set up examples
          examples = [
             {
            "position": f"""{position}""",
            "question": """How can you improve a dissatisfied customer's experience?""",
            "answer": """I've found the most successful strategy for turning an unhappy customer into a happy customer is by actively listening to what they're saying. Sometimes, customers just want you to listen to them, and they want to feel like the company cares about them and their opinions. \
                    For example, I once had a customer who got home to find there was only one shoe in their shoebox. They were quite upset, so I let them explain the issue and then I validated their feelings and provided them with a discount on the purchase along with the missing shoe. They left in a much better mood and became a loyal customer.""",
            "Rating" : "Good",
            #"qualitative_feedback": """The candidate's response is rated as 'Good.' The answer not only emphasizes the importance of active listening but also provides a specific and illustrative example to support the strategy. The candidate goes beyond general advice by recounting a real scenario where a customer faced an issue, demonstrating a practical application of the suggested approach. The mention of validating the customer's feelings and offering a discount, along with the missing shoe, shows a proactive and customer-focused problem-solving approach. This response indicates a strong understanding of customer service principles and an ability to apply them effectively in challenging situations, resulting in customer satisfaction and loyalty."""
            },{
            "position": f"""{position}""",
            "question": """How can you improve a dissatisfied customer's experience?""",
            "answer": """I've found the most successful strategy for turning an unhappy customer into a happy customer is by actively listening to what they're saying. Sometimes, customers just want you to listen to them, and they want to feel like the company cares about them and their opinions. """,
            "Rating": "Average",
            #"qualitative_feedback":"""The candidate's response is rated as 'Average.' While the answer acknowledges the importance of active listening, it lacks depth in providing a comprehensive strategy for improving a dissatisfied customer's experience. The candidate briefly mentions the significance of making the customer feel cared for, but there is a lack of specific actions or steps to address and resolve the customer's concerns. A stronger response could have included additional elements such as empathetic communication, prompt issue resolution, and, if applicable, offering appropriate compensation or solutions. The answer, though acknowledging a key aspect, falls short of providing a well-rounded and detailed approach to handling dissatisfied customers."""
            },{
            "position": f"""{position}""",
            "question": """How can you improve a dissatisfied customer's experience?""",
            "answer": "  I was playing a game.",
            "Rating" : "Poor",
            #"qualitative_feedback": """The candidate's response is rated as 'Poor.' The answer provided does not address the question and appears to be irrelevant to the context of improving a dissatisfied customer's experience. It lacks any relevant information or insight into customer service strategies. A strong response should have focused on practical approaches, communication skills, and problem-solving methods to enhance the customer experience. The candidate's answer demonstrates a misunderstanding of the question and an inability to provide a relevant and thoughtful response."""
                }
              ]
     elif position == "Nurse":
    #set up examples
          examples = [
             {
            "position": f"""{position}""",
            "question": """ how do you handle the stress of the job ?""",
            "answer": """I find the best way to handle the stress of the job is through meticulous organization and attention to detail. By making lists and prioritizing what needs to get done throughout my day I find that tasks which might seem overwhelming all at once are much more manageable. This also makes it possible for me to stay calm and remain focused on what needs to get done when unexpected situations arise.""",
            "Rating" : "Good",
            #"qualitative_feedback": """The candidate's response is rated as 'Good.' They provide a well-thought-out and practical approach to handling the stress of the nursing job. The emphasis on meticulous organization, attention to detail, and prioritization through making lists is a strong strategy for managing workload and preventing tasks from becoming overwhelming. The candidate's acknowledgment of the inevitability of unexpected situations and the ability to remain calm and focused in such scenarios demonstrates adaptability and resilience. Overall, the response showcases effective coping mechanisms that align with the demands of a nursing role, indicating a proactive and organized approach to stress management."""
            },{
            "position": f"""{position}""",
            "question": """how do you handle the stress of the job ?""",
            "answer": """I handle stress by focusing on the most important thing the care of the patient. I feel I owe it to my patients to stay calm and focused on them. """,
            "Rating": "Average",
            #"qualitative_feedback":"""The candidate's response is rated as 'Average.' While the answer acknowledges a strategy for handling stress by focusing on patient care, it lacks depth in providing additional coping mechanisms or self-care strategies. A more robust response could have included personal methods for maintaining work-life balance, seeking support from colleagues, or engaging in stress-relief activities outside of work. Additionally, the candidate could have elaborated on how maintaining focus on patient care contributes to their overall stress management. While the answer is acceptable, it falls slightly short of providing a more comprehensive understanding of the candidate's approach to handling stress in the nursing role."""
            },{
            "position": f"""{position}""",
            "question": """ how do you handle the stress of the job ?""",
            "answer": "I like a fast-paced pressure-filled environment that makes my job invigorating.",
            "Rating" : "Poor",
            #"qualitative_feedback": """The candidate's response is rated as 'Poor.' While expressing a preference for a fast-paced and pressure-filled environment can indicate adaptability, the answer lacks depth in addressing how the candidate actively manages and handles stress in the nursing job. A strong response would have included specific strategies or coping mechanisms, such as organization, prioritization, or self-care practices, to demonstrate a proactive approach to stress management. The current answer is vague and does not provide insight into the candidate's ability to handle the inherent stress of the nursing role, which is crucial for the position. A more detailed and focused response would have been more appropriate."""
                }
              ]
     elif position == "Marketing Manager":
    #set up examples
          examples = [
             {
            "position": f"""{position}""",
            "question": """Are you a team player? """,
            "answer": """I am absolutely a team player. My perspective has always been that if my team succeeds, I succeed, and if I succeed, my team succeeds. I think work is a lot more fun when you're sharing your time and energy with people who want to raise each other up.""",
            "Rating" : "Good",
            #"qualitative_feedback": """The candidate's response is rated as 'Good.' They express a positive and collaborative attitude towards teamwork. The candidate emphasizes the mutual success of both individual and team, demonstrating an understanding of the interconnectedness of personal and team achievements. The mention of finding work more enjoyable when sharing time and energy with supportive team members adds a personal touch to the answer. Overall, the response conveys a strong commitment to teamwork and suggests that the candidate values a collaborative work environment, which is a positive trait for a Marketing Manager role."""
            },{
            "position": f"""{position}""",
            "question": """Are rich snippets important for SEO ?""",
            "answer": """"Having rich snippets can help search results stand out and increase the click-through rate. In the long run, it can positively affect page ranking, too.""",
            "Rating": "Average",
            #"qualitative_feedback":"""The candidate's response is rated as 'Average.' While the answer acknowledges the importance of rich snippets for SEO by mentioning that they can help search results stand out and increase click-through rates, it lacks depth in providing a more comprehensive explanation. A stronger response could have delved into the specific types of information that can be included in rich snippets, their impact on user engagement, and how they contribute to a better user experience. Additionally, the candidate could have elaborated on how search engines use rich snippets to understand the content better. The answer, though correct in recognizing the value of rich snippets, falls short of providing a more detailed and insightful response."""
            },{
            "position": f"""{position}""",
            "question": """Can you discuss a time when a marketing campaign didn't perform as expected? How did you handle it, and what did you learn from the experience?""",
            "answer": " I never had a campaign fail on me. All my campaigns were successful.",
            "Rating" : "Poor",
            #"qualitative_feedback": """The candidate's response is rated as 'Poor.' The answer lacks credibility and does not align with the reality of marketing, where not all campaigns are guaranteed to be successful. A more realistic and honest approach would have been to acknowledge that marketing campaigns can face challenges and share a specific instance where a campaign did not perform as expected. This would have provided an opportunity for the candidate to demonstrate problem-solving skills, adaptability, and the ability to learn from setbacks. The lack of humility and the claim that all campaigns were successful suggests a lack of transparency and self-awareness, which are crucial qualities for a Marketing Manager."""
                }
              ]
     elif position == "Sales Manager":
    #set up examples
          examples = [
             {
            "position": f"""{position}""",
            "question": """Why do you want the sales manager position?""",
            "answer": """ I enjoyed what I read about this company and your products. I am ecstatic at the possibility of working for you. I love working with teams and helping to guide them to give it their all every day because that’s what I will do as the sales manager. I appreciate all the rave reviews about your products and want to help get your sales to the next level.\
                          In my previous job, I was promoted to start a new sales team and got to choose team members. I looked at everyone’s personalities, experiences, strengths and weaknesses to create a team that would balance each other. I know I can succeed as the sales manager for this company and want the opportunity to show you how I can help this company reach new heights.""",
            "Rating" : "Good",
            #"qualitative_feedback": """The candidate's response is rated as 'Good.' They provide a well-rounded answer that demonstrates genuine enthusiasm for the company and the sales manager position. The mention of enjoying what they read about the company and its products, along with expressing excitement at the possibility of working there, conveys a positive attitude. The candidate articulates a passion for working with teams and guiding them to excel, aligning with the responsibilities of a sales manager. Additionally, the mention of past success in starting a new sales team and strategically selecting team members showcases relevant experience and leadership skills. The candidate's commitment to contributing to the company's growth and taking it to the next level adds value to their response. Overall, the answer effectively communicates a strong interest in the position and the ability to make meaningful contributions to the sales team."""
            },{
            "position": f"""{position}""",
            "question": """Why do you want the sales manager position?""",
            "answer": """"I enjoyed what I read about this company and your products. I am ecstatic at the possibility of working for you. I love working with teams and helping to guide them to give it their all every day because that’s what I will do as the sales manager. I appreciate all the rave reviews about your products and want to help get your sales to the next level.\
                          """,
            "Rating": "Average",
            #"qualitative_feedback":"""The candidate's response is rated as 'Average.' While expressing excitement about the company and the products, the answer lacks specific details about the candidate's qualifications or experiences that make them suitable for the sales manager position. The mention of loving to work with teams and guide them is positive, but it could be enhanced by providing examples of past successes or leadership experiences in managing sales teams. Additionally, the candidate expresses a desire to help elevate sales but does not offer a clear strategy or insights into how they plan to achieve this goal. A stronger response would include more concrete details about the candidate's skills, experiences, and how they intend to contribute to the company's sales growth."""
            },{
            "position": f"""{position}""",
            "question": """Why do you want the sales manager position?""",
            "answer": " I enjoyed what I read about this company and your products.",
            "Rating" : "Poor",
            #"qualitative_feedback": """The candidate's response is rated as 'Poor.' The answer is overly brief and lacks substance. While expressing enjoyment about the company and its products is positive, it does not provide any meaningful insights into the candidate's qualifications, motivations, or specific reasons for wanting the sales manager position. A strong response would include details about the candidate's relevant skills, experiences, and how they plan to contribute to the success of the sales team. The current answer falls short of demonstrating a genuine interest in the role and does not convey a strong commitment to the position."""
                }
              ]
     #position == "Medical Assistance"
     else  :
    #set up examples
          examples = [
             {
            "position": f"""{position}""",
            "question": """Can you tell me about a time you overcame a difficult situation?""",
            "answer": """ When I was working at the hospital, I communicated with an upset mother who insisted on being in the operating room with her son during his surgery. As this violated hospital rules, I knew I couldn't allow her in the room. Instead of becoming impatient with her, I tried to be empathetic about her situation. I understood she felt scared and didn't know about our safety procedures.\
                          I told her I understood her situation and knew she just wanted the best for her son. Next, I informed her politely of the hospital's policies and why they were in place, emphasizing that following them would help keep her son safe. I even promised to give her hourly updates, which comforted her and increased her trust in the medical team. She thanked me for speaking with her and providing great care for her son.""",
            "Rating" : "Good",
            #"qualitative_feedback": """The candidate's response is rated as 'Good.' They provide a detailed and well-structured example of overcoming a difficult situation in a medical setting. The candidate effectively demonstrates strong communication and empathy skills in dealing with an upset mother. They not only recognized and validated the mother's emotions but also explained the hospital's policies with empathy and understanding. The offer of hourly updates to comfort the mother and build trust in the medical team shows a proactive and patient-focused approach. Overall, the response showcases the candidate's ability to handle challenging situations with empathy, effective communication, and a commitment to patient care."""
            },{
            "position": f"""{position}""",
            "question": """Can you tell me about a time you overcame a difficult situation?""",
            "answer": """"When I was working at the hospital, I communicated with an upset mother who insisted on being in the operating room with her son during his surgery. As this violated hospital rules, I knew I couldn't allow her in the room. Instead of becoming impatient with her, I tried to be empathetic about her situation. I understood she felt scared and didn't know about our safety procedures.
                          """,
            "Rating": "Average",
            #"qualitative_feedback":"""The candidate's response is rated as 'Average.' While they provide a specific example of overcoming a difficult situation in a medical setting, the response lacks some depth. The candidate effectively communicates empathy and understanding towards the upset mother's situation, which is positive. However, the answer could be improved by providing more details about the resolution or outcome of the situation. Offering insights into how the candidate successfully navigated the violation of hospital rules, the mother's reaction to the explanation, or any additional steps taken would have added more substance to the response. Overall, while the answer is acceptable, there is room for enhancement in providing a more comprehensive account of the situation."""
            },{
            "position": f"""{position}""",
            "question": """Can you tell me about a time you overcame a difficult situation?""",
            "answer": " When I was working at the hospital, I communicated with an upset mother who insisted on being in the operating room with her son during his surgery.",
            "Rating" : "Poor",
            #"qualitative_feedback": """The candidate's response is rated as 'Poor.' While the candidate starts to describe a challenging situation involving an upset mother, the answer is incomplete and lacks necessary details. The response does not provide information on how the candidate handled the situation, what actions were taken, or the resolution of the problem. A strong answer to this question should include specific actions taken, the candidate's thought process, and the positive outcome or lessons learned from overcoming the difficult situation. In its current form, the response lacks the depth and completeness needed to showcase the candidate's problem-solving and interpersonal skills effectively."""
                }
              ]
    #set up example_template
     example_template = """
          position: {position} .\
          question: {question} \
          answer: {answer}.\
          Rating:{Rating}.\
         """
         #qualitative_feedback:{qualitative_feedback}.\

    #set up example_prompt
     example_prompt = PromptTemplate(
      input_variables=["position", "question", "answer","Rating"],
      template=example_template
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

     few_shot_prompt_template = FewShotPromptTemplate(
      examples=examples,
      example_prompt=example_prompt,
      prefix=prefix,
      suffix=suffix,
      input_variables=["position", "question", "answer"],
      example_separator="\\\n\\\n" )


    # send prompt to LLM using the common function
     response = llm_inference(
                model_type="huggingface",
                input_variables_list=[ position, question, answer],
                prompt_template=few_shot_prompt_template,
                hf_repo_id="mistralai/Mistral-7B-Instruct-v0.1",
                inference_type = "evaluation",
                temperature=0.1,
                max_length=32000,
            )



     return response