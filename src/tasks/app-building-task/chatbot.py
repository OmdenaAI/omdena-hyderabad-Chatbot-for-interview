# dependencies
import logging
import streamlit as st
import speech_recognition as sr
from pathlib import Path
from chatbot_functionalities.generate_questions import generate_questions
from chatbot_functionalities.vectordb_operations import get_collection_from_vector_db
import time

# enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple-chatbot")


# function to initialize web app for the first time
def initialize_app():
    # do one time processing here
    st.session_state.p01_show_chat_section = False
    st.session_state.p01_profile_details_taken = False
    st.session_state.p01_questions_generated = False
    st.session_state.p01_chat_history = []
    st.session_state.p01_start_recording_disabled = False
    # set the flag that indiciates initialization is done
    st.session_state.p01_init_complete = True
    # Fetch chromadb object
    st.session_state.p01_collection_object = get_collection_from_vector_db(
        (Path.cwd() / "src" / "data" / "chromadb").__str__(), "question_collection"
    )


# function(s) to process user interactions
def process_start_interview():
    # call question generation function

    # store the questions in a dataframe in session state

    st.session_state.p01_show_chat_section = True
    st.session_state.p01_profile_details_taken = False
    st.session_state.p01_chat_history = []


def p01_process_user_query():
    user_message = dict(role="user", content=st.session_state.p01_user_query)
    st.session_state.p01_chat_history.append(user_message)

    bot_message = dict(
        role="assistant", content="Thank you. Please give another text input"
    )
    st.session_state.p01_chat_history.append(bot_message)


def record_audio():
    # st.session_state.p01_start_recording_disabled = True
    transcription = ""
    r = sr.Recognizer()
    with st.spinner("Recording in progress"):
        with sr.Microphone() as source:
            # with st.empty():
            #     st.write("Start with your answer")

            audio = r.listen(source, timeout=5, phrase_time_limit=20)
            try:
                text = r.recognize_google(audio)
                transcription += text + " "

                st.session_state.p01_transcription = transcription

                # append the utterance from the candidate to chat history
                st.session_state.p01_chat_history.append(
                    dict(role="user", content=transcription)
                )
            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError as e:
                st.error(
                    f"Could not request results from speech recognition service; {e}"
                )


def get_profile_details():
    # Ask the candidate for their profile:
    # profile_message = dict(
    #     role="assistant",
    #     content="Please tell us about your previous work experience and educational details which may be relevant for the position you selected.",
    # )
    # st.session_state.p01_chat_history.append(profile_message)
    with st.chat_message("assistant"):
        st.markdown(
            """Please tell us briefly about your previous work experience and educational details which may be relevant for the job position you selected.
            \nAll responses will be captured through the microphone available on your device. Ensure that the microphone is working and configured correctly.
            """
        )

    if st.button(
        label="Start Recording Profile Details",
        type="primary",
        # on_click=record_audio(),
        # disabled=st.session_state.p01_start_recording_disabled,
        key="p01_start_recording_profile_details",
    ):
        record_audio()
        st.session_state.p01_profile_details_taken = True
        st.success("Thank you for providing your details ✅")
        st.write(
            f'The transcribed text is as follows: \n\n {st.session_state.p01_chat_history[0]["content"]}'
        )

        # while st.spinner("Generating relevant questions!"):
        st.session_state.p01_questions_df = generate_questions(
            st.session_state.p01_job_position,
            st.session_state.p01_chat_history[0]["content"],
            st.session_state.p01_collection_object,
        )
        st.session_state.p01_questions_generated = True
        # time.sleep(2)
        st.write(st.session_state.p01_questions_df)


# function for rendering the main web application
def run_web_app():
    # setup page title (this shows up as window title as well)
    st.set_page_config(page_title="Simple Chatbot")

    # call initialization function (only for the first time)
    if "p01_init_complete" not in st.session_state:
        initialize_app()

    # setup sidebar
    st.sidebar.title("Candidate Profile")

    candidate_name = st.sidebar.text_input(
        label="Candidate Name",
        placeholder="Enter your name",
        key="p01_candidate_name",
    )

    job_position_options = [
        "Customer Service Representative",
        "Sales Manager",
        "Marketing Manager ",
        "Nurse",
        "Medical Assistance",
    ]
    job_position = st.sidebar.selectbox(
        label="Job Position",
        placeholder="Select a job position",
        options=job_position_options,
        key="p01_job_position",
    )

    start_interview_button = st.sidebar.button(
        label="Start Interview",
        on_click=process_start_interview,
        key="p01_start_interview",
    )

    # render chat section
    if st.session_state.p01_show_chat_section:
        # set page heading (this is a title for the main section of the app)
        st.markdown(
            "<h3 style='color: orange;'>Interview Prep Chatbot</h3>",
            unsafe_allow_html=True,
        )

        if st.session_state.p01_profile_details_taken == False:
            get_profile_details()
        else:
            # button to start recording
            if st.button(
                label="Start Recording",
                type="primary",
                # on_click=record_audio(),
                # disabled=st.session_state.p01_start_recording_disabled,
                key="p01_start_recording",
            ):
                record_audio()

            # loop through chat history and show the messages if they exist
            for message in st.session_state.p01_chat_history[::-1]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # display the captured text
            if "p01_transcription" in st.session_state:
                pass
                # st.write(f"Text from STT: {st.session_state.p01_transcription}")


# call the function to render the main web application
if __name__ == "__main__":
    run_web_app()
