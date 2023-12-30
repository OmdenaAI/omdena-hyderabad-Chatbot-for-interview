from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message
from langchain.llms import HuggingFaceHub
import torch



user_token = st.sidebar.text_input(
    label="#### your hugging face authentication's token 👇",
    placeholder="Paste your HF koken here",
    type="password")


if user_token:


    path = "/content/drive/MyDrive/chroma_db/chroma_db"

    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    vectors = Chroma(persist_directory = path, embedding_function = embeddings)


    hf_repo_id = 'mistralai/Mistral-7B-Instruct-v0.1'


    llm = HuggingFaceHub(
            repo_id=hf_repo_id,
            model_kwargs={"temperature": 0.2, "max_length": 32000}, huggingfacehub_api_token = user_token
        )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retrieval = vectors.as_retriever(k=3)


    def conversational_chat(query):

        chain = ConversationalRetrievalChain.from_llm(llm = llm,

                                                  retriever=retrieval,
                                                  memory = memory ,
                                                  chain_type="map_reduce"
                                                  )

        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))

        del chain ; torch.cuda.empty_cache()


        return result["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me any question that you expect in your interview " +  " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):

            user_input = st.text_input("Query:", placeholder=" ask you question here (:", key='input')
            submit_button = st.form_submit_button(label='Send')


        if submit_button and user_input:


            output = conversational_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
