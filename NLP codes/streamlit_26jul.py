
import streamlit as st

import tempfile
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.chains.conversation.memory import ConversationBufferWindowMemory,ConversationBufferMemory
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import os
from langchain.llms import OpenAI
import openai
from dotenv import load_dotenv
load_dotenv()

# user_api_key = st.sidebar.text_input(
#     label="Your OpenAI API key :point_down:",
#     placeholder="Paste your openAI API key, sk-",
#     type="password")


# initialise API key.
api_key = os.getenv("OPENAI_API_KEY")
user_api_key = api_key

# Upload file .
uploaded_file = st.sidebar.file_uploader(label="Upload your dummy csv file here.", type="csv")
if uploaded_file :

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ','})
    data = loader.load()

# Tokenization.

    tokenizer = tiktoken.get_encoding('cl100k_base')
    tiktoken.encoding_for_model('gpt-3.5-turbo')
    def tiktoken_len(text):
        tokens=tokenizer.encode(
        text,
        disallowed_special=()
        )
        return len(tokens)
    token_counts=[tiktoken_len(doc.page_content) for doc in data]

#Chunking data.

    text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    separators=['\n\n','\n',' ','']
    )       

    data=text_splitter.split_documents(data)


# Vector Embedding.

    embeddings = OpenAIEmbeddings()# default text davinchi .
    vectorstore = FAISS.from_documents(data, embeddings)

# Initialise llm.
    llm = ChatOpenAI(
        temperature=0.0,
        model_name="gpt-3.5-turbo",
        openai_api_key=user_api_key)

    # llm = OpenAI(
    #     # temperature=0.0,
    #     # model_name="gpt-3.5-turbo",
    #     openai_api_key=user_api_key)


#Initialise Memory

    memory = ConversationBufferWindowMemory(
        llm=llm,
        k=10,
        # output_key='answer',
        memory_key='chat_history',
        return_messages=True)

# Initialise retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 150, "include_metadata": True})

# Conversation Retrival Chain.
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        # prompt=prompt,
        chain_type="stuff",
        retriever=retriever,

        verbose=True)

#Conversation_chat code.

def conversational_chat(query):
        result = chain({"question": query,
        "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
##########################
# if 'buffer_memory' not in st.session_state:
#     st.session_state.buffer_memory=ConversationBufferWindowMemory(k=10,return_messages=True)
########################
if 'history' not in st.session_state:
        st.session_state['history'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything from csv provided"]
if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey !"]
#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()
with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
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