import os
import json
import csv
import ast
import tiktoken
import streamlit as st
from loguru import logger

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate 
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma  # Chroma를 임포트합니다

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# code related logo
Hyundai_logo = "images/Hyundai_logo.png"
horizontal_logo = "images/Hyundai_logo_horizen.png"

# 여기다가 pChunk의 content와 ID 경로 설정해 주세요!!!
filepath_pChunk_content = "/home/r22000245/fullDocument_pChunk.json"
filepath_pChunk_IDs = "/home/r22000245/fullDocument_mapping.json"

def main():
    st.set_page_config(
        page_title="Hyundai Motor Company - Motor Vehicle Law ",
        page_icon=Hyundai_logo
    )

    st.title("_:blue[Hyundai Motor]_ - Motor Vehicle Law Data :blue[QA Chatbot] :scales:")
    st.markdown("Hyundai Motor Company & Handong Grobal University")

    # sidebar
    st.html(
        """
    <style>
    [data-testid="stSidebarContent"] {
        color: white;
        background-color: #AACAE6;
    }
    </style>
    """
    )
    st.logo(
        horizontal_logo,
        icon_image=Hyundai_logo
    )
    st.sidebar.markdown("Place your legal documents in the space in the sidebar. Enter your OpenAI API Key below it and press Process!")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "vectorstore" not in st.session_state:
        # Chroma 벡터스토어를 처음 로드할 때 session_state에 저장
        st.session_state.vectorstore = load_vectorstore('db/chroma/cchunk')
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Start chatting")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        st.session_state.conversation = get_conversation_chain(openai_api_key)
        st.session_state.processComplete = True
        
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "color": "#002c5f", "content": "Hi! If you have any questions about a given legal document, feel free to ask!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("Please enter your question."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            if chain is not None:
                with st.spinner("Thinking..."):
                    
                    cChunk_IDs = get_ID_pChunk(st.session_state.vectorstore, query) # cChunk ID 획득
                    st.write("Retrieved cChunk_IDs:", cChunk_IDs)
                    if not cChunk_IDs:
                        st.write("No cChunk IDs found. Check your vector store or query.")
                    pChunk_infos = get_pChunk(filepath_pChunk_content, filepath_pChunk_IDs, cChunk_IDs) # pChunk content 획득
                    

                    # LLMChain을 실행하여 결과를 받음
                    result = chain.run({"infos": pChunk_infos, "query": query})
                    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append({"role": "user", "content": query})
                    st.session_state.chat_history.append({"role": "assistant", "content": result})

                    response = result  # chain.run()에서 반환된 문자열을 그대로 사용
                    st.markdown(response)

                                        # 참고 문서 확인 (pChunk 내용과 관련된 문서들)
                    with st.expander("참고 문서 확인"):
                        st.markdown(f"Source: Parent Chunk", help=pChunk_infos)
                        st.markdown(pChunk_infos) 
                    # 대화 내용을 세션에 저장
                    st.session_state.messages.append({"role": "assistant", "content": response})


def get_ID_pChunk(vector_db, query, k=3):
    IDs_matched = []
    cChunks_matched = vector_db.similarity_search(query)

    for rank in range(k):
        cChunk_byJSON = json.loads(cChunks_matched[rank].page_content)
        ID_cChunk = cChunk_byJSON["ID"]
        IDs_matched.append(ID_cChunk)  
    return(IDs_matched)
    
def get_pChunk(pChunk_dbPath, pChunk_IDsPath, cChunk_IDs):
    # Parent Chunk 내용 로드
    with open(pChunk_dbPath, 'r') as infile:
        parent_chunks = json.load(infile)

    # Child-Parent 매핑 로드
    with open(pChunk_IDsPath, 'r') as id_file:
        child_to_parent_map = json.load(id_file)

    set_target_pChunk = ""
    for cChunk_ID in cChunk_IDs:
        # Child Chunk에 해당하는 Parent Chunk ID 찾기
        parent_chunk_id = None
        for parent, children in child_to_parent_map.items():
            if cChunk_ID in children:
                parent_chunk_id = parent
                break
        
        # Parent Chunk가 있으면 그 내용을 추가
        if parent_chunk_id and parent_chunk_id in parent_chunks:
            content_target_pChunk = parent_chunks[parent_chunk_id]
            set_target_pChunk += str(content_target_pChunk) + "\n\n"

    # pChunk_infos가 제대로 출력되는지 로그 출력
    st.write("Final pChunk_infos:", set_target_pChunk)
    return set_target_pChunk
    
def apply_prompt_template():
    system_prompt_template = PromptTemplate(
        input_variables=["infos","question"],
        template=""""
        Answer the following questions as best you can. 
        Be sure to base your answer on the information provided. Don't say you don't know, and be sure to explain why 
        The information provided is in the form of a Json file. You have access to the following informations:
        
        {infos}
        
        Begin!
        
        Question: {query}
        
        """)
    return system_prompt_template

def load_vectorstore(db_path):
    # Hugging Face 임베딩 모델 로드 (임베딩 모델 정보는 저장된 벡터스토어와 동일해야 함)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 저장된 Chroma 벡터스토어 로드
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return vectorstore

def get_conversation_chain(openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4o-mini-2024-07-18', temperature=0)

    conversation_chain = LLMChain(
        llm=llm,
        prompt=apply_prompt_template(),
        verbose=True,
    )
    return conversation_chain

if __name__ == '__main__':
    main()
