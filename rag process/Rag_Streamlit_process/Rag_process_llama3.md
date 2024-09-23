# Rag_process_llama3

https://huggingface.co/akjindal53244/Llama-3.1-Storm-8B모델을  사용함

```jsx
import os
import streamlit as st
from loguru import logger

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# code related logo
Hyundai_logo = "images/Hyundai_logo.png"
horizontal_logo = "images/Hyundai_logo_horizen.png"

def main():
    st.set_page_config(
        page_title="Hyundai Motor Company - Motor Vehicle Law ",
        page_icon=Hyundai_logo
    )

    st.title("_:blue[Hyundai Motor]_ - Motor Vehicle Law Data :blue[QA Chatbot] :scales:")
    st.markdown("Hyundai Motor Company & Handong Grobal University")
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

    with st.sidebar:
        process = st.button("Start Chatting")

    if process:
        # FAISS 벡터스토어 로드
        vectorstore = load_vectorstore('./db/faiss')

        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            if chain is not None:
                with st.spinner("Thinking..."):
                    result = chain({"question": query})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    source_documents = result['source_documents']

                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:
                            st.markdown(doc.metadata['source'], help=doc.page_content)

                    st.session_state.messages.append({"role": "assistant", "content": response})

def load_vectorstore(db_path):
    # Hugging Face 임베딩 모델 로드 (임베딩 모델 정보는 저장된 벡터스토어와 동일해야 함)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 저장된 FAISS 벡터스토어 로드
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def get_conversation_chain(vectorstore):
    # Hugging Face 모델 로드
    model_name = "akjindal53244/Llama-3.1-Storm-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 모델 파이프라인 생성
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0
    )

    # langchain의 HuggingFacePipeline을 사용하여 LLM 생성
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # ConversationalRetrievalChain 생성
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

if __name__ == '__main__':
    main()
```

코드 전문

기존 db를 생성하고 불러오는 부분과, streamlit의 디자인적인 요소, 문서 임베딩 부분 전부 동일하며 답변을 생성하는 최종 llm만 gpt-4o-mini-2024-07-18에서 akjindal53244/Llama-3.1-Storm-8B로 오픈소스 모델로 바뀌었음

```jsx
from langchain.llms import HuggingFacePipeline
```

허깅페이스의 파이프라인으로 답변을 쉽게 불러오는 것이 가능함

```jsx

def get_conversation_chain(vectorstore):
    # Hugging Face 모델 로드
    model_name = "akjindal53244/Llama-3.1-Storm-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 모델 파이프라인 생성
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0
    )

    # langchain의 HuggingFacePipeline을 사용하여 LLM 생성
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # ConversationalRetrievalChain 생성
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
```

이전과는 바뀐 부분인 get_conversation_chain함수부분 우선 모델을 불러오는 작업부터 시작함

```jsx
# Hugging Face 모델 로드
    model_name = "akjindal53244/Llama-3.1-Storm-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
```

LLama 3.1 8b모델에 대해 추가적인 튜닝을 시킨 모델로서, 자세한 설명은 https://huggingface.co/akjindal53244/Llama-3.1-Storm-8B에서 확인 가능

모델에 대해 사전학습된→ 허깅페이스 필드에 올라간 모델을 불러오며 이를 model에 저장함

또한 모델에서 제공되는 토크나이저도 불러옴

```jsx
# 모델 파이프라인 생성
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0
    )
```

허깅페이스의 파이프라인 함수를 이용하여, 모델과 토크나이저를 불러오고, 텍스트 생성을 위한 준비를 함, 이때 max_length를 512로, temperature를 0으로 설정 max_length는 생성되는 텍스트의 최대 길이를 뜻하며, temperature는 생성된 답변의 랜덤성을 뜻함

```jsx
# langchain의 HuggingFacePipeline을 사용하여 LLM 생성
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # ConversationalRetrievalChain 생성
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
```

허깅페이스의 파이프라인을 사용하여 모델을 생성하여 llm에 전달하고,

conversation Chain을 생성함.

이후는 모델만 허깅페이스의 모델로 바뀔 뿐, 기존의 rag_process와 동일함

이상의 코드는 질문에 대해 db로부터 연관성이 높은 context를 찾고, 질문과 context를 함께 허깅페이스의 모델에 전달하여 최종적인 답안을 생성함.