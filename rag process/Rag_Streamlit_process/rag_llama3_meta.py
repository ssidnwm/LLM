import os
import streamlit as st
from loguru import logger

import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from transformers import BitsAndBytesConfig
from langchain.prompts import PromptTemplate

# code related logo
Hyundai_logo = "images/Hyundai_logo.png"
horizontal_logo = "images/Hyundai_logo_horizen.png"

def postprocess_result(result):
    # 검색 결과에서 불필요한 문서 정보는 제거하고 순수한 답변만 반환
    if 'source_documents' in result:
        del result['source_documents']  # 문서 관련 정보는 답변에서 제외
    
    answer = result.get('answer', '답변을 생성할 수 없습니다.')  # 답변만 추출
    return answer

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
                    formatted_query = apply_prompt_template(query)
                    result = chain({"question": formatted_query})
                    response = postprocess_result(result)

                    source_documents = result.get('source_documents', [])
                    st.write(f"검색된 문서 수: {len(source_documents)}")

                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:
                            st.markdown(doc.metadata['source'], help=doc.page_content)
                    
                    
                    check_and_clear_gpu_memory()
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
def check_and_clear_gpu_memory():
    st.write(f"GPU 메모리 할당: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    st.write(f"GPU 메모리 예약: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    torch.cuda.synchronize()  # CPU와 GPU 작업 동기화
    torch.cuda.empty_cache()  # 캐시 정리
    st.write("GPU 캐시 정리 완료.")
    st.write(f"정리 후 GPU 메모리 할당: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    st.write(f"정리 후 GPU 메모리 예약: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def apply_prompt_template(query):
    system_prompt_template = PromptTemplate(
        input_variables=["question"],
        template="All answers are based on the documentation,If you don't answer, explain why You should never answer I DON'T KNOW. and The document comes from JSON. Answer concisely and do not include document context\nQuestion: {question}"
    )
    return system_prompt_template.format(question=query)
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
    # Hugging Face 모델 로드 및 4bit 양자화 설정
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # 4bit 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16  # Float16 설정으로 성능 최적화
    )

    # 토크나이저 및 양자화된 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,  # 4bit 양자화 적용
        device_map="auto"  # 자동으로 GPU에 할당
    )
    model.gradient_checkpointing_enable()

    # 모델 파이프라인 생성
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        truncation=True,
        temperature=0.1
    )

    # HuggingFacePipeline을 langchain에 적용하여 LLM 생성
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # ConversationalRetrievalChain 생성
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='similarity', verbose=True, k=1),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    
    st.write("양자화된 대화 체인이 생성되었습니다.")
    return conversation_chain

if __name__ == '__main__':
    main()