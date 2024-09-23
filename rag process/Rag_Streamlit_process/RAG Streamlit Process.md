# RAG Streamlit Process

## CreateDB.py

```jsx
import os
import tiktoken
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

필요한 라이브러리 임포트, 사용된 데이터의 포멧이 jsonl이기 때문에 langchain.document_loaders에서 jsonloader사용

```jsx
# OpenAI API Key 설정
os.environ["OPENAI_API_KEY"] = ''

# tiktoken을 이용한 텍스트 길이 계산
tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# JSONL 파일 경로 지정
file_path = './fullDocuments.jsonl'

# JSONL 파일에서 개별 JSON을 추출하여 파이썬 객체로 로딩
loader = JSONLoader(
    file_path=file_path,
    jq_schema='.',
    text_content=False,
    json_lines=True
)
```

openai 의 tiktoken을 사용해 텍스트의 길이를 계산, json파일의 경로를 지정하고 해당 파일에서 개별 객체를 추출하여 파이썬 객체로 로딩

```jsx
data_toEmbedded = loader.load()

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=tiktoken_len
)
split_documents = text_splitter.split_documents(data_toEmbedded)
```

가진 데이터를  RecursiveCharacterTextSplitter로 분리한다, 그러나 이미 json에서 단락별로 잘려져 있기 때문에 단락 내에서 chunk size가 넘어가는 것들만 분리한다.

```jsx

# 허깅 페이스 모델을 이용한 임베딩 생성
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# FAISS 벡터스토어 생성 및 문서 인덱싱
vectorstore = FAISS.from_documents(split_documents, embedding=hf_embeddings)

# 벡터스토어를 로컬에 저장
vectorstore.save_local('./db/faiss')

print("벡터스토어가 성공적으로 생성되고 './db/faiss'에 저장되었습니다.")
```

이후 허깅페이스의 BAAI/bge-large-en-v1.5모델을 사용하여 임베딩을 생성한 후, faiss로 벡터스토어에 이 임베딩과 split_documents들을 저장한다.

## Rag_streamlit.py

```jsx
import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate 
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
```

필요한 라이브러리를 import한다

```jsx
# code related logo
Hyundai_logo = "images/Hyundai_logo.png"
horizontal_logo = "images/Hyundai_logo_horizen.png"
# brake_pad = "images/brake-pad.png"

# 이미지 출처 - <a href="https://www.flaticon.com/kr/free-icons/-" title="브레이크 패드 아이콘">브레이크 패드 아이콘 제작자: Justin Blake - Flaticon</a>
```

디자인적인 요소를  채운다

```jsx
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

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Start chatting")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # FAISS 벡터스토어 로드
        vectorstore = load_vectorstore('db/faiss')

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
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
                    formatted_query = apply_prompt_template(query)

                    result = chain({"question": formatted_query})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    source_documents = result['source_documents']

                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:
                            st.markdown(doc.metadata['source'], help=doc.page_content)

                    st.session_state.messages.append({"role": "assistant", "content": response})
```

main 로직, main 타이틀과 로고 등등의 디자인적인 요소를 추가

openai의 api key를 입력한 후 start chatting을 눌러 코드가 실행되게끔 한다.

db/faiss의 경로에서 db를 불러온 후 대화를 시작

사용자의 질문을 세션상태의 리스트의 user역할에 추가함

이후 prompt를 작성한 내용과 함께 LLm에 전달, 대화chain을 이용해 모델에서 응답을 받고, 해당 내용을 출력함

```jsx
def apply_prompt_template(query):
    system_prompt_template = PromptTemplate(
        input_variables=["question"],
        template="All answers are based on the documentation,If you don't answer, explain why You should never answer I DON'T KNOW. and The document comes from JSON. \nQuestion: {question}"
    )
    return system_prompt_template.format(question=query)
```

프롬프트를 작성하는 함수,

prompt Template를 이용하여 질문에 해당 template를 추가하여 전달

```jsx
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
```

벡터스토어를 로드하는 코드, 여기서 사용한 임베딩 모델은 db를 만들때의 모델과 동일해야함. 이후 faiss 벡터스토어를 로드함

```jsx
def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4o-mini-2024-07-18', temperature=0)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="refine",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True,
    )
    return conversation_chain

if __name__ == '__main__':
    main()
```

기반 LLM에 답변을 요청하는 코드