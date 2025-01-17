import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
import openai
from PIL import Image


# 환경 변수 설정
os.environ["LANGCHAIN_PROJECT"] = "장학금 QNA 챗봇"
openai_api_key = ""
openai.api_key = openai_api_key  # OpenAI API 키 설정
st.title("장학금 QNA 챗봇")
logo_url = "https://github.com/ssidnwm/Data-Visualization/blob/main/rogo.png?raw=true"
logohead = "https://github.com/ssidnwm/Data-Visualization/blob/main/Group%2038.png?raw=true"
logotext = "https://github.com/ssidnwm/Data-Visualization/blob/main/Group%2037.png?raw=true"
# 페이지 기본 설정
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

st.sidebar.image(logohead)
st.sidebar.image(logotext)
sidebar_style = """
                <style>
                [data-testid="stSidebar"] > div:first-child {
                    background-image: url("https://i.imgur.com/6QLoG4r.png");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                }
                </style>
                """
st.markdown(sidebar_style, unsafe_allow_html=True)

background_image = """
                <style>
                [data-testid="stAppViewContainer"] {
                    background-image: url("https://i.imgur.com/zgPNYoY.png");
                    background-size: cover;
                    background-position: center;  
                    background-repeat: no-repeat;
                }
                [data-testid="stHeader"] {
                    background-color: rgba(0,0,0,0);
                }
                footer {
                    background-color: rgba(0,0,0,0);  /* 바텀 배경 투명 */
                }
                </style>
                """
st.markdown(background_image, unsafe_allow_html=True)

# CSS로 메인 페이지 배경색 변경
st.markdown(
    """
    <style>
    /* 메인 페이지 배경색 */
    [data-testid="stAppViewContainer"] {
        background-color: #F5F5F5; /* 원하는 배경색 코드 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# DB 로드
embedding_model = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")

qa_db = Chroma(
    persist_directory="./db/chroma/qaset",
    embedding_function=embedding_model
)
notice_db = Chroma(
    persist_directory="./db/chroma/outdata",
    embedding_function=embedding_model
)
regulation_db = Chroma(
    persist_directory="./db/chroma/regulations",
    embedding_function=embedding_model
)

# 질문 분석 함수
def analyze_question_with_llm(question):
    """
    LLM을 사용하여 질문을 분석하고 적합한 DB를 선택합니다.
    """
    analysis_prompt = f"""
    사용자의 질문을 분석하여 아래 중 하나를 선택하세요:
    1. "공지"에 대한 질문인지
    - 예시: "익산시 장학금은 어떻게 되나요?", "신청 마감일이 언제인가요?", "신청 자격이 어떻게 되나요?"
    - 주의할 키워드: "장학금", "신청 자격", "지원금", "신청 기한", "제출 서류", "문의"
    - 공지 DB에는 다양한 장학금에 대한 종합적인 설명이 포함되어 있습니다. 사용자가 특정 장학금에 대해 질문할 경우 해당 장학금의 정보(신청 자격, 지원금, 신청 기한, 제출 서류, 문의 사항 등)를 제공해야 합니다.

     2. "규제"에 대한 질문인지
    - 예시: "장학금 수혜 규정이 무엇인가요?"
    - 주의할 키워드: "규정", "학칙", "정책", "신청자격의 제한"
    - 규제 DB에는 장학금에 대한 대학의 규정이 포함되어 있습니다. 예를 들어:
    - "제 6 조 (신청자격의 제한): 다음 각 호에 해당하는 학생은 당해 학기에 장학금을 신청할 수 없다."와 같은 내용이 있습니다.

     3. "QA 질문"에 대한 질문인지
    - 예시: "장학금 신청 시 필요한 서류는 무엇인가요?", "지원 자격이 어떻게 되나요?"
    - 주의할 키워드: "서류", "추천서", "장학금", "문의"
    - QA DB에는 장학금과 관련하여 학생들의 개인적인 질문과 담당자의 답변이 포함되어 있습니다. 예를 들어:
    - "장학금 수혜 확인서를 발급받으려면 어떻게 해야 하나요?"와 같은 질문에 대한 답변이 포함됩니다.


    질문: "{question}"
    선택 결과만 출력하세요: 공지, 규제, 기타
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": analysis_prompt}]
    )
    response =  response.choices[0].message.content.strip() 
    if response == "공지":
        return "공지 데이터", notice_db, response
    elif response == "규제":
        return "규제 데이터", regulation_db, response
    else:
        return "QA 데이터", qa_db, response

# 사용자 질문 입력
if prompt := st.chat_input("무엇이 궁금하세요?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    if openai_api_key:
        model = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o-mini", temperature=0)
        selected_db_name, selected_db, response = analyze_question_with_llm(prompt)
        
        st.write(f"선택된 데이터베이스: {selected_db_name}")


        # MultiQueryRetriever 설정
        retriever = MultiQueryRetriever.from_llm(
            retriever=selected_db.as_retriever(search_kwargs={"k": 3}),
            llm=model
        )

        # 검색 수행
        results = retriever.get_relevant_documents(prompt)

        # ChatPromptTemplate 설정
        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="당신은 한동대학교 소속의 장학금 관련 민원을 처리해주는 선생님입니다."),
                HumanMessagePromptTemplate.from_template(
                    """
                    질문 : {question}
                    질문에 대해서 반드시 아래의 문맥에 기반하여 답해주세요.  
                    주어진 질문의 의도에 대해 명확하게 답변하고, 답변이 완료된 이후에는 주어진 문맥을 추가로 제공하여 이해를 돕습니다.
                    주어진 문맥 : {context}
                    """
                ),
            ]
        )

        # 검색된 문맥 조합
        combined_context = "\n".join([doc.page_content for doc in results])

        # ChatGPT 모델에 전달할 메시지 생성
        message = chat_template.format_messages(
            question=prompt, context=combined_context
        )

        def stream_response(stream):
            for chunk in stream:
                yield chunk.content

        # Streamlit UI에 응답 출력
        with st.chat_message("assistant"):
            stream = model.stream(message)
            st.write_stream(stream_response(stream))

        # 검색 내용 확인
        with st.expander("검색 내용 확인"):
            if results:
                for i, doc in enumerate(results):
                    st.write(f"문서 {i+1}:")
                    st.text(doc.page_content)  # 문서 내용 출력
            else:
                st.write("검색된 문서가 없습니다.")
