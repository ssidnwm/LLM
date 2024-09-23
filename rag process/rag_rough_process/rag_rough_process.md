# rag구현 소스코드(colab)

[Google Colab](https://colab.research.google.com/drive/1Ajo7wgxDFSUcIiy7R5OxkGk15TbvF5He?usp=sharing)

랭체인으로 rag를 구현하는 소스코드, 아직 실제로 코드를 돌려보지 않았기에 오류가 있을수 있습니다.

RAG 프로세스별 성능 최적화 방안

```jsx
!pip install chromadb tiktoken transformers sentence_transformers openai langchain pypdf
```

크로마 DB, tiktoken, 트랜스포머, openai, 랭체인, pypdf사용. 

백터 DB를 크로마가 아닌 다른 DB를 이용한다거나, 기반 LLM을 openai가 아닌 다른 것을 이용하는 등 여러 방법에 따라 install을 바꿀 수 있음

```jsx
!pip install -U langchain-community
```

Pypdf로 구글 드라이브에서 문서를 로드할때 필요한 라이브러리

```jsx
from google.colab import drive
drive.mount('/content/drive')
```

구글 드라이브 마운트, 구글 드라이브에 올라간 PDF파일을 업로드하여 이용할 수 있음. 만일 다른 방법으로 pdf파일이나 혹은 다른 문서를 업로드한다면 변경할 수 있음.

```jsx
import os
import openai
os.environ["OPENAI_API_KEY"] = 'API-KEY'
```

openai의 api키를 가지고 옴. 이후 LLM을 openai의 gpt3.5를 이용하기 때문에 필요, 만일 LLM을 copilot이나 gemini를 사용한다면, 구글이나 ms의 api를 저장하는 것이 필요

```jsx
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)
```

tiktoken의 tokenizer을 이용하여 문서의 토큰을 계산함. 이때 문서의 토큰은 문서의 청크를 분할하는데 쓰이며, 토크나이저의 인코딩 모델은 기반모델인 GPT에서 사용하는 cl100k_base를 사용, 다른 언어모델을 사용할 경우 그에 맞는 토크나이저 모델이 필요

```jsx
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
```

랭체인의 retrivalQA, openai의 챗봇 모델, RecursiveCharacterTextSplitter(추후  설명), 크로마DB, Pypdf loader등의 라이브러리 사용

```jsx
loader = PyPDFLoader("/content/drive/MyDrive/R152r2E.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function = tiktoken_len)
texts = text_splitter.split_documents(pages)

from langchain.embeddings import HuggingFaceEmbeddings

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

docsearch = Chroma.from_documents(texts, hf)
```

우선 pdf로더를 사용해 구글 드라이브에 올라간 학습용 PDF파일을 다운로드하고 분할하여 pages로 저장( 이때의 분할은 페이지별 분할) 

이후 텍스트 스플리터를 활용해 문서를 분할하는데, 청크사이즈는 500토큰, 오버랩 50토큰이며

분할의 요소로서 문단변화, 줄바꿈, 마침표, 쉼표 순으로 재귀적으로 분할함. → 이 외에 다른 방법으로 CharacterTextSplitter이 있는데, 이 방법은 구분자 1개를 기준으로 분할함, 때문에 max토큰을 지키지 못하는 경우가 발생할 수 있음

RecursiveCharacterTextSplitter는 기본적으로는 문단에 따라 청크를 분할하며, 이때 문단이 max토큰을 초과한 경우 그 다음 줄바꿈이나 마침표에 따라 청크를 분리함. 

이후 임베딩 모델로서 huggingface의 오픈소스 모델을 사용. 리더보드 상위권에 위치한 모델이며, 이 임베딩 모델을 다른 openai나 LLM의 모델로 변경하는 것도 가능.

```jsx
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

openai = ChatOpenAI(model_name="gpt-3.5-turbo",
                    streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
                    temperature = 0)

qa = RetrievalQA.from_chain_type(llm = openai,
                                 chain_type = "stuff",
                                 retriever = docsearch.as_retriever(
                                    search_type="mmr",
                                    search_kwargs={'k':3, 'fetch_k': 10}),
                                 return_source_documents = True)

query = "What is the role of emergency braking assist systems?"
result = qa(query)
```

openai의 챗gpt 3.5-turbo모델을 사용하여 qa답변을 사용, 이때 temperature는 생성하는 답변의 일관성을 높임. (0~2까지 설정이 가능하며 0이 매우 일관되게, 2가 일관되지 않게 함)

qa질문은 리트리버를 사용하고, llm은 위에서 정의한 gpt3.5모델을 사용함

- chain_type에는 4가지 방법이 있는데,
    
    stuff
    
    가장 단순한 형태로, 유사성이 높은 청크를 
    
    ```jsx
    question:{질문}
    Context:{검색한 문서 청크} 
    ```
    
    의 형태로 보냄. 이때, 청크를 유사도 검색 상위 K개를 모두 보내기 때문에 토큰 이슈가 발생할 수 있음
    
    MapReduce
    
     질문에 대한 청크들을 가져온 후, 각각에 대해 1차적으로 요약을 실행. 이 과정을 맵이라고 하며, 생성된 K개의 요약을 가지고 다시 한번 최종적인 요약을 작성 컨텍스트를 압축하여 짧은 문장으로 LLM에게 전달하여 답변을 생성함.
    
    토큰이슈를 우회할 수는 있지만 지속적으로 api를 호출하여 청크를 요약하고 답변을 받기 때문에 속도가 느림.
    
    Refine
    
     보다 좋은 품질의 답변을 받기 위해 사용됨
    
    ```jsx
    question:{질문}
    intermediate : {K-1번째 청크에 대한 답변}
    Context : {k번째 문서 청크}
    ```
    
    청크들을 하나씩 llm에 보낸 후, 생기는 답변을 intermediate로, 그 다음 우선도의 문서 k를 context로 LLM에 보내 답변의 품질을 높이는 방법. 질문에 관한 문서청크와, 이전 질문의 답변을 넣어가며 계속해서 답변의 품질을 높임, 다만 병렬적인 수행이 아니기 때문에 답변의 속도가 매우 느림.
    
    Map-Rerank
    
    사용자가 질문한 것에 대해, 연관문서들 각각에 대해 하나씩 프롬프트를 생성함
    
    질문과 context를 넣은 프롬프트로 나온 답변 각각에 대해 추가로 Score을 매김. 이중에서 Score가 가장 높은 하나를 답변으로 채택함. 품질은 뛰어나지만 시간도 오래걸리고 비용도 다른 방법에 비해 많이 발생함.
    

chroma DB를 검색기로서 이용하여 텍스트를 검색함. 이때, chroma DB는 저장소로서 기능하는 것이 아닌, 검색기로서 사용

search_type mmr은 검색하는 문서(청크) 여러 개에서 참고하기 위한 타입, 연관성이 높은 순서대로 가져오는 것이 아닌, 연관성이 높은 문서들 안에서도 다양성을 챙기기 위한 타입

search_kwrgs는 그에 대한 세부 파라미터

fetch_k는 연관 문서 후보들 중에서 다양하게 문서를 가져올 때, 후보의 갯수를 지정해주는 매개변수

K는 최종적으로 사용할 문서의 갯수

이후 quary를 통해 질문을 하고, result를 통해 query에 대한 답변을 받음.

RAG의 전체 프로세스 및 성능개선을 위한 최적화 방안.

RAG의 프로세스는 문서 다운로드, 텍스트 분할, 텍스트 임베딩, 벡터스토어 및 리트리버 단계로 구분가능

각 단계별로 사용할 수 있는 방법들이 다양한데 각각에 대해 조사 후 기술, 최적화 방안 마련 필요

![Untitled](rag%E1%84%80%E1%85%AE%E1%84%92%E1%85%A7%E1%86%AB%20%E1%84%89%E1%85%A9%E1%84%89%E1%85%B3%E1%84%8F%E1%85%A9%E1%84%83%E1%85%B3(colab)%200214b727ccc34accafb6c9b9782b51d7/Untitled.png)

- 데이터 전처리
- 문서 다운로드
    
    가장 먼저 rag를 이용하기 위해서 학습용 데이터 파일을 다운로드(혹은 input)하는 과정.
    
    벡터 데이터베이스에 법률 데이터를 저장하기 위해서든 모델에 직접적으로 데이터를 밀어넣기 위해서든 우선 가지고 있는 데이터 파일의 내용을 로컬환경 혹은 가상환경에서 열람할 수 있도록 하는 과정
    
    이 단계에서 할 수 있는 최적화 방법은 다음과 같음.
    
    데이터 파일의 형식, 내용 등에 따라서 최적의 loader을 선택.
    
    PDF, Word, csv등등의 데이터 뿐 아니라, 웹에 있는 데이터 또한 URL Document Loader를 이용하여 가져올 수 있음
    
    추가적인 데이터가 필요하거나, 업로드가 필요하다면 그에 맞는 데이터파일 형태의 loader을 이용.
    

- 텍스트 분할
    
    TextSplitter 파라미터 조정
    
    → max token수 조정, chunk overlab조정, length function조정 등
    
    데이터의 형식과 내용에 따라 적절히 조
    
    - RecursiveCharacterTextSplitter
        
        줄바꿈, 마침표, 쉼표 순으로 재귀적으로 분할
        
        추가로, TextSpliter은 파이썬 등의 코드나, LeTax 등의 컴퓨터 언어는 이해하지 못하기 때문에 이런 기타 문서를 분할하기 위해서는 RecursiveCharacterTextSpliter에서 Language를 이용한다.
        
        ```jsx
        from langchain.text_splitter import (RecursiveCharacterTextSplitter,Language)
        
        RecursiveCharacterTextSplitter.get_seperators_for_language(language,언어(python))
        ```
        
    - CharacterTextSplitter
        
        구분자 1개를 기준으로 분할(설정 필요)
        
    

- 텍스트 임베딩
    
    임베딩 모델의 종류에 따라 임베딩 성능이 달라질 수 있음.
    
    데이터의 언어와 출력을 위한 언어에 따라 임베딩 모델 종류가 달라질 수 있음.
    
    거대모델의 임베딩 모델은 api를 이용하는데 들어가는 비용이 청구됨.
    

- 벡터스토어
    
    chromaDB, FAISS등 용도와 역할에 따라 다름
    
    chromaDB의 경우 무료로 이용할 수 있지만, FAISS는 비용이 필요
    
    chromaDB는 벡터공간에 저장하고 사용하는데에 용이,
    
    FAISS는 벡터를 찾는데에 용이
    
    persist()함수를 이용하여 로컬에 벡터 저장소를 저장하고 필요할 때 다시 불러서 사용 가능(구현 필요)
    

- 리트리버
    
    chain_type4가지를 적절히 이용하며 모델의 답변성능 및 생성속도 조절,
    
    기반LLM을 오픈소스 모델 및 거대모델로 적절히 구현
    

모델 사전평가

질문: Describe the most important thing you learned in this article(이 문서에서 가장 중요한 부분은 무엇인가?)

LLM답변: The most important thing learned from the provided text is the detailed requirements and procedures for testing and verifying the safety and performance of "The System" in vehicles, as per UN Regulation No. 152. This includes the robustness of the system, the criteria for passing test scenarios, the analysis of failed test runs, and the documentation requirements for demonstrating the system's capabilities. It emphasizes the importance of ensuring that the system can reliably deliver the required performances and maintain control within defined boundaries.

(제공된 텍스트에서 가장 중요한 것은 UN 규정 152호에 따라 차량에서 "시스템"의 안전 및 성능을 테스트하고 검증하기 위한 세부 요건과 절차입니다. 여기에는 시스템의 견고성, 테스트 시나리오 통과 기준, 테스트 실행 실패 분석, 시스템 기능 시연을 위한 문서 요건 등이 포함됩니다. 이는 시스템이 필요한 성능을 안정적으로 제공하고 정의된 범위 내에서 제어를 유지할 수 있는지 확인하는 것이 중요하다는 점을 강조합니다.)

# Reference

[모두의 AI](https://www.youtube.com/watch?v=tQUtBR3K1TI&t=1285s)
