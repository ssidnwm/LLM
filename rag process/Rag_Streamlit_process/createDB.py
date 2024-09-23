import os
import tiktoken
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

data_toEmbedded = loader.load()

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=tiktoken_len
)
split_documents = text_splitter.split_documents(data_toEmbedded)

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
