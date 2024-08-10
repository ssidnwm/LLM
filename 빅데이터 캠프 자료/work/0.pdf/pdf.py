from langchain_community.document_loaders import PyMuPDFLoader

# pdf파일에서 정보 추출(text가 인식된 정보만 추출)
loader = PyMuPDFLoader("little_prince.pdf")
docs = loader.load()

# 단순하게 쭉 써서 txt파일로 저장
f = open("test.txt", "w")
for x in docs:
    f.write(str(x))