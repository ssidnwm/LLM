from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("little_prince.pdf")
docs = loader.load()

f = open("test.txt", "w")
for x in docs:
    f.write(str(x))