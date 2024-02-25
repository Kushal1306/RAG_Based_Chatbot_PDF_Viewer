from flask import Flask, request, jsonify, send_from_directory
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import fitz  # PyMuPDF
import torch
import nltk
import os
nltk.download('punkt')

load_dotenv()




app = Flask(__name__,static_url_path='',static_folder='frontend')
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
embedder=SentenceTransformer("all-MiniLM-L6-v2")


def find_similar_text(all_text,response):
    # splitting the whole text into sentences
    sentences=nltk.sent_tokenize(all_text)

    # Initializing SentenceTransformer
    #embedder=SentenceTransformer("all-MiniLM-L6-v2")
    
    #Converting into embeddings

    all_text_embeddings = embedder.encode(sentences, convert_to_tensor=True)
    response_embedding = embedder.encode(response, convert_to_tensor=True)

    hits=util.semantic_search(response_embedding,all_text_embeddings,top_k=1)
    hits = hits[0]



    for hit in hits:
        corpus_id = hit['corpus_id']
        similarity_score = hit['score']
        similar_text = sentences[corpus_id]

    # hit_id=hits['corpus_id']
    # similar_text=sentences[hit_id]

    return similar_text

def find_answer(all_text,query):

        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text=all_text)
        embeddings=OpenAIEmbeddings()
        vectorstore=FAISS.from_texts(chunks,embedding=embeddings)
        if query:
            docs=vectorstore.similarity_search(query=query,k=3)
           # print(docs)
            llm=OpenAI()
            chain=load_qa_chain(llm=llm,chain_type="stuff")

            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs,question=query)
        return response

    


def process_pdf(pdf_path, longest_similar_text,num_pages):
    
   
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
  
    pdf_document = fitz.open(pdf_path)
   
    
    for page_num in range(num_pages):
        page = pdf_document[page_num] 
        highlight = page.search_for(longest_similar_text)
        page.add_highlight_annot(highlight)

    # Save the processed PDF file
    processed_pdf_filename = f'processed_{os.path.basename(pdf_path)}'
    processed_pdf_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_pdf_filename)
    pdf_document.save(processed_pdf_path)
    pdf_document.close()

    return processed_pdf_filename

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/process_query', methods=['POST'])
def process_query():
    question = request.form.get('question')
    pdf_file = request.files.get('pdfFile')

    # Save the uploaded PDF file
    pdf_filename = pdf_file.filename
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    pdf_file.save(pdf_path)

    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        num_pages = len(pdf_reader.pages)
        # for page_num in range(num_pages):
        #     page = pdf_reader.pages[page_num]
        #     text = page.extract_text()
        #     if question.lower() in text.lower():
        #         answer = "Answer found in the PDF."
        #         break

    # Highlight the answer in the PDF (using PyMuPDF)
    pdf_document = fitz.open(pdf_path)
    all_text=''
    for page_num in range(num_pages):
        page=pdf_document[page_num]
        all_text+=page.get_text("text")

    # Process the PDF file
    answer=find_answer(all_text,question)
    longest_similar_text=find_similar_text(all_text,answer)
    processed_pdf_filename = process_pdf(pdf_path,longest_similar_text,num_pages)

    # Construct URLs for the processed PDF file
    processed_pdf_url = f'/download/{processed_pdf_filename}'

    # Prepare response
    response = {
        'answer': answer,
        'processed_pdf_url': processed_pdf_url
    }

    return jsonify(response)

@app.route('/download/<path:filename>', methods=['GET'])
def download(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask,jsonify

# app = Flask(__name__, static_url_path='', static_folder='frontend')

# @app.route('/')
# def index():
#     return app.send_static_file('index.html')

# @app.route('/api/data')
# def get_data():
#     data={'message':'This is a response from backend!'}
#     return jsonify(data)


# if __name__ == '__main__':
#     app.run(debug=True)
