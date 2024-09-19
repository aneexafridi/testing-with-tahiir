import google.generativeai as genai
import os
from flask import Flask, request, jsonify,render_template
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

genai.configure(api_key="")
llm = genai.GenerativeModel('gemini-1.5-pro')

embedding_dir = "Embedding"

def query_find(user_query, embedding_dir):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="A")
    embedding_storage = FAISS.load_local(embedding_dir, embeddings, allow_dangerous_deserialization=True)
    docs = embedding_storage.similarity_search(user_query, k=1)
    ind = docs[0].page_content

    print("Embedding result: >>>>>>>>>>>>", ind)
    start_index = ind.find("how")

    if start_index != -1:  # Ensure 'how' was found
        result = ind[start_index:]
    else:
        result = ""

    print(result)
    return result

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def display_directory_content(directory_path):
    if os.path.isdir(directory_path):
        list_images = []
        # Display images in sequence
        image_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.png')])
        for image_file in image_files:
            image_path = os.path.join(directory_path, image_file)
            encoded_image = encode_image(image_path)
            list_images.append(encoded_image)

        # print(">>>>>>>>>>>>>list of images", list_images)
        text_file = os.path.join(directory_path, os.path.basename(directory_path) + '.txt')
        if os.path.isfile(text_file):
            with open(text_file, 'r') as file:
                text_content = file.read()
                res_gen = llm.generate_content(f"""You are a drawing teacher for kids of age 5 to 10.\n
                Your task is to make a useful guide out of the provided text.
                Text:{text_content}""")
                gen_text = res_gen.text
                print(res_gen.text)
        else:
            gen_text = "No text file found."
            print("No text file found.")

    else:
        print("Invalid directory path.")
        return [], "Invalid directory path."

    return list_images, gen_text

@app.route('/chat', methods=['POST','GET'])
def home():
    return render_template('fourth-page.html')

@app.route('/', methods=['POST'])
def main():
    if request.method == 'POST':
        user_query = request.json['user_query']
        print("userquery",user_query)
        result_path = query_find(user_query, embedding_dir)
        if result_path:
            image_data, generated_text = display_directory_content(f"Data/{result_path}")

            return jsonify({
                'images': image_data
                
            })
            
        else:
            return jsonify({'error': 'No result found.'}), 404

if __name__ == "__main__":
    app.run(debug=True)
