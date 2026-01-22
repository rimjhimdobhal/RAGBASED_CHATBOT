import os
import google.generativeai as genai
from pdfextractor import text_extractor
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Let's configure the models

# 1. LLM Model
gemini_key = os.getenv('GOOGLE_API_KEY2')
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# 2. Embedding model
embedding_model = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')

# Let's create the main page
st.markdown(
    """
    <h1 style="
        font-size:52px;
        font-weight:700;
        display:inline-block;
    ">
        <span style="color:#3F9AAE;">RAG CHATBOT:</span>
        <span style="color:#F96E5B;"> AI-Powered Document Intelligence</span>
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    "#### Ask questions, retrieve insights, and chat intelligently with your documents."
)
st.markdown("""
**How it works**
1. Upload a PDF document from the sidebar  
2. Write your query and start the chat
""")

# Let's create the sidebar
st.sidebar.title('Upload your document')
st.sidebar.subheader('Supported format: PDF only')
pdf_file = st.sidebar.file_uploader('Choose a file', type = ['pdf'])

if pdf_file:
    st.sidebar.success('PDF uploaded successfully')

    file_text = text_extractor(pdf_file)

    # Step 1: Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                          chunk_overlap = 200)
    chunks = splitter.split_text(file_text)  

    # Step 2: Create the vector database (FAISS)
    vector_store = FAISS.from_texts(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_kwargs = {'k':3})

    def generate_content(query):
        # Step 3: Retriever (R)
        retrieved_docs = retriever.invoke(query)
        context = '\n'.join([d.page_content for d in retrieved_docs])

        # Step 4: Augmenting (A)
        augmented_prompt = f'''
        <Role> You are a helpful assistant using RAG.
        <Goal> Answer the question asked by the user. Here is the question: {query}
        <Context> Here are the documents retrieved from the vector database to support
        the answer which you have to generate {context}'''

        # Step 5: Generate (G)
        response = model.generate_content(augmented_prompt)
        return response.text
    
    # Create ChatBot in order to start the conversation
    # Initialize chat: Create history if not created
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display the history
    # for msg in st.session_state.history:
        # if msg['role'] == 'user':
            # st.write(f'**USER:** {msg['text']}')
        # else:
            # st.write(f'**CHATBOT:** {msg['text']}')

    # Display the history
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style="
                    background-color:#F6F7F9;
                    padding:12px 16px;
                    border-radius:10px;
                    margin-bottom:10px;
                    border-left:4px solid #C5CCD6;
                ">
                    <strong>USER</strong><br>
                    {msg['text']}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="
                    background-color:#F6F7F9;
                    padding:12px 16px;
                    border-radius:10px;
                    margin-bottom:18px;
                    border-left:4px solid #D6D6D6;
                ">
                    <strong>CHATBOT</strong><br>
                    {msg['text']}
                </div>
                """,
                unsafe_allow_html=True
            )



    # Input from the user using streamlit form
    with st.form('Chatbot Form', clear_on_submit= True):
        user_query = st.text_area('Ask anything')
        send = st.form_submit_button('Send')

    # Start the conversation and append output and query in history
    if user_query and send:
        st.session_state.history.append({'role': 'user', 
                                         'text': user_query})
        st.session_state.history.append({'role': 'chatbot',
                                        'text': generate_content(user_query)})
        st.rerun()







