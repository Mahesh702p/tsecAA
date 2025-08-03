import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="TSEC Admission Assistant", page_icon="🎓")

# Sidebar for branding and info
with st.sidebar:
    st.image(r"C:\Users\Lenovo\Desktop\tsecAA\assets\tsec_logo.jpg", width=200)
    st.title("TSEC Admit Assist")
    st.markdown(
        """
        **Ask me anything about TSEC admissions!**
        - Courses
        - Eligibility
        - Fees
        - Placements
        - Campus life
        """
    )
    st.info("Powered by LangChain, Gemini, and Streamlit.")

@st.cache_resource
def setup_chain():
    loader = PyPDFLoader(r"C:\Users\Lenovo\Desktop\tsecAA\assets\COLLEGE-INFORMATION.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    template = """You are "TSEC_Admit_Assist", the official admission chatbot for Thadomal Shahani Engineering College. Your primary role is to assist with admission-related queries while maintaining a professional yet friendly tone.

### Response Guidelines:
1. **Accuracy**: 
   - First check the provided context from the PDF
   - If exact information exists, provide precise details with relevant numbers/dates
   - If unsure, say: "Based on available information, [details]. For exact details, please visit [relevant department]."

2. **Formatting**:
   - Use clear headings (###) for different sections
   - Present lists with bullet points
   - Bold important details (**key dates**, **deadlines**, **requirements**)
   - Use tables for comparative information when appropriate

3. **Behavior**:
   - Be proactive in asking clarifying questions when needed
   - Maintain TSEC's professional reputation
   - Politely decline unrelated requests: "I specialize in TSEC admissions. How can I help with that?"

4. **Special Cases**:
   - For course queries: Include duration, fees, eligibility, and placement stats
   - For deadlines: Highlight **important dates** in bold
   - For comparisons: Present in table format when possible

### Current Context:
{context}

### Conversation History:
{history}

### Question:
{question}

### Response Format:
[Start with appropriate greeting if new conversation]
[Provide concise, accurate information]
[End with relevant follow-up question/suggestion]
Context: {context}

Conversation History:
{history}

Question: {question}
Answer:"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "history", "question"]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)
    memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=True)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={
            "prompt": prompt,
            "memory": memory
        },
        return_source_documents=True
    )
    return qa_chain

qa_chain = setup_chain()

if "history" not in st.session_state:
    st.session_state["history"] = []

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem;}
    .stChatMessage {margin-bottom: 1.5rem;}
    .header-box {
        margin-top: 4.5rem;
        background-color: #3f5870;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add the rectangular header at the top of the chat section
st.markdown('<div class="header-box">TSEC ADMIT ASSIST</div>', unsafe_allow_html=True)

# Chat display using chat bubbles
for speaker, msg in st.session_state["history"]:
    with st.chat_message("user" if speaker == "You" else "assistant"):
        st.markdown(msg)

# Chat input at the bottom
if prompt := st.chat_input("Type your query"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("tsecAA is typing..."):
        try:
            result = qa_chain.invoke({"query": prompt})
            response = result["result"]
            if response.strip().lower().startswith(("i don't know", "i'm not sure")):
                response = "I'm sorry, I couldn't find specific information about that. " + response
            elif not any(response.startswith(prefix) for prefix in ["Please", "Kindly", "Based on", "The"]):
                response = response
        except Exception as e:
            response = f"Error: {e}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state["history"].append(("You", prompt))
    st.session_state["history"].append(("tsecAA", response))