<img width="840" height="591" alt="image" src="https://github.com/user-attachments/assets/76e21575-c9bc-40b3-972a-919337a7df09" />
<h1>📄 AskResume: AI-Powered Resume Q&amp;A</h1>

<p>
This project enables users to interact with Subash’s resume using a conversational AI interface powered by 
<strong>LangChain</strong>, <strong>OpenAI</strong>, <strong>FAISS</strong>, and <strong>AWS S3</strong>. 
It provides two Streamlit-based applications:
</p>

<ul>
  <li><strong>resumeask.py</strong> → A <em>chatbot UI</em> that answers questions about Subash’s resume using a Retrieval-Augmented Generation (RAG) pipeline.</li>
  <li><strong>admin.py</strong> → An <em>admin tool</em> for uploading PDF resumes, splitting content, generating embeddings, and creating a searchable FAISS vector store stored in AWS S3.</li>
</ul>

<h2>🚀 Features</h2>
<ul>
  <li>📥 <strong>PDF Upload &amp; Processing</strong>: Upload a resume in PDF format and automatically split it into manageable text chunks.</li>
  <li>🧠 <strong>Semantic Search with FAISS</strong>: Create and query a FAISS vector store backed by OpenAI embeddings for efficient semantic retrieval.</li>
  <li>☁️ <strong>AWS S3 Integration</strong>: Store and retrieve FAISS index files in S3 for persistent access across sessions.</li>
  <li>🤖 <strong>Conversational Q&amp;A</strong>: Streamlit-powered chat interface lets you ask natural language questions about Subash’s resume.</li>
  <li>🔧 <strong>Customizable Parameters</strong>: Adjust LLM temperature and max tokens via the Streamlit sidebar for tailored responses.</li>
</ul>

<h2>📂 Project Structure</h2>
<pre>
├── resumeask.py   # Streamlit chat app for resume Q&amp;A
├── admin.py       # Admin tool for uploading resumes and creating FAISS index
├── requirements.txt
├── Dockerfile
└── .gitignore
</pre>

<h2>⚙️ Tech Stack</h2>
<ul>
  <li><strong>Python 3.11+</strong></li>
  <li><strong>Streamlit</strong> for the UI</li>
  <li><strong>LangChain</strong> for embeddings &amp; retrieval</li>
  <li><strong>OpenAI GPT-4</strong> for conversational answers</li>
  <li><strong>FAISS</strong> for vector search</li>
  <li><strong>AWS S3 (boto3)</strong> for index storage</li>
  <li><strong>dotenv</strong> for environment variable management</li>
</ul>

<h2>🔑 Setup Instructions</h2>
<ol>
  <li><strong>Clone the repository</strong>
    <pre>
git clone https://github.com/nethajisubash/askResume.git
cd askResume
    </pre>
  </li>
  <li><strong>Create a .env file</strong> (never commit this file!)
    <pre>
OPENAI_API_KEY=your_openai_api_key
BUCKET_NAME=your_s3_bucket
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
    </pre>
  </li>
  <li><strong>Install dependencies</strong>
    <pre>pip install -r requirements.txt</pre>
  </li>
  <li><strong>Run the Admin Tool</strong> (to upload resume &amp; build FAISS index)
    <pre>streamlit run admin.py</pre>
  </li>
  <li><strong>Run the Q&amp;A Chatbot</strong>
    <pre>streamlit run resumeask.py</pre>
  </li>
</ol>

<h2>📌 Use Case</h2>
<p>
This app is ideal for:
</p>
<ul>
  <li>Showcasing Subash’s professional profile interactively</li>
  <li>Practicing Retrieval-Augmented Generation with real documents</li>
  <li>Learning how to build a production-ready AI chatbot using FAISS, LangChain, and AWS</li>
</ul>

