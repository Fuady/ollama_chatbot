# 🤖 Ollama Chatbot with Streamlit

A powerful chatbot application that runs locally on your machine using Ollama as the AI brain. This application supports two modes:
1. **Free Chat Mode**: Direct conversation with the AI model
2. **Document-based RAG Mode**: Answers questions based on a knowledge base using Retrieval-Augmented Generation (RAG)

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Use Cases](#use-cases)
- [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

Before you begin, ensure you have the following installed on your computer:

1. **Python 3.8 or higher**
   - Check version: `python --version` or `python3 --version`
   - Download from: https://www.python.org/downloads/

2. **Ollama Software**
   - Download and install from: https://ollama.ai/download
   - Supported platforms: macOS, Linux, Windows

---

## 📥 Installation

### Step 1: Install and Setup Ollama

1. **Download Ollama**
   - Visit https://ollama.ai/download
   - Download the installer for your operating system
   - Run the installer

2. **Verify Ollama Installation**
   ```bash
   ollama --version
   ```

3. **Pull an AI Model**
   
   You need to download at least one model. Here are some popular options:
   
   ```bash
   # Recommended for beginners (smaller, faster)
   ollama pull llama2
   
   # Or try these alternatives:
   ollama pull mistral        # Good balance of speed and quality
   ollama pull llama2:13b     # Larger, more capable
   ollama pull phi            # Very small and fast
   ollama pull codellama      # Optimized for coding
   ```

4. **Start Ollama Server**
   
   Ollama usually starts automatically, but you can manually start it:
   ```bash
   ollama serve
   ```
   
   Keep this terminal window open while using the chatbot.

### Step 2: Setup Python Environment

1. **Clone or Download Project Files**
   
   Create a new directory and save these files:
   - `chatbot_app.py` (main application)
   - `requirements.txt` (dependencies)
   - `README.md` (this file)

2. **Create Virtual Environment** (Recommended)
   ```bash
   # Navigate to project directory
   cd path/to/your/project
   
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install:
   - Streamlit (web interface)
   - Requests (API communication)
   - Sentence-Transformers (document embeddings)
   - NumPy (numerical operations)
   - PyTorch (deep learning backend)

---

## 🚀 Running the Application

### Start the Chatbot

1. **Ensure Ollama is Running**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # If not running, start it
   ollama serve
   ```

2. **Launch the Streamlit Application**
   ```bash
   streamlit run chatbot_app.py
   ```

3. **Access the Application**
   - The application will automatically open in your default web browser
   - Default URL: http://localhost:8501
   - If it doesn't open automatically, copy the URL from the terminal

---

## 💡 Use Cases

### Use Case 1: Free Chat Mode (Open-Ended Questions)

This mode allows you to have a natural conversation with the AI about any topic.

**How to Use:**

1. In the sidebar, select **"Free Chat"** mode
2. Choose your preferred model from the dropdown
3. Type your question in the chat input at the bottom
4. Press Enter or click Send

**Example Questions:**
- "Explain quantum computing in simple terms"
- "Write a Python function to calculate fibonacci numbers"
- "What are the benefits of exercise?"
- "Tell me a story about a robot learning to cook"
- "How do I start learning machine learning?"

**Features:**
- Full conversation history maintained
- Context-aware responses (remembers previous messages)
- Supports any topic or question type

### Use Case 2: Document-Based RAG Mode (Knowledge Base)

This mode answers questions based on a specific document collection, making responses more accurate and grounded in specific information.

**How to Use:**

1. In the sidebar, select **"Document-based (RAG)"** mode
2. Click **"Load Documents"** button
   - The system will download and process documents from the specified URL
   - This may take 30-60 seconds on first load
   - You'll see a success message when ready
3. Adjust the "Number of context documents" slider (1-5)
   - More documents = more context but slower responses
4. Type your question related to the document content
5. View retrieved source documents by clicking "View Source Documents" below each answer

**What Happens Behind the Scenes:**
1. Your question is converted to a numerical embedding (vector)
2. The system searches the document database for similar content
3. Top relevant documents are retrieved
4. These documents are sent to the AI along with your question
5. The AI generates an answer based on the provided context

**Example Questions:**
- "What is the process for machine learning?"
- "Explain the concept mentioned in the documents"
- "How does the system work according to the documentation?"

**Features:**
- See which documents were used to answer your question
- Similarity scores show relevance of each document
- Adjustable number of context documents
- Reload documents at any time

---

## 🎯 Application Features

### Sidebar Controls

- **Model Selection**: Choose from available Ollama models
- **Chat Mode Toggle**: Switch between Free Chat and RAG modes
- **Document Loading**: Load and manage the knowledge base
- **Context Documents Slider**: Control how many documents to use (RAG mode)
- **Clear Chat History**: Reset the conversation

### Main Chat Interface

- **Message History**: All previous messages are displayed
- **Source Documents**: Expandable section showing retrieved documents (RAG mode)
- **Real-time Streaming**: Watch responses generate in real-time
- **Error Handling**: Clear error messages if something goes wrong

---

## 🔍 Troubleshooting

### Problem: "No models found" error

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull a model if none exist
ollama pull llama2

# Restart Ollama
ollama serve
```

### Problem: "Error connecting to Ollama"

**Solutions:**
1. Verify Ollama is running: `ollama serve`
2. Check if port 11434 is available
3. Restart Ollama service
4. Check firewall settings

### Problem: "Failed to load documents"

**Solutions:**
1. Check your internet connection
2. Verify the document URL is accessible
3. Try reloading documents using the button in sidebar
4. Check console for detailed error messages

### Problem: Slow responses

**Solutions:**
1. Use a smaller model (e.g., `phi` or `llama2` instead of `llama2:13b`)
2. Reduce number of context documents in RAG mode
3. Ensure your computer has sufficient RAM
4. Close other memory-intensive applications

### Problem: Python package installation errors

**Solutions:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install packages one by one
pip install streamlit
pip install requests
pip install sentence-transformers
pip install numpy
pip install torch

# On macOS with M1/M2 chip:
pip install torch torchvision torchaudio
```

### Problem: "Module not found" errors

**Solution:**
```bash
# Ensure virtual environment is activated
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

---

## 📊 Model Recommendations

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `phi` | 1.3GB | Fast | Good | Quick responses, testing |
| `llama2` | 3.8GB | Medium | Good | General purpose |
| `mistral` | 4.1GB | Medium | Very Good | Balanced performance |
| `llama2:13b` | 7.4GB | Slow | Excellent | Complex tasks |
| `codellama` | 3.8GB | Medium | Good | Programming questions |

---

## 🛠️ Advanced Configuration

### Change Ollama API URL

If Ollama is running on a different port or machine:

```python
# Edit chatbot_app.py
OLLAMA_API_URL = "http://your-host:your-port/api"
```

### Use Different Document Source

```python
# Edit chatbot_app.py
DOCUMENTS_URL = "your-document-url.json"
```

The document format should be:
```json
[
  {
    "question": "Sample question",
    "text": "Sample answer",
    "other_fields": "optional"
  }
]
```

### Customize Embedding Model

```python
# In DocumentRetriever.initialize_embeddings()
self.model = SentenceTransformer('your-preferred-model')
```

Popular alternatives:
- `all-mpnet-base-v2` (higher quality, slower)
- `all-MiniLM-L6-v2` (balanced, default)
- `paraphrase-MiniLM-L6-v2` (good for paraphrasing)

---

## 📝 Tips for Best Results

### For Free Chat Mode:
1. Be specific and clear in your questions
2. Provide context when needed
3. Use follow-up questions to dive deeper
4. The AI remembers conversation history

### For RAG Mode:
1. Ask questions related to the document content
2. Start with broader questions, then get specific
3. Check source documents to verify information
4. Adjust number of context documents if answers seem incomplete

### General Tips:
1. Use larger models for complex tasks
2. Clear chat history for fresh conversations
3. Experiment with different models for different tasks
4. Be patient - first responses may be slower while models load

---

## 📚 Additional Resources

- **Ollama Documentation**: https://github.com/ollama/ollama
- **Streamlit Documentation**: https://docs.streamlit.io
- **Sentence Transformers**: https://www.sbert.net
- **Model Library**: https://ollama.ai/library

---

## 🤝 Support

If you encounter issues:
1. Check the Troubleshooting section
2. Review Ollama logs: `ollama logs`
3. Check Streamlit logs in the terminal
4. Ensure all dependencies are correctly installed

---

## 📄 License

This project is open source and available for educational and personal use.

---

**Happy Chatting! 🚀**
