# AI Projects Repository

This repository contains a collection of AI and machine learning projects focusing on various aspects of natural language processing, data processing, and AI applications. Below is a comprehensive guide to all projects and their recommended running order.

## 🚀 Project Overview

### 1. Basic Projects (Start Here)

- **1.1-openai**: Introduction to OpenAI API and basic implementations
- **ChatBot**: Basic chatbot implementation
- **TextSummarization**: Text summarization using AI models

### 2. LangChain Projects

- **LCEL**: LangChain Expression Language examples
- **LangChain**: Core LangChain implementations
- **Pydantic**: Data validation and settings management

### 3. Advanced Applications

- **ChatBot_RAG**: Chatbot with Retrieval-Augmented Generation
- **RAG_and_Groq**: RAG implementation with Groq integration
- **SearchEngine**: AI-powered search engine implementation
- **SQLAgent**: AI agent for SQL operations
- **MathsGPT**: Mathematical problem-solving AI
- **NLP**: Natural Language Processing projects

## 📋 Prerequisites

Before running any project, ensure you have:

1. Python 3.8+ installed
2. Virtual environment set up
3. Required dependencies installed

### Setup Instructions

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

## 🏃‍♂️ Running Order

### Phase 1: Basic Understanding

1. Start with `1.1-openai` to understand basic API interactions
2. Move to `ChatBot` for basic conversational AI
3. Try `TextSummarization` for text processing

### Phase 2: LangChain Framework

1. Begin with `LCEL` to understand LangChain Expression Language
2. Explore `LangChain` for core implementations
3. Study `StructuredOutput` for structured data handling
4. Review `Pydantic` for data validation

### Phase 3: Advanced Applications

1. Start with `ChatBot_RAG` for RAG implementation
2. Try `RAG_and_Groq` for performance optimization
3. Explore `SearchEngine` for search capabilities
4. Work with `SQLAgent` for database interactions
5. Experiment with `MathsGPT` for mathematical problem-solving
6. Dive into `NLP` for advanced language processing

## 📁 Project Structure

```
.
├── 1.1-openai/          # Basic OpenAI implementations
├── ChatBot/             # Basic chatbot
├── ChatBot_RAG/         # RAG-enhanced chatbot
├── LCEL/                # LangChain Expression Language
├── LangChain/           # Core LangChain implementations
├── MathsGPT/            # Mathematical problem-solving
├── NLP/                 # Natural Language Processing
├── Pydantic/            # Data validation
├── RAG_and_Groq/        # RAG with Groq
├── SearchEngine/        # AI search engine
├── SQLAgent/            # SQL operations agent
├── TextSummarization/   # Text summarization
├── templates/           # Project templates
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## 🔧 Dependencies

All projects share common dependencies listed in `requirements.txt`. Some projects may have additional specific requirements mentioned in their respective directories.

## 🤝 Contributing

Feel free to contribute to any project by:

1. Forking the repository
2. Creating a new branch
3. Making your changes
4. Submitting a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for their API and models
- LangChain team for their framework
- All contributors to the open-source libraries used in these projects
