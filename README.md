

# Mental Health Chatbot - Omdena 🌱

## Project Description
- The Mental Health Chatbot is a tool designed to assist in identifying potential mental health issues such as stress, anxiety, and depression based on users' inputs. The chatbot leverages advanced machine learning models and embeddings to analyze text data and provide relevant feedback.

## Features
1. Natural Language Processing: Users can interact with the chatbot using natural language.
2. Mental Health Detection: The chatbot evaluates the user's input to detect signs of mental health issues.
3. Advanced Models: Utilizes Google Palm's text-bison model and HuggingFace's Instructor Embeddings for accurate analysis.
4. Vector Database: Uses FAISS (Facebook AI Similarity Search) for efficient similarity search and retrieval of context.
5. Streamlit Interface: Provides an easy-to-use web interface for user interaction.

Setup and Installation

1. Clone the repository:
``` bash
git clone https://github.com/Meet19aug/Mental-Health-Chatbot.git
cd Mental-Health-Chatbot
```

2. Create and activate a virtual environment:

``` bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
``` bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a .env file in the project root and add your OpenAI API key:
```
GOOGLE_API_KEY=your_google_api_key
```

Run the Streamlit application:

``` bash
streamlit run main.py
```

## Usage
- Open the Streamlit application in your web browser.
- Enter your thoughts or feelings in the provided text input.
- The chatbot will analyze your input and provide feedback on potential mental health issues.

## Files Description
- dreaddit/: Contains CSV files with Reddit data used for training and validation.
- faiss_index/: Contains files related to the FAISS vector index.
- google_palm_basic_q_and_a.ipynb: Jupyter notebook with examples of using Google Palm's model.
- langchain_helper.py: Helper functions for setting up the vector database and the QA chain.
- main.py: The main script for running the Streamlit application.
- requirements.txt: List of dependencies required for the project.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

