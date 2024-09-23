This Chatbot is a user-friendly research tool designed for effortless information retrieval. Users can input article URLs or web links and ask questions to receive relevant insights from those articles.

Features 

-Load URLs or upload text files containing URLs to fetch article content.

-Process article content through LangChain's UnstructuredURL Loader.

-Construct an embedding vector using Google GeminiAI embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information.

-Interact with the LLM's (Chatbot) by inputting queries and receiving answers along with source URLs.

Run the Streamlit app by executing: "streamlit run main.py" in the terminal.

Project Structure

=main.py: The main Streamlit application script.

-requirements.txt: A list of required Python packages for the project.

-faiss_index: A file to store the FAISS index.

-.env: Configuration file for storing your Gemini API key.
