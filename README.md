# LLM-Comparision

### Project Description

This repository contains a web application designed to evaluate the responses of various Language Model (LLM) models in the Question and Answering (Q&A) task. The implemented application integrates Pinecone as the vector database to store information scraped from specified websites, enabling enhanced search capabilities.

#### Models for Comparison:
The application compares the performance of the following LLM models:
1. gpt-3.5-turbo
2. gpt-4
3. Llama-2-70b-chat
4. Falcon-40b-instruct

#### Features:
1. **Web Application Development**: Designed and developed a web application using Python and Flask that accepts user prompts and provides responses from the LLM models.
   
2. **Scraping Logic**: Implemented a scraping mechanism to extract information from the provided website (`https://u.ae/en/information-and-services`) and its subpages. The scraped data is stored in Pinecone for enhanced search capabilities.
   
3. **Query Mechanism**: Upon receiving a new user prompt, the application queries all specified LLMs.
   
4. **Performance Evaluation**: Implemented a mechanism to compare and evaluate the generated outputs from the LLM models using answer relevancy to determine the best-performing model for a given user input.
   
5. **Real-time Streaming**: Utilized Flask SocketIO to enable real-time streaming of responses to the frontend, providing a seamless user experience.

#### Technologies Used:
- **Backend Language**: Python
- **Web Framework**: Flask
- **Vector Database**: Pinecone
- **LLM APIs**:
  - OpenAI API ([Documentation](https://platform.openai.com/docs/api-reference))
  - Replicate ([Llama-2-70b-chat](https://replicate.com/replicate/llama-2-70b-chat), [Falcon-40b-instruct](https://replicate.com/joehoover/falcon-40b-instruct))


