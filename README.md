<!-- @summary
This project, named RAG, is designed to integrate and enhance retrieval augmented generation systems using Python. It includes modules for embedding management, knowledge graph construction, and vector store integration, along with directories for configuration settings, technical documentation, logs, processed files, and prompts.
@end-summary -->

# RAG

## Overview
This project integrates and enhances retrieval augmented generation (RAG) systems using Python. It comprises core modules for embedding management, knowledge graph construction, and vector store integration, alongside directories for configuration settings, technical documentation, logs, processed files, and prompts.

## Architecture
The architecture of this project includes major components such as the embedding manager, knowledge graph builder, and vector store integrator. These components work together to facilitate efficient data retrieval and generation within a RAG framework.

## Directory Map

| Directory | Purpose |
| --- | --- |
| config/ | Contains configuration settings for the RAG system, including paths to directories and models used in operation. |
| documents/ | Technical documentation files covering Python programming basics, machine learning fundamentals, and an overview of the RAG framework. |
| logs/ | Logs related to query processing, detailing iterations, results, reformulations, and confidences without exports or imports. |
| processed/ | Organized into JSON chunks and cleaned markdown formats, containing processed documents for Python, machine learning, and the RAG framework. |
| prompts/ | Documentation for evaluating query quality and reformulating queries in a technical knowledge base system. |
| src/ | Core modules for embedding management, knowledge graph construction, and vector store integration, including subdirectories for document ingestion and retrieval functionalities. |

## Entry Points
- `ingest.py`: Main script for ingesting documents into the RAG system.
- `query.py`: Script for processing user queries within the RAG framework.

## Key Configuration
Configuration settings are managed in the `config/` directory, where paths to directories and models used by the system are defined.