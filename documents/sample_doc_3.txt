# @summary
# Summary file for RAG framework documentation.
# Exports: None
# Deps: None
# @end-summary
  Subject: Retrieval-Augmented Generation (RAG) Explained
  From: Technical Architecture Team <arch-team@acmetech.example.com>
  To: Engineering All-Hands <engineering@acmetech.example.com>
  Date: Thu, 28 Nov 2024 10:30:00 -0500
  MIME-Version: 1.0
  Content-Type: text/plain; charset="utf-8"

  -----

  Hi everyone,

  Following up on last week\u2019s discussion, here\u2019s a write-up on RAG for the team knowledge base.

  == What is RAG? ==

  Retrieval-Augmented Generation, commonly known as RAG, is an AI framework that enhances large language model outputs by incorporating external knowledge retrieval. Instead of relying solely on the knowledge encoded during training, RAG systems dynamically retrieve relevant information from a knowledge base to ground their responses in factual, up-to-date data.

  == Architecture ==

  The RAG architecture consists of two main components: a retriever and a generator.  The retriever searches through a collection of documents or data sources to find information relevant to the input query.   The generator, typically a large language model, then uses both the original query and the retrieved information to produce a response. This two-stage approach significantly reduces hallucinations and improves factual accuracy.

  == Document Processing ==

  Document processing is a critical first step in building a RAG system. Raw documents must be split into manageable chunks, cleaned, and converted into vector embeddings. Chunking strategies include fixed-size splitting, recursive character splitting, and semantic chunking.  The choice of chunk size affects retrieval quality: smaller chunks provide more precise retrieval, while larger chunks preserve more context.

  == Vector Embeddings ==

  Vector embeddings are dense numerical representations of text that capture semantic meaning. Models like BAAI\u2019s BGE series, OpenAI\u2019s text-embedding-ada-002, and Cohere\u2019s embed models convert text into high-dimensional vectors. These vectors are stored in vector databases such as Weaviate, Pinecone, Milvus, or ChromaDB, which enable efficient similarity search through approximate nearest neighbor (ANN) algorithms.

  == Hybrid Search ==

  Hybrid search combines dense vector search with traditional keyword-based search like BM25. This approach leverages the semantic understanding of vector search with the precision of keyword matching. Weaviate natively supports hybrid search, allowing users to balance between vector and keyword components using an alpha parameter. This is particularly effective when queries contain specific technical terms or proper nouns.

  == Reranking ==

  Reranking is an additional step that improves retrieval quality by re-scoring initial search results using a cross-encoder model.  While bi-encoders (used for initial retrieval) encode queries and documents independently, cross-encoders process the query-document pair together, producing more accurate relevance scores at the cost of higher computational overhead.  Models like BAAI\u2019s bge-reranker series are widely used for this purpose.

  == Advanced Techniques ==

  Advanced RAG techniques include query expansion, hypothetical document embeddings (HyDE), multi-step retrieval, and agentic RAG where an LLM orchestrates the retrieval process. Evaluation metrics for RAG systems include context relevance, answer faithfulness, and answer relevance, which can be measured using frameworks like RAGAS.

  -----

  Let me know if you have questions. We\u2019ll be doing a deep-dive in next Friday\u2019s tech talk.

  Best,
  Alex Park
  Principal Architect
  ACME Technologies

  --
  This email and any attachments are confidential and intended solely for the
  use of the individual or entity to whom they are addressed. If you have
  received this email in error, please notify the sender immediately.
