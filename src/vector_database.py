import logging
import os
import chromadb
import uuid

logger = logging.getLogger('vector_database')

class VectorDatabase:
    def __init__(self, persist_directory="./output/vector_db"):
        """Initialize the vector database with persistence."""
        self.persist_directory = persist_directory
        try:
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(
                name="social_care_documents"
            )
            logger.info(f"Initialized vector database at {persist_directory}")
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            raise
    
    def add_documents(self, embedded_chunks):
        try:
            # Prepare data for Chroma
            ids = [str(uuid.uuid4()) for _ in range(len(embedded_chunks))]
            documents = [chunk['content'] for chunk in embedded_chunks]
            embeddings = [chunk['embedding'] for chunk in embedded_chunks]
            
            # Clean metadata - convert all values to strings to avoid type issues
            metadatas = []
            for chunk in embedded_chunks:
                clean_metadata = {}
                for key, value in chunk['metadata'].items():
                    # Convert values to simple string, number, or boolean
                    if isinstance(value, (list, dict)):
                        clean_metadata[key] = str(value)
                    else:
                        clean_metadata[key] = value
                metadatas.append(clean_metadata)
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(embedded_chunks), batch_size):
                end_idx = min(i + batch_size, len(embedded_chunks))
                self.collection.add(
                    ids=ids[i:end_idx],
                    documents=documents[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
                logger.info(f"Added batch {i//batch_size + 1}/{(len(embedded_chunks)-1)//batch_size + 1} to vector database")
            
            logger.info(f"Successfully added {len(embedded_chunks)} documents to vector database")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {str(e)}")
            return False
    
    def similarity_search(self, query_embedding, top_k=5):
        """
        Perform similarity search on the vector database.
        
        Args:
            query_embedding: The embedding vector for the query
            top_k: Number of results to return
            
        Returns:
            List of matching documents with content and metadata
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            documents = []
            for i in range(len(results['documents'][0])):
                doc = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []