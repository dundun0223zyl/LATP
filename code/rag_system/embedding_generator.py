import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import tqdm

logger = logging.getLogger('embedding_generator')

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize with a specified embedding model."""
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {str(e)}")
            raise
    
    def generate_embeddings(self, chunks):
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of dicts with 'content' and 'metadata' keys
            
        Returns:
            List of chunks with embeddings added
        """
        embedded_chunks = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        num_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in tqdm.tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings", total=num_batches):
            batch = chunks[i:i+batch_size]
            try:
                # Extract text content for the batch
                texts = [chunk['content'] for chunk in batch]
                
                # Generate embeddings
                embeddings = self.model.encode(texts)
                
                # Add embeddings to chunks
                for j, chunk in enumerate(batch):
                    embedded_chunk = {
                        'content': chunk['content'],
                        'embedding': embeddings[j].tolist(),  # Convert numpy array to list for storage
                        'metadata': chunk['metadata']
                    }
                    embedded_chunks.append(embedded_chunk)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {str(e)}")
                # Try to process chunks one by one in case of batch failure
                for chunk in batch:
                    try:
                        text = chunk['content']
                        embedding = self.model.encode([text])[0]
                        embedded_chunk = {
                            'content': text,
                            'embedding': embedding.tolist(),
                            'metadata': chunk['metadata']
                        }
                        embedded_chunks.append(embedded_chunk)
                    except Exception as e2:
                        logger.error(f"Error generating embedding for individual chunk: {str(e2)}")
        
        logger.info(f"Completed embedding generation. Total embedded chunks: {len(embedded_chunks)}")
        return embedded_chunks