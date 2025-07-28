import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger('text_chunker')

class TextChunker:
    def __init__(self, chunk_size=1024, chunk_overlap=200):
        """Initialize the chunker with configurable parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info(f"Initialized chunker with size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_documents(self, documents):
        """
        Split documents into chunks with metadata preserved.
        
        Args:
            documents: List of dicts with 'content' and 'metadata' keys
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        for doc in documents:
            try:
                content = doc['content']
                metadata = doc['metadata']
                
                # Split the text into chunks
                text_chunks = self.text_splitter.split_text(content)
                
                # Create document chunks with metadata
                for i, chunk_text in enumerate(text_chunks):
                    chunk = {
                        'content': chunk_text,
                        'metadata': {
                            **metadata,  # Include original metadata
                            'chunk_id': i,
                            'total_chunks': len(text_chunks)
                        }
                    }
                    chunks.append(chunk)
                
                logger.info(f"Created {len(text_chunks)} chunks from document: {metadata['filename']}")
            except Exception as e:
                logger.error(f"Error chunking document {doc.get('metadata', {}).get('filename', 'unknown')}: {str(e)}")
        
        logger.info(f"Completed chunking. Total chunks created: {len(chunks)}")
        return chunks