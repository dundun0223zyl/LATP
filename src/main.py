import os
import json
import argparse
import logging
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from rich.markdown import Markdown
import time
from dotenv import load_dotenv

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Import our modules
from information_extractor import InformationExtractor
from text_chunker import TextChunker
from embedding_generator import EmbeddingGenerator
from vector_database import VectorDatabase
from rag_system import RAGSystem
from tracing import TracingManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('main')

# Set up console for rich output
console = Console()

def load_config():
    """Load configuration from config.json file."""
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        # Check if API key needs to be loaded from environment
        if config.get("openai_api_key") == "${OPENAI_API_KEY}" or not config.get("openai_api_key"):
            # Get API key from environment variable
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                config["openai_api_key"] = api_key
                logger.info("Using OpenAI API key from environment variables")
            else:
                logger.warning("No OpenAI API key found in environment variables or config.json")
                logger.warning("The system will run in retrieval-only mode")
                config["openai_api_key"] = ""
        
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        
        # Get API key from environment variable for fallback config
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("No OpenAI API key found in environment variables")
            logger.warning("The system will run in retrieval-only mode")
        
        return {
            "openai_api_key": api_key,
            "chunk_size": 1024,
            "chunk_overlap": 200,
            "top_k": 10,
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_db_path": "./output/vector_db",
            "results_path": "./output/results"
        }

def format_answer_for_display(result):
    """Format the RAG system's answer for better display."""
    query = result['query']
    answer = result['answer']
    
    # Format the sources section
    sources = []
    if 'retrieved_documents' in result:
        for i, doc in enumerate(result['retrieved_documents']):
            source_name = doc['metadata'].get('source', 'Unknown source')
            # Clean up the source name
            clean_source = source_name.replace('-', ' ').replace('_', ' ')
            if '.' in clean_source:
                clean_source = clean_source.split('.')[0]  # Remove file extension
            
            doc_type = doc['metadata'].get('type', 'Unknown type')
            source_url = doc['metadata'].get('source_url', '')
            
            source_info = f"- {clean_source} ({doc_type})"
            if source_url:
                source_info += f" - [URL]({source_url})"
            
            # Add page count for PDFs
            if doc_type == 'pdf' and 'page_count' in doc['metadata']:
                source_info += f" - {doc['metadata']['page_count']} pages"
                
            sources.append(source_info)
    
    sources_text = "\n".join(sources) if sources else "No sources available"
    
    # Create a formatted markdown output
    formatted_result = f"""
# Social Care Information

**Question:** {query}

**Answer:**
{answer}

---
**Sources Referenced:**
{sources_text}
"""
    return formatted_result

def main():
    parser = argparse.ArgumentParser(description="Social Care RAG System")
    parser.add_argument("--crawl_dir", required=True, help="Directory containing crawl output")
    parser.add_argument("--build_db", action="store_true", help="Build/update the vector database")
    parser.add_argument("--query", help="Query to process (if not building DB)")
    parser.add_argument("--raw", action="store_true", help="Display raw output without formatting")
    parser.add_argument("--langsmith_key", help="LangChain Smith API key")
    parser.add_argument("--test_prompts", action="store_true", help="Test different prompt variations")
    parser.add_argument("--evaluate", action="store_true", help="Automatically evaluate response quality")
    parser.add_argument("--vector_db_path", help="Custom path for vector database")  # New argument
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Use command-line vector DB path if provided, otherwise use config value
    vector_db_path = Path(args.vector_db_path if args.vector_db_path else config["vector_db_path"])
    console.print(f"[cyan]Using vector database at: {vector_db_path}[/cyan]")
    
    # Create output directories
    results_dir = Path(config["results_path"])
    results_dir.mkdir(exist_ok=True, parents=True)
    
    vector_db_path.mkdir(exist_ok=True, parents=True)
    
    # Set up LangSmith environment variables
    langsmith_api_key = args.langsmith_key or os.getenv("LANGSMITH_API_KEY")
    if langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Use V2 tracing for better performance
        os.environ["LANGCHAIN_PROJECT"] = "social_care_rag"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com"
    
        # These are for backward compatibility
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = "social_care_rag"
    
        console.print("[green]LangSmith tracing enabled[/green]")
    else:
        console.print("[yellow]LangSmith tracing disabled (no API key provided)[/yellow]")
    
    # Build/update the vector database if requested
    if args.build_db:
        console.print("[bold green]Building vector database...[/bold green]")
        
        with Progress() as progress:
            # Step 1: Extract information from crawled data
            task1 = progress.add_task("[cyan]Extracting documents...", total=1)
            extractor = InformationExtractor(args.crawl_dir)
            documents = extractor.extract_all()
            progress.update(task1, completed=1)
            
            console.print(f"[green]Extracted {len(documents)} documents[/green]")
            
            # Save a sample of extracted documents for reference
            sample_docs_file = results_dir / "sample_extracted_documents.json"
            with open(sample_docs_file, "w") as f:
                json.dump(documents[:5], f, indent=2)
            console.print(f"[cyan]Saved sample documents to {sample_docs_file}[/cyan]")
            
            # Step 2: Chunk the documents
            task2 = progress.add_task("[cyan]Chunking documents...", total=1)
            chunker = TextChunker(chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"])
            chunks = chunker.chunk_documents(documents)
            progress.update(task2, completed=1)
            
            console.print(f"[green]Created {len(chunks)} chunks[/green]")
            
            # Save a sample of chunks for reference
            sample_chunks_file = results_dir / "sample_chunks.json"
            with open(sample_chunks_file, "w") as f:
                json.dump(chunks[:5], f, indent=2)
            console.print(f"[cyan]Saved sample chunks to {sample_chunks_file}[/cyan]")
            
            # Step 3: Generate embeddings
            task3 = progress.add_task("[cyan]Generating embeddings...", total=1)
            embedder = EmbeddingGenerator(model_name=config["embedding_model"])
            embedded_chunks = embedder.generate_embeddings(chunks)
            progress.update(task3, completed=1)
            
            console.print(f"[green]Generated embeddings for {len(embedded_chunks)} chunks[/green]")
            
            # Step 4: Store in vector database
            task4 = progress.add_task("[cyan]Storing in vector database...", total=1)
            vector_db = VectorDatabase(persist_directory=str(vector_db_path))  # Use the custom path
            success = vector_db.add_documents(embedded_chunks)
            progress.update(task4, completed=1)
            
            if success:
                console.print("[bold green]Vector database build complete![/bold green]")
            else:
                console.print("[bold red]Error building vector database.[/bold red]")
    
    # Process a query if provided
    if args.query:
        console.print(f"[bold blue]Processing query: {args.query}[/bold blue]")
        
        # Initialize the vector database with the custom path
        vector_db = VectorDatabase(persist_directory=str(vector_db_path))
        
        # Show LLM status based on API key
        if not config["openai_api_key"]:
            console.print("[yellow]No OpenAI API key provided. Running in retrieval-only mode.[/yellow]")
        
        # Initialize the RAG system
        rag = RAGSystem(
            vector_db, 
            embedding_model=config["embedding_model"],
            openai_api_key=config["openai_api_key"] or os.getenv("OPENAI_API_KEY"),
            langsmith_api_key=langsmith_api_key
        )
        
        # Process the query
        start_time = time.time()
        result = rag.process_query(args.query, top_k=config["top_k"])
        end_time = time.time()
        
        # Evaluate the response if requested
        if args.evaluate and 'feedback_session' in result:
            result = rag.evaluate_response(result)
            
            if 'evaluations' in result:
                console.print("\n[bold blue]Automatic Evaluations:[/bold blue]")
                for criterion, data in result['evaluations'].items():
                    if criterion != 'overall_tags':
                        score = data.get('score')
                        comment = data.get('comment')
                        if score is not None:
                            console.print(f"[cyan]{criterion}:[/cyan] {score:.2f} - {comment}")
                
                # Display tags
                tags = result['evaluations'].get('overall_tags', [])
                if tags:
                    console.print(f"[green]Tags:[/green] {', '.join(tags)}")
        
        # Test prompt variations if requested
        if args.test_prompts and 'retrieved_documents' in result:
            console.print("\n[bold blue]Testing Prompt Variations:[/bold blue]")
            
            # Get the formatted context
            context = result['context']
            
            # Run the comparison
            prompt_results = rag.test_prompt_variations(args.query, context, result['retrieved_documents'])
            
            # Display a summary of results
            if prompt_results:
                console.print("Tested the following prompt variations:")
                for version, data in prompt_results.items():
                    if 'error' in data:
                        console.print(f"[red]{version}: Error - {data['error']}[/red]")
                    else:
                        response_preview = data['response'][:100] + "..." if len(data['response']) > 100 else data['response']
                        console.print(f"[green]{version}:[/green] {response_preview}")
                        
                console.print("\nView detailed comparison in LangSmith UI")
        
        # Display trace information if available
        if 'trace' in result:
            console.print("\n[bold blue]LangSmith Tracing:[/bold blue]")
            console.print(f"Run ID: {result['trace']['run_id']}")
            console.print(f"View detailed trace: {result['trace']['url']}")
            console.print("You can provide feedback and evaluate this response in LangSmith")
        
        # Save the result
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"query_result_{timestamp}.json"
        with open(output_file, "w") as f:
            # Remove embeddings from the output to save space
            if "retrieved_documents" in result:
                for doc in result["retrieved_documents"]:
                    if "embedding" in doc:
                        del doc["embedding"]
            
            json.dump(result, f, indent=2)
        
        # Print the answer
        if args.raw:
            # Original raw output format
            console.print("\n" + "="*80)
            console.print("[bold]QUERY:[/bold]", result['query'])
            console.print("="*80)
            console.print("[bold]ANSWER:[/bold]")
            console.print(result['answer'])
            console.print("="*80)
        else:
            # Enhanced formatting with Markdown
            formatted_output = format_answer_for_display(result)
            console.print(Markdown(formatted_output))
            
        console.print(f"Query processed in {end_time - start_time:.2f} seconds")
        console.print(f"Full results saved to: {output_file}")
    
    # If neither build_db nor query was specified, show help
    if not args.build_db and not args.query:
        parser.print_help()

if __name__ == "__main__":
    main()