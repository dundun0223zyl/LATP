import os
import time
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from tracing import TracingManager

# At the top of your prompt_test.py file
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# For debugging, print the loaded values
print(f"OPENAI_API_KEY loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
print(f"LANGSMITH_API_KEY loaded: {'Yes' if os.getenv('LANGSMITH_API_KEY') else 'No'}")


# Configure logger
logger = logging.getLogger("social_care_rag")
console = Console()

# Define prompt variations
PROMPT_VARIATIONS = {
    "original": """
        Please answer the following question about social care services in UK local authorities based only on the information in the provided documents.

        Question: {query}

        Context from relevant documents:
        {context}

        Instructions:
        1. Base your answer ONLY on the provided context. If information isn't available, clearly state: "I don't have information about this in the available documents."
        2. When referencing sources, include both the document title and source URL when available. For PDF or DOC files, specify page numbers when relevant (e.g., "According to [Document Title] on page 3...").
        3. Format your answer in clear paragraphs for readability.
        4. Be concise and specific in your response.
        5. If the question is about budgets, finances or savings, provide exact figures when available in the context and cite the source clearly.
        6. If the information appears to be presented in a way that would be difficult for the general public to understand, mention this in your answer.

        Answer:
        """,

    "structured": """
        Please answer the following question about social care services in UK local authorities using ONLY the information from the provided documents.

        Question: {query}

        Context from relevant documents:
        {context}

        Instructions:
        1. Structure your answer with these headings when relevant:
           - SUMMARY (1-2 sentence direct answer)
           - AVAILABLE SERVICES (list key services mentioned)
           - ELIGIBILITY CRITERIA (who can access services)
           - FINANCIAL INFORMATION (costs, budgets, or funding details)
           - HOW TO ACCESS (application process or contact information)

        2. For each key piece of information, cite your source in brackets like this: [Document X]. For PDFs, include page numbers: [Document X, p.5].

        3. If the provided documents don't contain information to answer part of the question, clearly state: "The documents don't provide information about [specific aspect]."

        4. Present financial figures exactly as they appear in the documents with proper citation.

        5. Use bullet points for lists to improve readability.

        Answer:
        """,

    "financial_focus": """
        Please answer the following question about social care services in UK local authorities based exclusively on the provided documents.

        Question: {query}

        Context from relevant documents:
        {context}

        Instructions:
        1. Base your answer ONLY on the provided context. Never introduce outside information.

        2. FINANCIAL INFORMATION PRIORITY:
           - Always highlight financial figures in bold: **£24.5 million**
           - Report exact amounts, percentages, and time periods as stated in the documents
           - Clarify whether figures represent annual budgets, one-time funding, or other timeframes
           - Note if financial information is presented unclearly or without adequate context
           - If financial data is missing but would be relevant, explicitly note this gap

        3. For citations:
           - Financial information: Include document title, date, page number, and URL
           - Other information: Include document title and URL
           
        4. Be concise but comprehensive, focusing on factual information rather than interpretation.

        5. If information is unavailable, state specifically: "The available documents do not provide information about [specific aspect of the query]."

        Answer:
        """,

    "accessibility": """
        Please answer the following question about social care services in UK local authorities using only the information in the provided documents.

        Question: {query}

        Context from relevant documents:
        {context}

        Instructions:
        1. Begin with a "Plain English Summary" of 2-3 sentences that answers the question in the simplest possible terms.

        2. When analyzing the information:
           - Compare information across different documents when available (noting agreements and discrepancies)
           - Assess how accessible the information would be to the general public
           - Identify any technical terms or jargon that might be confusing and explain them
           - Note where important information appears to be missing or unclear

        3. Use this citation format: "According to [brief document description] ([URL if available])"

        4. For financial information, always include:
           - The exact amount
           - The time period it covers
           - What specifically it funds
           - Whether this represents an increase/decrease from previous periods (if mentioned)

        5. If you cannot answer any part of the question from the provided documents, clearly state what specific information is missing.

        Answer:
        """,

    "confidence": """
        Please answer the following question about social care services in UK local authorities using only the information provided in the documents.

        Question: {query}

        Context from relevant documents:
        {context}

        Instructions:
        1. Begin your response by stating your confidence level in the answer:
           - "HIGH CONFIDENCE: The documents provide clear, consistent information on this topic."
           - "MEDIUM CONFIDENCE: The documents provide some information, but it may be incomplete or unclear."
           - "LOW CONFIDENCE: The documents provide minimal or potentially outdated information."
           - "UNABLE TO ANSWER: The documents do not contain relevant information on this topic."

        2. Present information with appropriate qualifiers that reflect the certainty of the information:
           - For clear, definitive information: "The documents clearly state that..."
           - For implied information: "The documents suggest that..."
           - For incomplete information: "The documents partially address this by mentioning..."

        3. When citing sources, use this format: "[Document title/description] ([Year if available], [URL or page number])"

        4. For financial data, always include the date or period the figures apply to and note if they appear to be current or historical.

        5. If documents present conflicting information, acknowledge these discrepancies and present both perspectives.

        6. Format your response in short, clear paragraphs with bullet points where appropriate.

        Answer:
        """
}

# Define test queries
STANDARD_QUERIES = [
    {"id": "std_1", "text": "What social care services are available in {LA} and who is eligible for them?"},
    {"id": "std_2", "text": "What is the process for requesting a social care assessment in {LA}?"},
    {"id": "std_3", "text": "What are the costs of care services in {LA} and how do residents pay for these services?"},
    {"id": "std_4", "text": "What additional resources and support services are available for social care users in {LA}, including advocacy and complaints services?"},
    {"id": "std_6", "text": "What is {LA}'s total annual budget for the current financial year?"},
    {"id": "std_7", "text": "What specific savings targets has {LA} set in its budget plans?"},
    {"id": "std_9", "text": "What is the current budget allocation for Adult Social Care in {LA}?"}
]

SCENARIO_QUERIES = [
    {"id": "scn_1", "text": "I'm 83 and recently had a fall in my home in {LA}. I live alone and I'm worried about my safety. What services can help me stay independent?"},
    {"id": "scn_15", "text": "My husband has advanced dementia and I'm his full-time carer in {LA}. I'm exhausted and need support. What are my options?"},
    {"id": "scn_16", "text": "My care needs assessment in {LA} says I need 20 hours of home care weekly, but I'm worried about costs. What financial help is available?"},
    {"id": "scn_25", "text": "I'm concerned about an elderly neighbor in {LA} who seems confused and isn't eating properly. How do I report adult safeguarding concerns?"}
]

def save_json_safely(filepath, data):
    """Save JSON data with error handling."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {str(e)}")
        # Try again with a simplified approach
        try:
            simplified_data = json.loads(json.dumps(data, default=str))
            with open(filepath, 'w') as f:
                json.dump(simplified_data, f, indent=2)
            return True
        except Exception as e2:
            logger.error(f"Error in second attempt to save JSON: {str(e2)}")
            return False

class PromptTester:
    def __init__(self, vector_db_base_path="./output", results_path="./prompt_test_results", openai_api_key=None, langsmith_api_key=None):
        """Initialize the prompt tester with paths and configuration."""
        self.vector_db_base_path = Path(vector_db_base_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True, parents=True)
        
        # Ensure we have API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.langsmith_api_key = langsmith_api_key or os.getenv("LANGSMITH_API_KEY")
        
        # Create a consistent project name
        self.project_name = "social_care_rag"  # Use a consistent name instead of timestamp
        
        # Test results data structure
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "vector_db_base_path": str(vector_db_base_path),
                "results_path": str(results_path),
                "project_name": self.project_name
            },
            "test_runs": []
        }
        
        # Set up LangSmith environment variables if key is provided
        if self.langsmith_api_key:
            # First unset any existing environment variables to avoid conflicts
            for var in ["LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "LANGSMITH_TRACING"]:
                if var in os.environ:
                    del os.environ[var]
                    
            # Set fresh environment variables
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_PROJECT"] = self.project_name
            os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"
            
            # Initialize the tracing manager
            self.tracing_manager = TracingManager(
                api_key=self.langsmith_api_key,
                project_name=self.project_name
            )
            
            logger.info(f"LangSmith tracing enabled with project: {self.project_name}")
        else:
            logger.warning("LangSmith tracing disabled (no API key provided)")
            self.tracing_manager = None

    def _safe_get_evaluation(self, result, metric):
        """Safely get evaluation metric even if structure is incomplete."""
        try:
            if not result or not isinstance(result.get('evaluations'), dict):
                return None
            eval_data = result.get('evaluations', {}).get(metric, {})
            return eval_data.get('score', None) if isinstance(eval_data, dict) else None
        except:
            return None

    def test_prompt_variation(self, local_authority, query_info, prompt_name, prompt_template, top_k=10):
        """Test a specific prompt variation on a query for a local authority."""
        # Format the query for this LA
        query_text = query_info["text"].format(LA=local_authority)
        
        try:
            # Import here to avoid circular dependencies
            from rag_system import RAGSystem
            
            # You'll need to implement or import VectorDatabase
            try:
                from vector_database import VectorDatabase
            except ImportError:
                # If not available, create a simple mock for demonstration
                class VectorDatabase:
                    def __init__(self, persist_directory):
                        self.persist_directory = persist_directory
                        logger.info(f"Initialized vector database from {persist_directory}")
                    
                    def similarity_search(self, query_embedding, top_k=10):
                        # This would be implemented in your actual code
                        return []
            
            # Initialize RAG system
            vector_db_path = self.vector_db_base_path / f"{local_authority.lower()}_db"
            if not vector_db_path.exists():
                logger.error(f"Vector database not found for {local_authority} at {vector_db_path}")
                return None
                
            # Initialize the vector database
            vector_db = VectorDatabase(persist_directory=str(vector_db_path))
            
            # Create a RAG system
            rag = RAGSystem(
                vector_db, 
                openai_api_key=self.openai_api_key,
                langsmith_api_key=self.langsmith_api_key
            )
            
            # Store original prompt template
            original_template = rag.prompt_template
            
            # Set the prompt template for this test
            rag.prompt_template = prompt_template
            
            # Process the query with the custom prompt
            start_time = time.time()
            result = rag.process_query(
                query=query_text, 
                top_k=top_k, 
                prompt_variation_name=prompt_name,
                local_authority=local_authority,
                query_id=query_info['id']
            )
            end_time = time.time()
            
            # Add test metadata
            result['test_metadata'] = {
                'local_authority': local_authority,
                'query_id': query_info['id'],
                'prompt_variation': prompt_name,
                'prompt_template': prompt_template,
                'processing_time': end_time - start_time,
                'timestamp': datetime.now().isoformat(),
                'top_k': top_k
            }
            
            # Restore original template
            rag.prompt_template = original_template
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing {prompt_name} for {local_authority} on {query_info['id']}: {str(e)}")
            return {
                'error': str(e),
                'local_authority': local_authority,
                'query_id': query_info['id'],
                'prompt_variation': prompt_name
            }
        
    def run_tests(self, local_authorities, queries=None, prompt_variations=None, top_k=10, run_evaluation=True):
        """Run tests for the specified local authorities, queries, and prompt variations."""
        # Use default queries if none specified
        if queries is None:
            queries = STANDARD_QUERIES + SCENARIO_QUERIES[:3]  # Standard + first 3 scenario queries
            
        # Use all prompt variations if none specified
        if prompt_variations is None:
            prompt_variations = PROMPT_VARIATIONS
            
        # Create directories for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = self.results_path / f"test_run_{timestamp}"
        test_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize results list
        all_results = []
        
        # Create evaluation dataset if enabled
        evaluation_dataset = None
        if run_evaluation and self.tracing_manager and self.tracing_manager.client:
            logger.info("Creating evaluation dataset...")
            evaluation_dataset = self.tracing_manager.create_evaluation_dataset(
                local_authorities=local_authorities,
                queries=queries,
                name_suffix=f"test_{timestamp}"
            )
            if evaluation_dataset:
                logger.info(f"Created dataset: {evaluation_dataset.name}")
            else:
                logger.warning("Failed to create dataset. Evaluation will be skipped.")
                run_evaluation = False
        
        # Create progress bars
        total_tests = len(local_authorities) * len(queries) * len(prompt_variations)
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        ) as progress:
            main_task = progress.add_task("[cyan]Overall Testing Progress", total=total_tests)
            
            # Loop through local authorities
            for la in local_authorities:
                la_dir = test_dir / la
                la_dir.mkdir(exist_ok=True)
                
                la_task = progress.add_task(f"[green]Testing {la}", total=len(queries) * len(prompt_variations))
                
                # Loop through queries
                for query in queries:
                    query_dir = la_dir / query["id"]
                    query_dir.mkdir(exist_ok=True)
                    
                    # Loop through prompt variations
                    for prompt_name, prompt_template in prompt_variations.items():
                        test_description = f"{la} | {query['id']} | {prompt_name}"
                        progress.update(main_task, description=f"[cyan]Testing: {test_description}")
                        
                        # Run the test
                        result = self.test_prompt_variation(
                            local_authority=la,
                            query_info=query,
                            prompt_name=prompt_name,
                            prompt_template=prompt_template,
                            top_k=top_k
                        )
                        
                        # Save the result
                        if result:
                            output_file = query_dir / f"{prompt_name}.json"
                            
                            # Remove embeddings to save space
                            if "retrieved_documents" in result:
                                for doc in result["retrieved_documents"]:
                                    if "embedding" in doc:
                                        del doc["embedding"]
                            
                            # Use safe JSON saving function
                            save_json_safely(output_file, result)
                            
                            # Add to results list
                            all_results.append({
                                'local_authority': la,
                                'query_id': query['id'],
                                'query_text': query['text'].format(LA=la),
                                'prompt_variation': prompt_name,
                                'output_file': str(output_file),
                                'processing_time': result.get('test_metadata', {}).get('processing_time', None),
                                'error': result.get('error', None),
                                'trace_url': result.get('trace', {}).get('url', None),
                                'run_id': result.get('run_id', None),
                                'retrieved_docs_count': len(result.get('retrieved_documents', [])),
                                'answer_length': len(result.get('answer', '')),
                                'answer': result.get('answer', '')
                            })
                        
                        # Update progress
                        progress.update(main_task, advance=1)
                        progress.update(la_task, advance=1)
        
        # Create summary DataFrame
        df = pd.DataFrame(all_results)
        
        # Save results summary - use safe JSON saving
        summary_file = test_dir / "test_summary.csv"
        df.to_csv(summary_file, index=False)
        
        # Also save as JSON for programmatic access - use safe JSON saving
        summary_json = test_dir / "test_summary.json"
        df_dict = df.to_dict(orient='records')
        save_json_safely(summary_json, df_dict)
        
        # Run evaluation if requested and possible
        if run_evaluation and self.tracing_manager and evaluation_dataset:
            console.print("\n[bold blue]Running evaluation on test results...[/bold blue]")
        
            # Debug logging before evaluation
            logger.info(f"all_results contains {len(all_results)} items")
            if all_results:
                logger.info(f"First result query_id: {all_results[0].get('query_id')}")
                logger.info(f"First result has answer: {'Yes' if 'answer' in all_results[0] else 'No'}")
                logger.info(f"Answer sample: {all_results[0].get('answer', '')[:100] if all_results[0].get('answer') else 'No answer'}")

            
            # Define evaluation target function
            def target_fn(inputs):
                query = inputs.get("query", inputs.get("question", ""))
                # Extract local_authority and query_id from metadata if available
                metadata = inputs.get("metadata", {})
                local_authority = metadata.get("local_authority", "Unknown")
                query_id = metadata.get("query_id", "Unknown")
                prompt_variation = metadata.get("prompt_variation")

                logger.info(f"Target function called with query_id: {query_id}, LA: {local_authority}, prompt: {prompt_variation}")
            
                # Method 1: Match by query_id and local_authority
                if query_id and query_id != "Unknown" and prompt_variation and prompt_variation != "Unknown":
                    for result in all_results:
                        if (result.get('query_id') == query_id and 
                            result.get('local_authority') == local_authority and 
                            result.get('prompt_variation') == prompt_variation):
                            if result.get('answer'):
                                logger.info(f"Found exact match: {query_id}, {local_authority}, {prompt_variation}")
                                return {"answer": result['answer']}
                
                # If no match with prompt variation, fall back to just query_id and local_authority
                if query_id and query_id != "Unknown":
                    for result in all_results:
                        if result.get('query_id') == query_id and result.get('local_authority') == local_authority:
                            if result.get('answer'):
                                logger.info(f"Found match by ID and LA (ignoring prompt): {query_id}, {local_authority}")
                                return {"answer": result['answer']}
    
                
                # Method 2: Match by normalized query text
                normalized_query = query.lower().strip()
                for result in all_results:
                    result_query = result.get('query_text', '').lower().strip()
                    if (normalized_query == result_query and 
                        result.get('prompt_variation') == prompt_variation):
                        if result.get('answer'):
                            logger.info(f"Found match by query text and prompt variation")
                            return {"answer": result['answer']}
                
                # Method 3: Try partial matching as last resort
                for result in all_results:
                    result_query = result.get('query_text', '').lower().strip()
                    # Only try partial match for longer queries to avoid false positives
                    if (len(normalized_query) > 20 and 
                        (normalized_query in result_query or result_query in normalized_query) and
                        result.get('prompt_variation') == prompt_variation):
                        if result.get('answer'):
                           logger.info(f"Found partial match with matching prompt variation")
                           return {"answer": result['answer']}
            
                logger.warning(f"No match found for query_id: {query_id}, LA: {local_authority}, prompt: {prompt_variation}")
                return {"answer": f"No result found for query: {query[:50]}... in LA: {local_authority} with prompt: {prompt_variation}"}
        
            
            # Set up evaluators
            evaluators = self.tracing_manager.setup_evaluators()
            
            # Run evaluation using the latest method
            from langsmith import evaluate

            evaluator_functions = list(evaluators.values())
            logger.info(f"Using {len(evaluator_functions)} evaluator functions: {list(evaluators.keys())}")

            
            try:
                experiment_results = evaluate(
                    target_fn,
                    data=evaluation_dataset.id,
                    evaluators=evaluator_functions,
                    experiment_prefix=f"eval_{evaluation_dataset.name}",
                    max_concurrency=4
                )
                
                console.print(f"[bold green]Evaluation experiment started: {experiment_results}[/bold green]")
                console.print(f"Results will be available in LangSmith dashboard when complete.")
            except Exception as e:
                logger.error(f"Error running evaluation experiment: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Generate a more detailed HTML report
        self.generate_html_report(df, test_dir)
        
        console.print(f"\n[bold green]Testing complete! Results saved to {test_dir}[/bold green]")
        console.print(f"Summary: {summary_file}")
        console.print(f"HTML Report: {test_dir / 'test_report.html'}")
        
        return df, test_dir
    
    def generate_html_report(self, df, output_dir):
        """Generate an HTML report from the test results."""
        # Check if evaluation metrics exist
        eval_columns = ['relevance_score', 'completeness_score', 'accuracy_score']
        missing_columns = [col for col in eval_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing evaluation columns: {missing_columns}")
            # Add empty columns for missing metrics
            for col in missing_columns:
                df[col] = None
        
        # IMPROVED: Make a copy of the dataframe to avoid modifying the original
        df_numeric = df.copy()
        
        # IMPROVED: Check column types before trying numeric conversion
        for col in df_numeric.columns:
            if col in ['local_authority', 'query_id', 'prompt_variation', 'query_text', 'output_file', 
                      'trace_url', 'run_id', 'error', 'answer']:
                # Skip string columns that should never be numeric
                continue
                
            # For potentially numeric columns, convert safely with coercion
            try:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
            except:
                logger.warning(f"Could not convert column '{col}' to numeric")
        
        # IMPROVED: Only select columns that are actually numeric for aggregation
        numeric_eval_cols = [col for col in eval_columns if col in df_numeric.columns 
                            and pd.api.types.is_numeric_dtype(df_numeric[col])]
        
        # Calculate average scores per prompt variation - handle missing values
        if numeric_eval_cols:
            avg_scores = df_numeric.groupby('prompt_variation')[numeric_eval_cols].mean()
        else:
            # Create empty DataFrame if no evaluation metrics available
            avg_scores = pd.DataFrame(index=df_numeric['prompt_variation'].unique())
        
        # Add processing time if available
        if 'processing_time' in df_numeric.columns and pd.api.types.is_numeric_dtype(df_numeric['processing_time']):
            if avg_scores.empty:
                avg_scores = df_numeric.groupby('prompt_variation')['processing_time'].mean().to_frame()
            else:
                avg_times = df_numeric.groupby('prompt_variation')['processing_time'].mean()
                avg_scores = pd.concat([avg_scores, avg_times], axis=1)
        
        # Add a combined score column if possible
        if numeric_eval_cols and not avg_scores.empty:
            try:
                avg_scores['combined_score'] = avg_scores[numeric_eval_cols].mean(axis=1)
                # Sort by combined score
                avg_scores = avg_scores.sort_values('combined_score', ascending=False)
            except Exception as e:
                logger.warning(f"Could not calculate combined score: {str(e)}")
                # Don't sort if we can't calculate combined score
        elif 'processing_time' in avg_scores.columns:
            # Sort by processing time if no evaluation metrics
            avg_scores = avg_scores.sort_values('processing_time')
            
        # Generate HTML
        html = f"""
        <html>
        <head>
            <title>Prompt Variation Test Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .best {{ background-color: #d4edda; }}
                .worst {{ background-color: #f8d7da; }}
                .metric-bar {{ background-color: #007bff; height: 20px; border-radius: 2px; }}
            </style>
        </head>
        <body>
            <h1>Prompt Variation Test Results</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Overall Performance by Prompt Variation</h2>
            <table>
                <tr>
                    <th>Prompt Variation</th>
                    <th>Processing Time</th>
                    <th>Combined Score</th>
                    <th>Relevance</th>
                    <th>Completeness</th>
                    <th>Accuracy</th>
                </tr>
        """
        
        # Add rows for each prompt variation
        for prompt, row in avg_scores.iterrows():
            # Determine if this is the best prompt
            is_best = False
            if 'combined_score' in row and not pd.isna(row['combined_score']):
                is_best = row['combined_score'] == avg_scores['combined_score'].max()
            row_class = "best" if is_best else ""
            
            # Format values safely
            combined = f"{row['combined_score']:.2f}" if 'combined_score' in row and not pd.isna(row['combined_score']) else "N/A"
            relevance = f"{row['relevance_score']:.2f}" if not pd.isna(row['relevance_score']) else "N/A"
            completeness = f"{row['completeness_score']:.2f}" if not pd.isna(row['completeness_score']) else "N/A"
            accuracy = f"{row['accuracy_score']:.2f}" if not pd.isna(row['accuracy_score']) else "N/A"
            proc_time = f"{row['processing_time']:.2f}s" if 'processing_time' in row and not pd.isna(row['processing_time']) else "N/A"
            
            html += f"""
                <tr class="{row_class}">
                    <td>{prompt}</td>
                    <td>{proc_time}</td>
                    <td>{combined}</td>
                    <td>{relevance}</td>
                    <td>{completeness}</td>
                    <td>{accuracy}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Performance by Local Authority</h2>
            <table>
                <tr>
                    <th>Local Authority</th>
                    <th>Average Relevance</th>
                    <th>Average Completeness</th>
                    <th>Average Accuracy</th>
                </tr>
        """
        
        # Add rows for each local authority
        la_scores = df.groupby('local_authority')[eval_columns].mean()
        for la, row in la_scores.iterrows():
            # Format values safely
            relevance = f"{row['relevance_score']:.2f}" if not pd.isna(row['relevance_score']) else "N/A"
            completeness = f"{row['completeness_score']:.2f}" if not pd.isna(row['completeness_score']) else "N/A"
            accuracy = f"{row['accuracy_score']:.2f}" if not pd.isna(row['accuracy_score']) else "N/A"
            
            html += f"""
                <tr>
                    <td>{la}</td>
                    <td>{relevance}</td>
                    <td>{completeness}</td>
                    <td>{accuracy}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Performance by Query Type</h2>
            <table>
                <tr>
                    <th>Query ID</th>
                    <th>Average Relevance</th>
                    <th>Average Completeness</th>
                    <th>Average Accuracy</th>
                </tr>
        """
        
        # Add rows for each query type
        query_scores = df.groupby('query_id')[eval_columns].mean()
        for query, row in query_scores.iterrows():
            # Format values safely
            relevance = f"{row['relevance_score']:.2f}" if not pd.isna(row['relevance_score']) else "N/A"
            completeness = f"{row['completeness_score']:.2f}" if not pd.isna(row['completeness_score']) else "N/A"
            accuracy = f"{row['accuracy_score']:.2f}" if not pd.isna(row['accuracy_score']) else "N/A"
            
            html += f"""
                <tr>
                    <td>{query}</td>
                    <td>{relevance}</td>
                    <td>{completeness}</td>
                    <td>{accuracy}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>All Test Results</h2>
            <table>
                <tr>
                    <th>LA</th>
                    <th>Query</th>
                    <th>Prompt</th>
                    <th>Time (s)</th>
                    <th>Docs</th>
                    <th>LangSmith</th>
                </tr>
        """
        
        # Add rows for all tests
        for _, row in df.iterrows():
            time_str = f"{row['processing_time']:.2f}" if pd.notna(row['processing_time']) else "N/A"
            langsmith_link = f"<a href='{row['trace_url']}'>View Trace</a>" if pd.notna(row['trace_url']) else "N/A"
            
            html += f"""
                <tr>
                    <td>{row['local_authority']}</td>
                    <td>{row['query_id']}</td>
                    <td>{row['prompt_variation']}</td>
                    <td>{time_str}</td>
                    <td>{row['retrieved_docs_count']}</td>
                    <td>{langsmith_link}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        # Save the HTML report
        report_file = output_dir / "test_report.html"
        with open(report_file, "w") as f:
            f.write(html)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test prompt variations for RAG System")
    parser.add_argument("--vector_db_path", default="./output", help="Base path for vector databases")
    parser.add_argument("--results_path", default="./prompt_test_results", help="Path to save test results")
    parser.add_argument("--las", nargs='+', required=True, help="List of Local Authorities to test")
    parser.add_argument("--top_k", type=int, default=10, help="Number of documents to retrieve per query")
    parser.add_argument("--standard_queries_only", action="store_true", help="Use only standard queries, no scenarios")
    parser.add_argument("--scenario_queries_only", action="store_true", help="Use only scenario queries, no standard ones")
    parser.add_argument("--prompt", choices=list(PROMPT_VARIATIONS.keys()), help="Test only a specific prompt variation")
    parser.add_argument("--langsmith_key", help="LangSmith API key for tracing")
    parser.add_argument("--run_evaluation", action="store_true", help="Run LLM-as-judge evaluation")
    
    args = parser.parse_args()
    
    # Validate local authorities - ensure we have DB folders for them
    valid_las = []
    for la in args.las:
        db_path = Path(args.vector_db_path) / f"{la.lower()}_db"
        if db_path.exists():
            valid_las.append(la)
        else:
            logger.warning(f"No vector database found for {la} at {db_path}")
    
    if not valid_las:
        console.print("[bold red]Error: No valid local authorities with vector databases found![/bold red]")
        return
    
    console.print(f"[bold green]Will test {len(valid_las)} local authorities: {', '.join(valid_las)}[/bold green]")
    
    # Determine which queries to use
    if args.standard_queries_only and args.scenario_queries_only:
        console.print("[bold yellow]Warning: Both standard and scenario query flags set. Using all queries.[/bold yellow]")
        test_queries = STANDARD_QUERIES + SCENARIO_QUERIES
    elif args.standard_queries_only:
        test_queries = STANDARD_QUERIES
        console.print(f"[bold]Using {len(test_queries)} standard queries only[/bold]")
    elif args.scenario_queries_only:
        test_queries = SCENARIO_QUERIES
        console.print(f"[bold]Using {len(test_queries)} scenario queries only[/bold]")
    else:
        # Default: Use all standard queries and a subset of scenario queries
        test_queries = STANDARD_QUERIES + SCENARIO_QUERIES[:4]
        console.print(f"[bold]Using {len(test_queries)} queries ({len(STANDARD_QUERIES)} standard + 4 scenarios)[/bold]")
    
    # Determine which prompt variations to use
    if args.prompt:
        test_prompts = {args.prompt: PROMPT_VARIATIONS[args.prompt]}
        console.print(f"[bold]Testing only the '{args.prompt}' prompt variation[/bold]")
    else:
        test_prompts = PROMPT_VARIATIONS
        console.print(f"[bold]Testing all {len(test_prompts)} prompt variations[/bold]")
    
    # Initialize prompt tester
    tester = PromptTester(
        vector_db_base_path=args.vector_db_path,
        results_path=args.results_path,
        langsmith_api_key=args.langsmith_key or os.getenv("LANGSMITH_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Run the tests
    console.print("\n[bold blue]Starting prompt variation testing...[/bold blue]")
    df, results_dir = tester.run_tests(
        local_authorities=valid_las,
        queries=test_queries,
        prompt_variations=test_prompts,
        top_k=args.top_k,
        run_evaluation=args.run_evaluation
    )
    
    # Display summary statistics
    console.print("\n[bold]Summary Statistics:[/bold]")

    # Average scores by prompt variation - safely handle numeric columns
    console.print("\n[bold cyan]Average Scores by Prompt Variation:[/bold cyan]")

    # Create a copy for numeric operations
    df_numeric = df.copy()

    # Convert only numeric columns - safely handle any errors
    for col in df_numeric.columns:
        if col in ['local_authority', 'query_id', 'prompt_variation', 'query_text', 'output_file', 
                'trace_url', 'run_id', 'error', 'answer']:
            # Skip string columns
            continue
        
        # Convert potential numeric columns safely
        try:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
        except:
            logger.warning(f"Could not convert column '{col}' to numeric")

    # Only use numeric columns for groupby mean calculation
    numeric_cols = ['processing_time', 'retrieved_docs_count', 'answer_length']
    numeric_cols = [col for col in numeric_cols if col in df_numeric.columns 
                and pd.api.types.is_numeric_dtype(df_numeric[col])]

    # Get averages only for truly numeric columns
    avg_by_prompt = df_numeric.groupby('prompt_variation')[numeric_cols].mean().reset_index()

    # Find the best prompt variation based on processing time - with safety checks
    if 'processing_time' in avg_by_prompt.columns and not avg_by_prompt.empty:
        best_idx = avg_by_prompt['processing_time'].idxmin()
        if pd.notna(best_idx):
            best_prompt = avg_by_prompt.loc[best_idx, 'prompt_variation']
            best_time = avg_by_prompt.loc[best_idx, 'processing_time']
        else:
            best_prompt = "Unknown"
            best_time = float('nan')
    else:
        best_prompt = "Unknown"
        best_time = float('nan')

    # Print the table with safe formatting
    for _, row in avg_by_prompt.iterrows():
        time_str = f"Processing time={row['processing_time']:.2f}s, " if 'processing_time' in row and pd.notna(row['processing_time']) else "Processing time=N/A, "
        docs_str = f"Documents={row['retrieved_docs_count']:.1f}, " if 'retrieved_docs_count' in row and pd.notna(row['retrieved_docs_count']) else "Documents=N/A, "
        len_str = f"Answer length={row['answer_length']:.1f}" if 'answer_length' in row and pd.notna(row['answer_length']) else "Answer length=N/A"
    
        console.print(f"{row['prompt_variation']}: {time_str}{docs_str}{len_str}")

    # Only show best prompt if we found a valid one
    if best_prompt != "Unknown" and pd.notna(best_time):
        console.print(f"\n[bold green]Best performing prompt variation (by speed): {best_prompt} (Time: {best_time:.2f}s)[/bold green]")
    else:
        console.print("\n[bold yellow]Could not determine best prompt variation[/bold yellow]")

if __name__ == "__main__":
    main()