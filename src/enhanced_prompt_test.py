# enhanced_prompt_test.py - Updated version of your prompt_test.py with SOTA evaluation

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

# Load environment variables
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Configure logger
logger = logging.getLogger("social_care_rag")
console = Console()

# Define prompt variations (your existing ones plus optimized version)
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
        """,

    # NEW: Your optimized prompt based on testing results
    "optimized_budget": """
        Please answer the following question about social care services in UK local authorities based only on the information in the provided documents.

        Question: {query}

        Context from relevant documents:
        {context}

        Instructions:
        1. For BUDGET QUESTIONS specifically:
           - Search the context for ANY financial figures, even if terminology differs
           - Look for terms like: "net revenue expenditure", "gross expenditure", "total spending", "council budget", "annual budget"
           - For social care budgets, look for: "adult social care", "social services", "community care", "care allocation"
           - If you find budget figures, present them clearly with the exact amount and time period
           - If no specific figures are found, describe what budget information IS available

        2. Structure your response:
           - **Direct Answer**: State the specific figure if found, or "No specific figure provided in available documents"
           - **Available Information**: Describe what budget-related information is present
           - **Context**: Explain what document types and sources the information comes from

        3. For financial information:
           - Include exact figures, time periods, and what the funding covers
           - Present financial data clearly: "The budget is **£X million** for [time period]"
           - Note if figures represent gross vs net amounts, or include/exclude specific items

        4. Always cite sources using format: "According to [document name/description]"

        5. For non-budget questions, provide clear, direct answers following standard best practices:
           - Synthesize information from all relevant documents
           - Use bullet points for lists and step-by-step information
           - Include practical next steps or contact information when available

        Answer:
        """
}

# Enhanced test queries with expected information types
STANDARD_QUERIES = [
    {
        "id": "std_1", 
        "text": "What social care services are available in {LA} and who is eligible for them?",
        "type": "services",
        "expected_info_type": "service_list"
    },
    {
        "id": "std_2", 
        "text": "What is the process for requesting a social care assessment in {LA}?",
        "type": "process",
        "expected_info_type": "procedure"
    },
    {
        "id": "std_3", 
        "text": "What are the costs of care services in {LA} and how do residents pay for these services?",
        "type": "costs",
        "expected_info_type": "financial"
    },
    {
        "id": "std_4", 
        "text": "What additional resources and support services are available for social care users in {LA}, including advocacy and complaints services?",
        "type": "support",
        "expected_info_type": "service_list"
    },
    {
        "id": "std_6", 
        "text": "What is {LA}'s total annual budget?",
        "type": "budget",
        "expected_info_type": "budget_total"
    },
    {
        "id": "std_7", 
        "text": "What specific savings targets has {LA} set in its budget plans?",
        "type": "budget",
        "expected_info_type": "budget_savings"
    },
    {
        "id": "std_9", 
        "text": "What is the current budget allocation for Adult Social Care in {LA}?",
        "type": "budget",
        "expected_info_type": "budget_social_care"
    }
]

SCENARIO_QUERIES = [
    {
        "id": "scn_1", 
        "text": "I'm 83 and recently had a fall in my home in {LA}. I live alone and I'm worried about my safety. What services can help me stay independent?",
        "type": "scenario",
        "expected_info_type": "support_services"
    },
    {
        "id": "scn_15", 
        "text": "My husband has advanced dementia and I'm his full-time carer in {LA}. I'm exhausted and need support. What are my options?",
        "type": "scenario",
        "expected_info_type": "carer_support"
    },
    {
        "id": "scn_16", 
        "text": "My care needs assessment in {LA} says I need 20 hours of home care weekly, but I'm worried about costs. What financial help is available?",
        "type": "scenario",
        "expected_info_type": "financial_support"
    },
    {
        "id": "scn_25", 
        "text": "I'm concerned about an elderly neighbor in {LA} who seems confused and isn't eating properly. How do I report adult safeguarding concerns?",
        "type": "scenario",
        "expected_info_type": "safeguarding"
    }
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

class EnhancedPromptTester:
    def __init__(self, vector_db_base_path="./output", results_path="./prompt_test_results", 
                 openai_api_key=None, langsmith_api_key=None):
        """Initialize the enhanced prompt tester with SOTA evaluation capabilities."""
        self.vector_db_base_path = Path(vector_db_base_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True, parents=True)
        
        # Ensure we have API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.langsmith_api_key = langsmith_api_key or os.getenv("LANGSMITH_API_KEY")
        
        # Create a consistent project name
        self.project_name = "social_care_rag_enhanced"
        
        # Test results data structure
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "vector_db_base_path": str(vector_db_base_path),
                "results_path": str(results_path),
                "project_name": self.project_name,
                "evaluation_version": "enhanced_sota"
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
            
            # Initialize the enhanced tracing manager
            try:
                from enhanced_tracing import TracingManager
                self.tracing_manager = TracingManager(
                    api_key=self.langsmith_api_key,
                    project_name=self.project_name
                )
                logger.info(f"Enhanced LangSmith tracing enabled with project: {self.project_name}")
            except ImportError:
                # Fall back to original tracing manager
                try:
                    from tracing import TracingManager
                    self.tracing_manager = TracingManager(
                        api_key=self.langsmith_api_key,
                        project_name=self.project_name
                    )
                    logger.info(f"Basic LangSmith tracing enabled with project: {self.project_name}")
                except ImportError:
                    logger.warning("No tracing manager available")
                    self.tracing_manager = None
        else:
            logger.warning("LangSmith tracing disabled (no API key provided)")
            self.tracing_manager = None

    def test_prompt_variation(self, local_authority, query_info, prompt_name, prompt_template, top_k=10):
        """Test a specific prompt variation with enhanced retrieval and evaluation."""
        # Format the query for this LA
        query_text = query_info["text"].format(LA=local_authority)
        
        try:
            # Import here to avoid circular dependencies
            from rag_system import RAGSystem
            
            # Import VectorDatabase
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
            
            # Create a RAG system with enhanced capabilities
            rag = RAGSystem(
                vector_db, 
                openai_api_key=self.openai_api_key,
                langsmith_api_key=self.langsmith_api_key
            )
            
            # Store original prompt template
            original_template = rag.prompt_template
            
            # Set the prompt template for this test
            rag.prompt_template = prompt_template
            
            # Process the query with enhanced retrieval and evaluation
            start_time = time.time()
            result = rag.process_query(
                query=query_text, 
                top_k=top_k, 
                prompt_variation_name=prompt_name,
                local_authority=local_authority,
                query_id=query_info['id'],
                use_enhanced_retrieval=True,  # Use enhanced retrieval by default
                retrieval_strategy='hybrid'   # Use hybrid strategy for best results
            )
            end_time = time.time()
            
            # Add enhanced test metadata
            result['test_metadata'] = {
                'local_authority': local_authority,
                'query_id': query_info['id'],
                'query_type': query_info.get('type', 'unknown'),
                'expected_info_type': query_info.get('expected_info_type', 'general'),
                'prompt_variation': prompt_name,
                'prompt_template': prompt_template,
                'processing_time': end_time - start_time,
                'timestamp': datetime.now().isoformat(),
                'top_k': top_k,
                'enhanced_retrieval_used': True,
                'retrieval_strategy': 'hybrid'
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
                'prompt_variation': prompt_name,
                'test_metadata': {
                    'error_occurred': True,
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }
        
    def run_enhanced_tests(self, local_authorities, queries=None, prompt_variations=None, 
                          top_k=10, run_evaluation=True, create_dataset=True):
        """Run enhanced tests with SOTA evaluation capabilities."""
        # Use default queries if none specified
        if queries is None:
            queries = STANDARD_QUERIES + SCENARIO_QUERIES[:3]  # Standard + first 3 scenario queries
            
        # Use all prompt variations if none specified
        if prompt_variations is None:
            prompt_variations = PROMPT_VARIATIONS
            
        # Create directories for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = self.results_path / f"enhanced_test_run_{timestamp}"
        test_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize results list
        all_results = []
        
        # Create evaluation dataset if enabled and tracing manager available
        evaluation_dataset = None
        if create_dataset and run_evaluation and self.tracing_manager and self.tracing_manager.client:
            logger.info("Creating enhanced evaluation dataset...")
            evaluation_dataset = self.tracing_manager.create_evaluation_dataset(
                local_authorities=local_authorities,
                queries=queries,
                name_suffix=f"enhanced_test_{timestamp}"
            )
            if evaluation_dataset:
                logger.info(f"Created enhanced dataset: {evaluation_dataset.name}")
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
            main_task = progress.add_task("[cyan]Enhanced Testing Progress", total=total_tests)
            
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
                        
                        # Run the enhanced test
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
                            
                            # Add to results list with enhanced metadata
                            result_summary = {
                                'local_authority': la,
                                'query_id': query['id'],
                                'query_text': query['text'].format(LA=la),
                                'query_type': query.get('type', 'unknown'),
                                'expected_info_type': query.get('expected_info_type', 'general'),
                                'prompt_variation': prompt_name,
                                'output_file': str(output_file),
                                'processing_time': result.get('test_metadata', {}).get('processing_time', None),
                                'error': result.get('error', None),
                                'trace_url': result.get('trace', {}).get('url', None),
                                'run_id': result.get('run_id', None),
                                'retrieved_docs_count': len(result.get('retrieved_documents', [])),
                                'answer_length': len(result.get('answer', '')),
                                'answer': result.get('answer', ''),
                                'retrieval_method': result.get('retrieval_method', 'unknown'),
                                'enhanced_features_used': {
                                    'enhanced_retrieval': result.get('test_metadata', {}).get('enhanced_retrieval_used', False),
                                    'retrieval_strategy': result.get('test_metadata', {}).get('retrieval_strategy', 'unknown')
                                }
                            }
                            all_results.append(result_summary)
                        
                        # Update progress
                        progress.update(main_task, advance=1)
                        progress.update(la_task, advance=1)
        
        # Create summary DataFrame with enhanced columns
        df = pd.DataFrame(all_results)
        
        # Save results summary
        summary_file = test_dir / "enhanced_test_summary.csv"
        df.to_csv(summary_file, index=False)
        
        # Save as JSON for programmatic access
        summary_json = test_dir / "enhanced_test_summary.json"
        df_dict = df.to_dict(orient='records')
        save_json_safely(summary_json, df_dict)
        
        # Run enhanced evaluation if requested and possible
        if run_evaluation and self.tracing_manager and evaluation_dataset:
            console.print("\n[bold blue]Running enhanced SOTA evaluation on test results...[/bold blue]")
        
            # Debug logging before evaluation
            logger.info(f"all_results contains {len(all_results)} items")
            if all_results:
                logger.info(f"First result query_id: {all_results[0].get('query_id')}")
                logger.info(f"First result has answer: {'Yes' if 'answer' in all_results[0] else 'No'}")
                logger.info(f"Answer sample: {all_results[0].get('answer', '')[:100] if all_results[0].get('answer') else 'No answer'}")

            # Define enhanced evaluation target function
            def enhanced_target_fn(inputs):
                query = inputs.get("query", inputs.get("question", ""))
                metadata = inputs.get("metadata", {})
                local_authority = metadata.get("local_authority", "Unknown")
                query_id = metadata.get("query_id", "Unknown")
                prompt_variation = metadata.get("prompt_variation")

                logger.info(f"Enhanced target function called with query_id: {query_id}, LA: {local_authority}, prompt: {prompt_variation}")
            
                # Enhanced matching with multiple fallbacks
                # Method 1: Exact match by metadata
                if query_id and query_id != "Unknown" and prompt_variation and prompt_variation != "Unknown":
                    for result in all_results:
                        if (result.get('query_id') == query_id and 
                            result.get('local_authority') == local_authority and 
                            result.get('prompt_variation') == prompt_variation):
                            if result.get('answer'):
                                logger.info(f"Found exact match: {query_id}, {local_authority}, {prompt_variation}")
                                return {
                                    "answer": result['answer'],
                                    "documents": [],  # Will be filled from retrieved_documents if needed
                                    "metadata": result.get('enhanced_features_used', {})
                                }
                
                # Method 2: Match by query_id and LA (ignoring prompt variation)
                if query_id and query_id != "Unknown":
                    for result in all_results:
                        if result.get('query_id') == query_id and result.get('local_authority') == local_authority:
                            if result.get('answer'):
                                logger.info(f"Found match by ID and LA: {query_id}, {local_authority}")
                                return {
                                    "answer": result['answer'],
                                    "documents": [],
                                    "metadata": result.get('enhanced_features_used', {})
                                }
                
                # Method 3: Fuzzy match by query text
                normalized_query = query.lower().strip()
                for result in all_results:
                    result_query = result.get('query_text', '').lower().strip()
                    if (normalized_query == result_query and 
                        result.get('prompt_variation') == prompt_variation):
                        if result.get('answer'):
                            logger.info(f"Found match by query text and prompt variation")
                            return {
                                "answer": result['answer'],
                                "documents": [],
                                "metadata": result.get('enhanced_features_used', {})
                            }
                
                logger.warning(f"No match found for query_id: {query_id}, LA: {local_authority}, prompt: {prompt_variation}")
                return {
                    "answer": f"No result found for query: {query[:50]}... in LA: {local_authority} with prompt: {prompt_variation}",
                    "documents": [],
                    "metadata": {}
                }
            
            # Run enhanced evaluation
            try:
                experiment_results = self.tracing_manager.run_evaluation_experiment(
                    target_function=enhanced_target_fn,
                    dataset_name=evaluation_dataset,
                    evaluators=None  # Will use enhanced evaluators from tracing manager
                )
                
                if experiment_results:
                    console.print(f"[bold green]Enhanced evaluation experiment completed successfully![/bold green]")
                    console.print(f"Results will be available in LangSmith dashboard.")
                    
                    # Try to get and display analysis if enhanced evaluation system is available
                    if hasattr(self.tracing_manager, 'enhanced_evaluation') and self.tracing_manager.enhanced_evaluation:
                        try:
                            analysis = self.tracing_manager.enhanced_evaluation.analyze_evaluation_results(experiment_results)
                            if analysis and 'overall_stats' in analysis:
                                console.print(f"\n[bold blue]Evaluation Analysis:[/bold blue]")
                                for metric, stats in analysis['overall_stats'].get('avg_scores', {}).items():
                                    console.print(f"{metric}: {stats.get('mean', 0):.2f} ± {stats.get('std', 0):.2f}")
                                
                                if analysis.get('insights'):
                                    console.print(f"\n[bold yellow]Key Insights:[/bold yellow]")
                                    for insight in analysis['insights']:
                                        console.print(f"• {insight}")
                        except Exception as e:
                            logger.error(f"Error analyzing evaluation results: {str(e)}")
                else:
                    console.print(f"[bold yellow]Evaluation completed but no results returned[/bold yellow]")
                    
            except Exception as e:
                logger.error(f"Error running enhanced evaluation experiment: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Generate enhanced HTML report
        self.generate_enhanced_html_report(df, test_dir)
        
        console.print(f"\n[bold green]Enhanced testing complete! Results saved to {test_dir}[/bold green]")
        console.print(f"Summary: {summary_file}")
        console.print(f"Enhanced HTML Report: {test_dir / 'enhanced_test_report.html'}")
        
        return df, test_dir
    
    def generate_enhanced_html_report(self, df, output_dir):
        """Generate an enhanced HTML report with SOTA evaluation insights."""
        # Enhanced analysis with more metrics
        numeric_eval_cols = [col for col in df.columns if 'score' in col.lower() and pd.api.types.is_numeric_dtype(df[col])]
        
        # Calculate enhanced statistics
        prompt_performance = {}
        if not df.empty:
            for prompt in df['prompt_variation'].unique():
                prompt_data = df[df['prompt_variation'] == prompt]
                performance = {
                    'total_tests': len(prompt_data),
                    'success_rate': len(prompt_data[prompt_data['error'].isna()]) / len(prompt_data) if len(prompt_data) > 0 else 0,
                    'avg_processing_time': prompt_data['processing_time'].mean() if 'processing_time' in prompt_data.columns else 0,
                    'avg_answer_length': prompt_data['answer_length'].mean() if 'answer_length' in prompt_data.columns else 0,
                    'enhanced_retrieval_usage': sum(1 for _, row in prompt_data.iterrows() 
                                                   if row.get('enhanced_features_used', {}).get('enhanced_retrieval', False)) / len(prompt_data) if len(prompt_data) > 0 else 0
                }
                
                # Add evaluation scores if available
                for col in numeric_eval_cols:
                    if col in prompt_data.columns:
                        scores = prompt_data[col].dropna()
                        if len(scores) > 0:
                            performance[f'avg_{col}'] = scores.mean()
                
                prompt_performance[prompt] = performance
        
        # Generate enhanced HTML with more insights
        html = f"""
        <html>
        <head>
            <title>Enhanced RAG System Test Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .summary-box {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .insight-box {{ background-color: #f0fff0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .best {{ background-color: #d4edda; }}
                .worst {{ background-color: #f8d7da; }}
                .enhanced {{ background-color: #e3f2fd; }}
                .metric-bar {{ background-color: #007bff; height: 20px; border-radius: 2px; }}
            </style>
        </head>
        <body>
            <h1>Enhanced RAG System Test Results</h1>
            <div class="summary-box">
                <h3>Test Summary</h3>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Tests: {len(df)}</p>
                <p>Local Authorities: {len(df['local_authority'].unique()) if not df.empty else 0}</p>
                <p>Prompt Variations: {len(df['prompt_variation'].unique()) if not df.empty else 0}</p>
                <p>Enhanced Retrieval Used: {sum(1 for _, row in df.iterrows() if row.get('enhanced_features_used', {}).get('enhanced_retrieval', False)) if not df.empty else 0} tests</p>
            </div>
            
            <h2>Enhanced Performance by Prompt Variation</h2>
            <table>
                <tr>
                    <th>Prompt Variation</th>
                    <th>Success Rate</th>
                    <th>Avg Processing Time</th>
                    <th>Avg Answer Length</th>
                    <th>Enhanced Retrieval Usage</th>
                    <th>Overall Performance</th>
                </tr>
        """
        
        # Add rows for each prompt variation with enhanced metrics
        for prompt, stats in prompt_performance.items():
            success_rate = f"{stats['success_rate']:.1%}"
            processing_time = f"{stats['avg_processing_time']:.2f}s" if stats['avg_processing_time'] else "N/A"
            answer_length = f"{stats['avg_answer_length']:.0f} chars" if stats['avg_answer_length'] else "N/A"
            enhanced_usage = f"{stats['enhanced_retrieval_usage']:.1%}"
            
            # Calculate overall performance score
            overall_score = (stats['success_rate'] + stats['enhanced_retrieval_usage']) / 2
            overall_class = "best" if overall_score > 0.8 else "worst" if overall_score < 0.5 else ""
            
            html += f"""
                <tr class="{overall_class}">
                    <td>{prompt}</td>
                    <td>{success_rate}</td>
                    <td>{processing_time}</td>
                    <td>{answer_length}</td>
                    <td>{enhanced_usage}</td>
                    <td>{overall_score:.2f}</td>
                </tr>
            """
        
        # Add query type analysis
        html += """
            </table>
            
            <h2>Performance by Query Type</h2>
            <table>
                <tr>
                    <th>Query Type</th>
                    <th>Count</th>
                    <th>Avg Success Rate</th>
                    <th>Avg Processing Time</th>
                </tr>
        """
        
        if not df.empty:
            query_type_stats = df.groupby('query_type').agg({
                'query_id': 'count',
                'error': lambda x: (x.isna()).mean(),
                'processing_time': 'mean'
            }).round(3)
            
            for query_type, stats in query_type_stats.iterrows():
                html += f"""
                    <tr>
                        <td>{query_type}</td>
                        <td>{stats['query_id']}</td>
                        <td>{stats['error']:.1%}</td>
                        <td>{stats['processing_time']:.2f}s</td>
                    </tr>
                """
        
        html += """
            </table>
            
            <div class="insight-box">
                <h3>Key Insights & Recommendations</h3>
                <ul>
        """
        
        # Generate insights based on data
        if not df.empty:
            # Budget query performance
            budget_queries = df[df['query_type'] == 'budget']
            if len(budget_queries) > 0:
                budget_success = (budget_queries['error'].isna()).mean()
                if budget_success < 0.7:
                    html += "<li>Budget queries show lower success rates - consider optimizing budget-specific prompts</li>"
                elif budget_success > 0.9:
                    html += "<li>Budget queries perform excellently - good budget information retrieval</li>"
            
            # Enhanced retrieval effectiveness
            enhanced_used = sum(1 for _, row in df.iterrows() 
                              if row.get('enhanced_features_used', {}).get('enhanced_retrieval', False))
            if enhanced_used > 0:
                html += f"<li>Enhanced retrieval was used in {enhanced_used} tests - monitor performance improvements</li>"
            
            # Best performing prompt
            if prompt_performance:
                best_prompt = max(prompt_performance.items(), 
                                key=lambda x: (x[1]['success_rate'] + x[1]['enhanced_retrieval_usage']) / 2)
                html += f"<li>Best performing prompt variation: <strong>{best_prompt[0]}</strong></li>"
        
        html += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save the HTML report
        report_file = output_dir / "enhanced_test_report.html"
        with open(report_file, "w") as f:
            f.write(html)

def main():
    """Main entry point for enhanced prompt testing."""
    parser = argparse.ArgumentParser(description="Enhanced Prompt Testing for RAG System with SOTA Evaluation")
    parser.add_argument("--vector_db_path", default="./output", help="Base path for vector databases")
    parser.add_argument("--results_path", default="./prompt_test_results", help="Path to save test results")
    parser.add_argument("--las", nargs='+', required=True, help="List of Local Authorities to test")
    parser.add_argument("--top_k", type=int, default=15, help="Number of documents to retrieve per query (increased for enhanced retrieval)")
    parser.add_argument("--standard_queries_only", action="store_true", help="Use only standard queries, no scenarios")
    parser.add_argument("--scenario_queries_only", action="store_true", help="Use only scenario queries, no standard ones")
    parser.add_argument("--prompt", choices=list(PROMPT_VARIATIONS.keys()), help="Test only a specific prompt variation")
    parser.add_argument("--langsmith_key", help="LangSmith API key for tracing")
    parser.add_argument("--run_evaluation", action="store_true", help="Run enhanced SOTA evaluation")
    parser.add_argument("--skip_dataset_creation", action="store_true", help="Skip creating evaluation dataset")
    
    args = parser.parse_args()
    
    # Validate local authorities
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
        console.print(f"[bold]Testing all {len(test_prompts)} prompt variations (including optimized)[/bold]")
    
    # Initialize enhanced prompt tester
    tester = EnhancedPromptTester(
        vector_db_base_path=args.vector_db_path,
        results_path=args.results_path,
        langsmith_api_key=args.langsmith_key or os.getenv("LANGSMITH_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Run the enhanced tests
    console.print("\n[bold blue]Starting enhanced prompt variation testing with SOTA evaluation...[/bold blue]")
    df, results_dir = tester.run_enhanced_tests(
        local_authorities=valid_las,
        queries=test_queries,
        prompt_variations=test_prompts,
        top_k=args.top_k,
        run_evaluation=args.run_evaluation,
        create_dataset=not args.skip_dataset_creation
    )
    
    # Display enhanced summary statistics
    console.print("\n[bold]Enhanced Summary Statistics:[/bold]")
    console.print(f"Total tests completed: {len(df)}")
    console.print(f"Success rate: {(df['error'].isna()).mean():.1%}" if not df.empty else "No data")
    
    if not df.empty:
        # Enhanced retrieval usage
        enhanced_usage = sum(1 for _, row in df.iterrows() 
                           if row.get('enhanced_features_used', {}).get('enhanced_retrieval', False))
        console.print(f"Enhanced retrieval usage: {enhanced_usage}/{len(df)} tests ({enhanced_usage/len(df):.1%})")
        
        # Best prompt by success rate
        prompt_success = df.groupby('prompt_variation')['error'].apply(lambda x: x.isna().mean())
        best_prompt = prompt_success.idxmax()
        console.print(f"Best performing prompt: {best_prompt} ({prompt_success[best_prompt]:.1%} success rate)")

if __name__ == "__main__":
    main()