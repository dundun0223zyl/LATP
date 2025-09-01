import os
import time
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from tracing import TracingManager  # Use the simple tracing manager

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logger
logger = logging.getLogger("social_care_rag")
console = Console()

# Your optimized prompt template
OPTIMIZED_PROMPT = """
Please answer the following question about social care services in UK local authorities based only on the information in the provided documents.

Question: {query}

Context from relevant documents:
{context}

Instructions:
1. Provide a clear, direct answer that would help someone needing this information. If information isn't available, state: "The available documents don't provide specific information about [aspect]."

2. Synthesize information from ALL relevant documents to provide the most complete answer rather than analyzing documents individually.

3. Structure your response with:
   - A concise 1-2 sentence summary first
   - Relevant details organized in short, clear paragraphs
   - Bullet points for lists or step-by-step information
   - Bold text for key information the user should notice

4. For financial information:
   - Include exact figures, time periods, and what the funding covers
   - Present financial data clearly: "The adult social care budget is **£X million** for [time period]"

5. Always include practical next steps or resources:
   - Provide contact information, URLs, or application methods mentioned in the documents
   - Suggest where the person could find more information

6. When citing sources, use a simple format: "According to [document name/description]" or include URLs where available.

Answer:
"""

# Standard queries - divided into Service vs Financial (matching Phase 1)
SERVICE_STANDARD_QUERIES = [
    {"id": "std_1", "text": "What social care services are available in {LA} and who is eligible for them?", "category": "service_standard"},
    {"id": "std_2", "text": "What is the process for requesting a social care assessment in {LA}?", "category": "service_standard"},
    {"id": "std_3", "text": "What are the costs of care services in {LA} and how do residents pay for these services?", "category": "service_standard"},
    {"id": "std_4", "text": "What additional resources and support services are available for social care users in {LA}, including advocacy and complaints services?", "category": "service_standard"},
    {"id": "std_5", "text": "What accessibility options does {LA} provide for accessing social care information, including translations or easy-read formats?", "category": "service_standard"}
]

FINANCIAL_STANDARD_QUERIES = [
    {"id": "std_6", "text": "What is {LA}'s total annual budget for the current financial year?", "category": "financial_standard"},
    {"id": "std_7", "text": "What specific savings targets has {LA} set in its budget plans?", "category": "financial_standard"},
    {"id": "std_8", "text": "How much of {LA}'s planned savings are targeted to come from social care services?", "category": "financial_standard"},
    {"id": "std_9", "text": "What is the current budget allocation for Adult Social Care in {LA}?", "category": "financial_standard"}
]

# Combine for standard queries
STANDARD_QUERIES = SERVICE_STANDARD_QUERIES + FINANCIAL_STANDARD_QUERIES

# Adult Services Scenarios (1-9)
ADULT_SCENARIOS = [
    {"id": "scn_1", "text": "I'm 83 and recently had a fall in my home in {LA}. I live alone and I'm worried about my safety. What services can help me stay independent?", "category": "adult_care_scenarios"},
    {"id": "scn_2", "text": "My mother is 90 and lives in {LA}. She's becoming increasingly forgetful and I'm concerned she might be developing dementia. How can I arrange an assessment?", "category": "adult_care_scenarios"},
    {"id": "scn_3", "text": "I'm 75 living in {LA} and can no longer drive due to vision problems. How can I get to my medical appointments and do grocery shopping?", "category": "adult_care_scenarios"},
    {"id": "scn_4", "text": "My son has autism and we've just moved to {LA}. He's 15 and will need transition planning for adulthood soon. What support services are available?", "category": "transition_scenarios"},
    {"id": "scn_5", "text": "I use a wheelchair and my bathroom in my {LA} council house is no longer accessible for me. What home adaptation support is available?", "category": "housing_scenarios"},
    {"id": "scn_6", "text": "I have progressive multiple sclerosis and live in {LA}. I'm finding it harder to prepare meals and get dressed. What help can I receive?", "category": "adult_care_scenarios"},
    {"id": "scn_7", "text": "My teenage daughter in {LA} is experiencing severe anxiety that's affecting her school attendance. What mental health support is available for young people?", "category": "mental_health_scenarios"},
    {"id": "scn_8", "text": "I'm a working adult in {LA} struggling with depression. I need support but can only attend appointments in evenings or weekends. What options do I have?", "category": "mental_health_scenarios"},
    {"id": "scn_9", "text": "I'm caring for my partner who has bipolar disorder in {LA}. Their condition has worsened recently and I'm feeling overwhelmed. What crisis support exists?", "category": "mental_health_scenarios"}
]

# Children and Family Services Scenarios (10-12)
FAMILY_SCENARIOS = [
    {"id": "scn_10", "text": "I'm a single parent in {LA} with three children under 10. I'm struggling financially and emotionally. What family support services could help us?", "category": "family_scenarios"},
    {"id": "scn_11", "text": "We're considering becoming foster parents in {LA}. What is the application process and what support would we receive?", "category": "family_scenarios"},
    {"id": "scn_12", "text": "My 4-year-old in {LA} has developmental delays. How can I get an education, health and care (EHC) assessment?", "category": "family_scenarios"}
]

# Carer Support Scenarios (13-15)
CARER_SCENARIOS = [
    {"id": "scn_13", "text": "I work full-time but also care for my elderly father with Parkinson's in {LA}. What respite care options exist so I can have occasional breaks?", "category": "carer_support_scenarios"},
    {"id": "scn_14", "text": "I'm 16 and help care for my mother who has MS in {LA}. This is affecting my schoolwork. Is there any support for young carers?", "category": "carer_support_scenarios"},
    {"id": "scn_15", "text": "My husband has advanced dementia and I'm his full-time carer in {LA}. I'm exhausted and need support. What are my options?", "category": "carer_support_scenarios"}
]

# Financial Support Scenarios (16-18)
FINANCIAL_SCENARIOS = [
    {"id": "scn_16", "text": "My care needs assessment in {LA} says I need 20 hours of home care weekly, but I'm worried about costs. What financial help is available?", "category": "financial_scenarios"},
    {"id": "scn_17", "text": "I receive Attendance Allowance but my care needs have increased in {LA}. Can the council provide additional financial support?", "category": "financial_scenarios"},
    {"id": "scn_18", "text": "My son with learning disabilities is turning 18 in {LA}. How will his financial support change when he transitions to adult services?", "category": "financial_scenarios"}
]

# Housing and Accommodation Scenarios (19-21)
HOUSING_SCENARIOS = [
    {"id": "scn_19", "text": "I'm 67 with mobility issues and can no longer manage stairs in my {LA} home. What sheltered housing options are available?", "category": "housing_scenarios"},
    {"id": "scn_20", "text": "My adult daughter has Down syndrome in {LA} and wants to live more independently. What supported living options exist?", "category": "housing_scenarios"},
    {"id": "scn_21", "text": "I'm being discharged from hospital in {LA} after a stroke and can't return to my previous home. What temporary accommodation can social services provide?", "category": "housing_scenarios"}
]

# Assessment and Review Scenarios (22-24)
ASSESSMENT_SCENARIOS = [
    {"id": "scn_22", "text": "I've been told I need a financial assessment for care in {LA}. What documents should I prepare and how is my contribution calculated?", "category": "assessment_scenarios"},
    {"id": "scn_23", "text": "My wife's needs have changed significantly since her last assessment in {LA}. How do we request a reassessment and how long will it take?", "category": "assessment_scenarios"},
    {"id": "scn_24", "text": "I'm moving from {LA} to Southampton. Will I need a new care needs assessment or can my current care package transfer?", "category": "assessment_scenarios"}
]

# Emergency and Crisis Scenarios (25-27)
EMERGENCY_SCENARIOS = [
    {"id": "scn_25", "text": "I'm concerned about an elderly neighbor in {LA} who seems confused and isn't eating properly. How do I report adult safeguarding concerns?", "category": "emergency_scenarios"},
    {"id": "scn_26", "text": "My son with learning disabilities is displaying challenging behavior that I can no longer manage at home in {LA}. What emergency support is available?", "category": "emergency_scenarios"},
    {"id": "scn_27", "text": "I'm experiencing domestic abuse in {LA} and need to leave with my children. What emergency social care and housing support can I access?", "category": "emergency_scenarios"}
]

# Transition and Recovery Scenarios (28-30)
TRANSITION_SCENARIOS = [
    {"id": "scn_28", "text": "My father is being discharged from hospital in {LA} after a hip replacement. What reablement services can help during his recovery?", "category": "transition_scenarios"},
    {"id": "scn_29", "text": "My daughter with complex disabilities is moving from children's to adult services in {LA}. How do we ensure a smooth transition?", "category": "transition_scenarios"},
    {"id": "scn_30", "text": "I'm moving my mother with dementia from her own home to a care home in {LA}. What support is available during this transition?", "category": "transition_scenarios"}
]

# Mental Health Scenarios (extracted from adult scenarios)
MENTAL_HEALTH_SCENARIOS = [
    {"id": "scn_7", "text": "My teenage daughter in {LA} is experiencing severe anxiety that's affecting her school attendance. What mental health support is available for young people?", "category": "mental_health_scenarios"},
    {"id": "scn_8", "text": "I'm a working adult in {LA} struggling with depression. I need support but can only attend appointments in evenings or weekends. What options do I have?", "category": "mental_health_scenarios"},
    {"id": "scn_9", "text": "I'm caring for my partner who has bipolar disorder in {LA}. Their condition has worsened recently and I'm feeling overwhelmed. What crisis support exists?", "category": "mental_health_scenarios"}
]

# Combine all scenario queries
ALL_SCENARIO_QUERIES = (
    ADULT_SCENARIOS + FAMILY_SCENARIOS + CARER_SCENARIOS + 
    FINANCIAL_SCENARIOS + HOUSING_SCENARIOS + ASSESSMENT_SCENARIOS + 
    EMERGENCY_SCENARIOS + TRANSITION_SCENARIOS
)

def save_json_safely(filepath, data):
    """Save JSON data with error handling."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {str(e)}")
        return False

class OptimizedPromptTester:
    def __init__(self, vector_db_base_path="./output", results_path="./optimized_prompt_results", 
                 openai_api_key=None, langsmith_api_key=None):
        """Initialize the optimized prompt tester."""
        self.vector_db_base_path = Path(vector_db_base_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True, parents=True)
        
        # Ensure we have API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.langsmith_api_key = langsmith_api_key or os.getenv("LANGSMITH_API_KEY")
        
        # Create a consistent project name
        self.project_name = "social_care_rag_optimized"
        
        # Set up LangSmith environment variables if key is provided
        if self.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_PROJECT"] = self.project_name
            os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"
            
            # Initialize the simple tracing manager
            self.tracing_manager = TracingManager(
                api_key=self.langsmith_api_key,
                project_name=self.project_name
            )
            
            logger.info(f"Optimized LangSmith tracing enabled with project: {self.project_name}")
        else:
            logger.warning("LangSmith tracing disabled (no API key provided)")
            self.tracing_manager = None

    def test_optimized_prompt(self, local_authority, query_info, top_k=10):
        """Test the optimized prompt on a specific query."""
        # Format the query for this LA
        query_text = query_info["text"].format(LA=local_authority)
        
        try:
            # Import your RAG system (use the simplified version)
            from rag_system import RAGSystem  # Use your existing RAG system
            from vector_database import VectorDatabase
            
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
            
            # Set the optimized prompt template
            rag.prompt_template = OPTIMIZED_PROMPT
            
            # Process the query
            start_time = time.time()
            result = rag.process_query(
                query=query_text, 
                top_k=top_k, 
                prompt_variation_name="optimized",
                local_authority=local_authority,
                query_id=query_info['id'],
                use_enhanced_retrieval=True  # Use enhanced retrieval
            )
            end_time = time.time()
            
            # Add test metadata
            result['test_metadata'] = {
                'local_authority': local_authority,
                'query_id': query_info['id'],
                'query_category': query_info.get('category', 'unknown'),
                'prompt_variation': 'optimized',
                'prompt_template': OPTIMIZED_PROMPT,
                'processing_time': end_time - start_time,
                'timestamp': datetime.now().isoformat(),
                'top_k': top_k
            }
            
            # Restore original template
            rag.prompt_template = original_template
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing optimized prompt for {local_authority} on {query_info['id']}: {str(e)}")
            return {
                'error': str(e),
                'local_authority': local_authority,
                'query_id': query_info['id'],
                'prompt_variation': 'optimized',
                'test_metadata': {
                    'error_occurred': True,
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }

    def determine_queries_by_category(self, query_categories):
        """Determine which queries to use based on categories - updated version."""
        selected_queries = []
        
        if query_categories is None:
            # Default: Use all standard + sample scenarios for comprehensive testing
            selected_queries = STANDARD_QUERIES.copy()
            selected_queries.extend(ADULT_SCENARIOS[:2])  # First 2 adult scenarios  
            selected_queries.extend(FINANCIAL_SCENARIOS)   # All financial scenarios
            selected_queries.extend(EMERGENCY_SCENARIOS[:2])  # First 2 emergency scenarios
        else:
            # Use specific categories
            for category in query_categories:
                if category == "service_standard":
                    selected_queries.extend(SERVICE_STANDARD_QUERIES)
                elif category == "financial_standard":
                    selected_queries.extend(FINANCIAL_STANDARD_QUERIES)
                elif category == "standard":
                    selected_queries.extend(STANDARD_QUERIES)
                elif category == "adult":
                    selected_queries.extend(ADULT_SCENARIOS)
                elif category == "family":
                    selected_queries.extend(FAMILY_SCENARIOS)
                elif category == "carer":
                    selected_queries.extend(CARER_SCENARIOS)
                elif category == "financial":
                    selected_queries.extend(FINANCIAL_SCENARIOS)
                elif category == "housing":
                    selected_queries.extend(HOUSING_SCENARIOS)
                elif category == "assessment":
                    selected_queries.extend(ASSESSMENT_SCENARIOS)
                elif category == "emergency":
                    selected_queries.extend(EMERGENCY_SCENARIOS)
                elif category == "transition":
                    selected_queries.extend(TRANSITION_SCENARIOS)
                elif category == "mental_health":
                    selected_queries.extend(MENTAL_HEALTH_SCENARIOS)
                elif category == "all_scenarios":
                    selected_queries.extend(ALL_SCENARIO_QUERIES)
                else:
                    logger.warning(f"Unknown category: {category}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in selected_queries:
            if query['id'] not in seen:
                unique_queries.append(query)
                seen.add(query['id'])
        
        return unique_queries
        
    def run_optimized_tests(self, local_authorities, query_categories=None, 
                           top_k=10, run_evaluation=True):
        """Run tests with the optimized prompt on selected query categories."""
        
        # Use the new method to determine queries
        unique_queries = self.determine_queries_by_category(query_categories)
        
        console.print(f"[bold]Testing {len(unique_queries)} queries with optimized prompt[/bold]")
        
        # Create directories for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = self.results_path / f"optimized_test_run_{timestamp}"
        test_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize results list
        all_results = []
        
        # Create evaluation dataset if enabled and tracing manager available
        evaluation_dataset = None
        if run_evaluation and self.tracing_manager and self.tracing_manager.client:
            logger.info("Creating optimized evaluation dataset...")
            evaluation_dataset = self.tracing_manager.create_evaluation_dataset(
                local_authorities=local_authorities,
                queries=unique_queries,
                name_suffix=f"optimized_test_{timestamp}"
            )
            if evaluation_dataset:
                logger.info(f"Created optimized dataset: {evaluation_dataset.name}")
            else:
                logger.warning("Failed to create dataset. Evaluation will be skipped.")
                run_evaluation = False
        
        # Create progress bars
        total_tests = len(local_authorities) * len(unique_queries)
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        ) as progress:
            main_task = progress.add_task("[cyan]Optimized Testing Progress", total=total_tests)
            
            # Loop through local authorities
            for la in local_authorities:
                la_dir = test_dir / la
                la_dir.mkdir(exist_ok=True)
                
                # Loop through queries
                for query in unique_queries:
                    query_dir = la_dir / query["id"]
                    query_dir.mkdir(exist_ok=True)
                    
                    test_description = f"{la} | {query['id']} | {query.get('category', 'unknown')}"
                    progress.update(main_task, description=f"[cyan]Testing: {test_description}")
                    
                    # Run the test
                    result = self.test_optimized_prompt(
                        local_authority=la,
                        query_info=query,
                        top_k=top_k
                    )
                    
                    # Save the result
                    if result:
                        output_file = query_dir / "optimized.json"
                        
                        # Remove embeddings to save space
                        if "retrieved_documents" in result:
                            for doc in result["retrieved_documents"]:
                                if "embedding" in doc:
                                    del doc["embedding"]
                        
                        # Use safe JSON saving function
                        save_json_safely(output_file, result)
                        
                        # Add to results list
                        result_summary = {
                            'local_authority': la,
                            'query_id': query['id'],
                            'query_category': query.get('category', 'unknown'),
                            'query_text': query['text'].format(LA=la),
                            'prompt_variation': 'optimized',
                            'output_file': str(output_file),
                            'processing_time': result.get('test_metadata', {}).get('processing_time', None),
                            'error': result.get('error', None),
                            'trace_url': result.get('trace', {}).get('url', None),
                            'run_id': result.get('run_id', None),
                            'retrieved_docs_count': len(result.get('retrieved_documents', [])),
                            'answer_length': len(result.get('answer', '')),
                            'answer': result.get('answer', ''),
                            'retrieval_method': result.get('retrieval_method', 'unknown')
                        }
                        all_results.append(result_summary)
                    
                    # Update progress
                    progress.update(main_task, advance=1)
        
        # Create summary DataFrame
        df = pd.DataFrame(all_results)
        
        # Save results summary
        summary_file = test_dir / "optimized_test_summary.csv"
        df.to_csv(summary_file, index=False)
        
        # Save as JSON for programmatic access
        summary_json = test_dir / "optimized_test_summary.json"
        df_dict = df.to_dict(orient='records')
        save_json_safely(summary_json, df_dict)
        
        # Run evaluation if requested and possible
        if run_evaluation and self.tracing_manager and evaluation_dataset:
            console.print("\n[bold blue]Running evaluation on optimized prompt results...[/bold blue]")
        
            # Debug logging before evaluation
            logger.info(f"all_results contains {len(all_results)} items")

            # Define evaluation target function
            def optimized_target_fn(inputs):
                """Target function for optimized prompt evaluation."""
                query = inputs.get("query", inputs.get("question", ""))
                
                # Try to find matching result by query text
                for result in all_results:
                    result_query = result.get('query_text', '')
                    if query.strip() == result_query.strip():
                        answer = result.get('answer', '')
                        
                        # Try to construct context from the result if available
                        context = ""
                        if 'retrieved_documents' in result:
                            context_parts = []
                            for doc in result.get('retrieved_documents', [])[:3]:  # Use top 3 docs
                                content = doc.get('content', '')[:500]  # Truncate for evaluation
                                context_parts.append(content)
                            context = "\n\n".join(context_parts)
                        
                        return {
                            "answer": answer,
                            "context": context  # Include context for faithfulness evaluation
                        }
                
                # Fallback if no exact match found
                return {
                    "answer": f"No result found for query: {query[:50]}...",
                    "context": ""
                }
            
            # Set up evaluators
            evaluators = self.tracing_manager.setup_evaluators()
            
            if evaluators:
                try:
                    # Run evaluation
                    from langsmith import evaluate

                    evaluator_functions = list(evaluators.values())
                    logger.info(f"Using {len(evaluator_functions)} evaluator functions: {list(evaluators.keys())}")

                    experiment_results = evaluate(
                        optimized_target_fn,
                        data=evaluation_dataset.id,
                        evaluators=evaluator_functions,
                        experiment_prefix=f"optimized_eval_{evaluation_dataset.name}",
                        max_concurrency=2  # Keep it simple
                    )
                    
                    console.print(f"[bold green]Optimized evaluation experiment completed: {experiment_results}[/bold green]")
                    console.print(f"Results will be available in LangSmith dashboard.")
                    
                except Exception as e:
                    logger.error(f"Error running optimized evaluation experiment: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                console.print("[bold yellow]No evaluators available for evaluation[/bold yellow]")
        
        # Generate comprehensive HTML report
        self.generate_optimized_html_report(df, test_dir)
        
        console.print(f"\n[bold green]Optimized testing complete! Results saved to {test_dir}[/bold green]")
        console.print(f"Summary: {summary_file}")
        console.print(f"HTML Report: {test_dir / 'optimized_test_report.html'}")
        
        return df, test_dir
    
    def generate_optimized_html_report(self, df, output_dir):
        """Generate a comprehensive HTML report for optimized prompt testing."""
        # Calculate statistics by category
        category_stats = {}
        if not df.empty:
            for category in df['query_category'].unique():
                category_data = df[df['query_category'] == category]
                category_stats[category] = {
                    'total_tests': len(category_data),
                    'success_rate': len(category_data[category_data['error'].isna()]) / len(category_data) if len(category_data) > 0 else 0,
                    'avg_processing_time': category_data['processing_time'].mean() if 'processing_time' in category_data.columns else 0,
                    'avg_answer_length': category_data['answer_length'].mean() if 'answer_length' in category_data.columns else 0,
                    'avg_docs_retrieved': category_data['retrieved_docs_count'].mean() if 'retrieved_docs_count' in category_data.columns else 0
                }
        
        # Generate comprehensive HTML
        html = f"""
        <html>
        <head>
            <title>Optimized RAG System Test Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .summary-box {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .category-box {{ background-color: #f9f9f9; padding: 10px; border-left: 4px solid #007bff; margin-bottom: 15px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .excellent {{ background-color: #d4edda; }}
                .good {{ background-color: #d1ecf1; }}
                .needs-improvement {{ background-color: #f8d7da; }}
                .service {{ background-color: #e3f2fd; }}
                .financial {{ background-color: #fff3cd; }}
                .emergency {{ background-color: #f5c6cb; }}
            </style>
        </head>
        <body>
            <h1>Optimized RAG System Test Results</h1>
            <div class="summary-box">
                <h3>Test Summary</h3>
                <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Total Tests:</strong> {len(df)}</p>
                <p><strong>Local Authorities:</strong> {len(df['local_authority'].unique()) if not df.empty else 0}</p>
                <p><strong>Query Categories:</strong> {len(df['query_category'].unique()) if not df.empty else 0}</p>
                <p><strong>Overall Success Rate:</strong> {(df['error'].isna()).mean():.1%} if not df.empty else "No data"</p>
                <p><strong>Prompt Used:</strong> Optimized prompt template</p>
            </div>
            
            <h2>Performance by Query Category</h2>
        """
        
        # Add category performance table
        html += """
            <table>
                <tr>
                    <th>Category</th>
                    <th>Tests</th>
                    <th>Success Rate</th>
                    <th>Avg Time (s)</th>
                    <th>Avg Answer Length</th>
                    <th>Avg Docs Retrieved</th>
                </tr>
        """
        
        # Sort categories by success rate
        sorted_categories = sorted(category_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        
        for category, stats in sorted_categories:
            success_rate = f"{stats['success_rate']:.1%}"
            processing_time = f"{stats['avg_processing_time']:.2f}" if stats['avg_processing_time'] else "N/A"
            answer_length = f"{stats['avg_answer_length']:.0f}" if stats['avg_answer_length'] else "N/A"
            docs_retrieved = f"{stats['avg_docs_retrieved']:.1f}" if stats['avg_docs_retrieved'] else "N/A"
            
            # Determine row class based on performance
            row_class = ""
            if stats['success_rate'] >= 0.9:
                row_class = "excellent"
            elif stats['success_rate'] >= 0.7:
                row_class = "good"
            elif stats['success_rate'] < 0.5:
                row_class = "needs-improvement"
            
            # Special highlighting for important categories
            if "service" in category:
                row_class += " service"
            elif "financial" in category:
                row_class += " financial"
            elif "emergency" in category:
                row_class += " emergency"
            
            html += f"""
                <tr class="{row_class}">
                    <td><strong>{category.replace('_', ' ').title()}</strong></td>
                    <td>{stats['total_tests']}</td>
                    <td>{success_rate}</td>
                    <td>{processing_time}</td>
                    <td>{answer_length}</td>
                    <td>{docs_retrieved}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Performance by Local Authority</h2>
            <table>
                <tr>
                    <th>Local Authority</th>
                    <th>Tests</th>
                    <th>Success Rate</th>
                    <th>Avg Processing Time</th>
                </tr>
        """
        
        # Add LA performance
        if not df.empty:
            la_stats = df.groupby('local_authority').agg({
                'query_id': 'count',
                'error': lambda x: (x.isna()).mean(),
                'processing_time': 'mean'
            }).round(3)
            
            for la, stats in la_stats.iterrows():
                success_rate = f"{stats['error']:.1%}"
                processing_time = f"{stats['processing_time']:.2f}s" if pd.notna(stats['processing_time']) else "N/A"
                
                html += f"""
                    <tr>
                        <td>{la}</td>
                        <td>{stats['query_id']}</td>
                        <td>{success_rate}</td>
                        <td>{processing_time}</td>
                    </tr>
                """
        
        html += """
            </table>
            
            <h2>Detailed Test Results by Category</h2>
        """
        
        # Add detailed results by category
        for category in sorted(df['query_category'].unique()) if not df.empty else []:
            category_data = df[df['query_category'] == category]
            html += f"""
                <div class="category-box">
                    <h3>{category.replace('_', ' ').title()} ({len(category_data)} tests)</h3>
                    <table>
                        <tr>
                            <th>Query ID</th>
                            <th>Local Authority</th>
                            <th>Status</th>
                            <th>Time (s)</th>
                            <th>Docs</th>
                            <th>Answer Preview</th>
                            <th>LangSmith</th>
                        </tr>
            """
            
            for _, row in category_data.iterrows():
                status = "✅ Success" if pd.isna(row['error']) else f"❌ Error: {row['error']}"
                time_str = f"{row['processing_time']:.2f}" if pd.notna(row['processing_time']) else "N/A"
                answer_preview = (row['answer'][:100] + "...") if len(str(row['answer'])) > 100 else str(row['answer'])
                langsmith_link = f"<a href='{row['trace_url']}' target='_blank'>View Trace</a>" if pd.notna(row['trace_url']) else "N/A"
                
                row_class = "" if pd.isna(row['error']) else "needs-improvement"
                
                html += f"""
                    <tr class="{row_class}">
                        <td>{row['query_id']}</td>
                        <td>{row['local_authority']}</td>
                        <td>{status}</td>
                        <td>{time_str}</td>
                        <td>{row['retrieved_docs_count']}</td>
                        <td>{answer_preview}</td>
                        <td>{langsmith_link}</td>
                    </tr>
                """
            
            html += """
                    </table>
                </div>
            """
        
        # Add insights section
        html += """
            <h2>Key Insights for Dissertation Analysis</h2>
            <div class="summary-box">
                <h3>Performance Analysis</h3>
                <ul>
        """
        
        # Generate insights based on data
        if not df.empty:
            # Overall performance insight
            overall_success = (df['error'].isna()).mean()
            if overall_success >= 0.9:
                html += "<li><strong>Excellent Overall Performance:</strong> The optimized RAG system shows very high success rates across all categories.</li>"
            elif overall_success >= 0.7:
                html += "<li><strong>Good Overall Performance:</strong> The optimized RAG system performs well with room for minor improvements.</li>"
            else:
                html += "<li><strong>Performance Needs Improvement:</strong> Some query types are challenging for the current system.</li>"
            
            # Service vs Financial comparison (matching Phase 1 analysis)
            service_queries = df[df['query_category'] == 'service_standard']
            financial_queries = df[df['query_category'] == 'financial_standard']
            
            if len(service_queries) > 0 and len(financial_queries) > 0:
                service_success = (service_queries['error'].isna()).mean()
                financial_success = (financial_queries['error'].isna()).mean()
                
                if service_success > financial_success:
                    html += f"<li><strong>Service vs Financial:</strong> Service queries perform better ({service_success:.1%}) than financial queries ({financial_success:.1%}), similar to Phase 1 findings.</li>"
                elif financial_success > service_success:
                    html += f"<li><strong>Service vs Financial:</strong> Financial queries perform better ({financial_success:.1%}) than service queries ({service_success:.1%}), different from Phase 1 patterns.</li>"
                else:
                    html += f"<li><strong>Service vs Financial:</strong> Both service and financial queries show similar performance ({service_success:.1%}).</li>"
            
            # Emergency scenarios insight
            emergency_queries = df[df['query_category'] == 'emergency_scenarios']
            if len(emergency_queries) > 0:
                emergency_success = (emergency_queries['error'].isna()).mean()
                if emergency_success >= 0.8:
                    html += "<li><strong>Emergency Scenarios:</strong> System handles crisis situations well, showing reliability for urgent queries.</li>"
                else:
                    html += "<li><strong>Emergency Scenarios:</strong> Critical scenarios may need specialized handling to improve reliability.</li>"
            
            # Processing time insight
            avg_time = df['processing_time'].mean()
            if avg_time < 5:
                html += f"<li><strong>Response Time:</strong> Fast average response time of {avg_time:.1f} seconds, suitable for real-time applications.</li>"
            elif avg_time < 10:
                html += f"<li><strong>Response Time:</strong> Reasonable average response time of {avg_time:.1f} seconds.</li>"
            else:
                html += f"<li><strong>Response Time:</strong> Response time of {avg_time:.1f} seconds may need optimization for production use.</li>"
            
            # Document retrieval insight
            avg_docs = df['retrieved_docs_count'].mean()
            html += f"<li><strong>Document Retrieval:</strong> Average of {avg_docs:.1f} documents retrieved per query, indicating effective information gathering.</li>"
            
            # Consistency insight
            time_std = df['processing_time'].std()
            if time_std < 2:
                html += f"<li><strong>Consistency:</strong> Low variance in processing times (σ={time_std:.1f}s) indicates reliable performance.</li>"
            else:
                html += f"<li><strong>Consistency:</strong> High variance in processing times (σ={time_std:.1f}s) may indicate optimization opportunities.</li>"
        
        html += """
                </ul>
                <h3>Implications for Dissertation</h3>
                <ul>
                    <li><strong>RAG vs Direct LLM:</strong> Compare success rates and consistency with Phase 1 results</li>
                    <li><strong>Reliability:</strong> Analyze variance and error patterns for production readiness assessment</li>
                    <li><strong>Query Type Analysis:</strong> Document differential performance across service vs financial vs scenario queries</li>
                    <li><strong>Scalability:</strong> Processing times indicate feasibility for real-world deployment</li>
                    <li><strong>Next Steps:</strong> Use LangSmith traces to analyze failure modes and optimization opportunities</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save the HTML report
        report_file = output_dir / "optimized_test_report.html"
        with open(report_file, "w") as f:
            f.write(html)

def main():
    """Main entry point for optimized prompt testing."""
    parser = argparse.ArgumentParser(description="Optimized Prompt Testing for RAG System")
    parser.add_argument("--vector_db_path", default="./output", help="Base path for vector databases")
    parser.add_argument("--results_path", default="./optimized_prompt_results", help="Path to save test results")
    parser.add_argument("--las", nargs='+', required=True, help="List of Local Authorities to test")
    parser.add_argument("--top_k", type=int, default=10, help="Number of documents to retrieve per query")
    parser.add_argument("--categories", nargs='+', 
                       choices=["service_standard", "financial_standard", "standard", "adult", "family", 
                               "carer", "financial", "housing", "assessment", "emergency", "transition", 
                               "mental_health", "all_scenarios"],
                       help="Query categories to test")
    parser.add_argument("--langsmith_key", help="LangSmith API key for tracing")
    parser.add_argument("--run_evaluation", action="store_true", help="Run evaluation with faithfulness metric")
    
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
    
    # Display category information
    if args.categories:
        console.print(f"[bold]Testing categories: {', '.join(args.categories)}[/bold]")
    else:
        console.print("[bold]Using default query selection (standard + sample scenarios)[/bold]")
    
    # Initialize optimized prompt tester
    tester = OptimizedPromptTester(
        vector_db_base_path=args.vector_db_path,
        results_path=args.results_path,
        langsmith_api_key=args.langsmith_key or os.getenv("LANGSMITH_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Run the optimized tests
    console.print("\n[bold blue]Starting optimized prompt testing with expanded queries...[/bold blue]")
    df, results_dir = tester.run_optimized_tests(
        local_authorities=valid_las,
        query_categories=args.categories,
        top_k=args.top_k,
        run_evaluation=args.run_evaluation
    )
    
    # Display summary statistics
    console.print("\n[bold]Summary Statistics:[/bold]")
    console.print(f"Total tests completed: {len(df)}")
    console.print(f"Overall success rate: {(df['error'].isna()).mean():.1%}" if not df.empty else "No data")
    
    if not df.empty:
        # Performance by category
        console.print("\n[bold cyan]Performance by Category:[/bold cyan]")
        category_performance = df.groupby('query_category')['error'].apply(lambda x: x.isna().mean()).sort_values(ascending=False)
        for category, success_rate in category_performance.items():
            console.print(f"{category.replace('_', ' ').title()}: {success_rate:.1%}")
        
        # Service vs Financial comparison (matching Phase 1)
        service_queries = df[df['query_category'] == 'service_standard']
        financial_queries = df[df['query_category'] == 'financial_standard']
        
        if len(service_queries) > 0 and len(financial_queries) > 0:
            service_success = (service_queries['error'].isna()).mean()
            financial_success = (financial_queries['error'].isna()).mean()
            console.print(f"\n[bold magenta]Service vs Financial Analysis (Phase 1 Comparison):[/bold magenta]")
            console.print(f"Service Questions (Q1-Q5): {service_success:.1%}")
            console.print(f"Financial Questions (Q6-Q9): {financial_success:.1%}")
        
        # Best and worst performing categories
        if len(category_performance) > 1:
            best_category = category_performance.index[0]
            worst_category = category_performance.index[-1]
            console.print(f"\n[bold green]Best performing category: {best_category.replace('_', ' ').title()} ({category_performance[best_category]:.1%})[/bold green]")
            console.print(f"[bold yellow]Needs attention: {worst_category.replace('_', ' ').title()} ({category_performance[worst_category]:.1%})[/bold yellow]")

if __name__ == "__main__":
    main()