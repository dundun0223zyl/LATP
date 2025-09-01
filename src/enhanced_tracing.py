# enhanced_tracing.py - Complete updated version of your tracing.py with SOTA integration

import os
import time
import logging
import re
from pathlib import Path
from datetime import datetime
import traceback
from typing import Dict, Any, List, Optional

# Set up logging
logger = logging.getLogger("social_care_rag")

class TracingManager:
    def __init__(self, api_key=None, project_name="social_care_rag"):
        """Initialize the tracing manager with enhanced evaluation capabilities."""
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        self.project_name = project_name
        self.client = None
        
        # Initialize enhanced evaluation system
        self.enhanced_evaluation = None
        
        if self.api_key:
            try:
                from langsmith import Client
                self.client = Client(api_key=self.api_key)
                logger.info(f"LangSmith client initialized for project: {project_name}")
                
                # Initialize enhanced evaluation
                try:
                    from enhanced_evaluation import EnhancedEvaluationSystem
                    self.enhanced_evaluation = EnhancedEvaluationSystem(
                        api_key=self.api_key,
                        project_name=project_name,
                        openai_api_key=os.getenv("OPENAI_API_KEY")
                    )
                    logger.info("Enhanced evaluation system initialized")
                except ImportError:
                    logger.warning("Enhanced evaluation system not available - using fallback evaluators")
                    
            except Exception as e:
                logger.error(f"Error initializing LangSmith client: {str(e)}")
                self.client = None

    def get_tracer_callback(self):
        """Get a tracer callback for LangChain."""
        if not self.client:
            return None
            
        try:
            from langchain.callbacks.tracers import LangChainTracer
            return LangChainTracer(project_name=self.project_name)
        except Exception as e:
            logger.error(f"Error creating tracer callback: {str(e)}")
            return None

    def log_prompt_version(self, prompt_template, version_name, description=None):
        """Log a prompt version using the LangSmith SDK."""
        if not self.client:
            return None

        try:
            # Generate a timestamp to ensure uniqueness
            timestamp = int(time.time())
            
            # Format the version_name to be API-compliant
            api_version_name = f"{version_name.lower().replace('.', '_').replace(' ', '_')}_{timestamp}"
            
            # Ensure it starts with a-z (if it starts with a number, prepend 'v')
            if not api_version_name[0].isalpha():
                api_version_name = 'v' + api_version_name
                
            logger.info(f"Creating prompt version with name: {api_version_name}")
            
            # Create a LangChain prompt template
            from langchain.prompts import PromptTemplate
            from langchain_core.prompts import ChatPromptTemplate
            
            # Try to create a formatted prompt template
            try:
                # First try as a chat prompt template
                chat_prompt = ChatPromptTemplate.from_template(prompt_template)
                
                # Push the prompt to LangSmith
                url = self.client.push_prompt(
                    api_version_name,  # Use the unique version name
                    object=chat_prompt,
                    description=description or f"Version: {version_name}"
                )
                
                logger.info(f"Logged prompt version '{version_name}' to LangSmith: {url}")
                return url
            except Exception as e1:
                # Fall back to regular prompt template
                logger.warning(f"Chat prompt creation failed: {str(e1)}")
                try:
                    regular_prompt = PromptTemplate.from_template(prompt_template)
                    
                    # Push the prompt to LangSmith
                    url = self.client.push_prompt(
                        api_version_name,  # Use the unique version name
                        object=regular_prompt,
                        description=description or f"Version: {version_name} (created {timestamp})"
                    )
                    
                    logger.info(f"Logged prompt version '{version_name}' to LangSmith as '{api_version_name}': {url}")
                    return url
                except Exception as e2:
                    logger.warning(f"Regular prompt creation failed: {str(e2)}")
                    raise e2
                    
        except Exception as e:
            logger.error(f"Error logging prompt version: {str(e)}")
            # If all else fails, save locally
            try:
                # Store the prompt template in a central location
                prompt_dir = Path("./prompts")
                prompt_dir.mkdir(exist_ok=True)
                
                with open(prompt_dir / f"{version_name}_{timestamp}.txt", "w") as f:
                    f.write(f"# {version_name}\n")
                    f.write(f"# {description or ''}\n\n")
                    f.write(prompt_template)
                
                logger.info(f"Saved prompt version '{version_name}' locally")
                return f"local:{version_name}_{timestamp}"
            except Exception as e2:
                logger.error(f"Error saving prompt locally: {str(e2)}")
                return None

    def add_run_tags(self, run_id, tags):
        """Add tags to a run (like 'success', 'failure', etc.)."""
        if not self.client:
            return False

        try:
            self.client.update_run(run_id, tags=tags)
            logger.info(f"Added tags {tags} to run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding tags to run {run_id}: {str(e)}")
            return False

    def get_run_url(self, run_id):
        """Get the URL for viewing a run in the LangSmith UI."""
        if not run_id:
            return None
        return f"https://eu.smith.langchain.com/o/me/runs/{run_id}"
    
    def create_evaluation_dataset(self, local_authorities, queries, prompt_variations_param=None, name_suffix=""):
        """Create an evaluation dataset using enhanced evaluation system."""
        if not self.client:
            logger.warning("LangSmith client not available. Cannot create dataset.")
            return None
        
        # Use enhanced evaluation system if available
        if self.enhanced_evaluation:
            try:
                dataset = self.enhanced_evaluation.create_evaluation_dataset(
                    local_authorities=local_authorities,
                    queries=queries,
                    name_suffix=name_suffix,
                    description=f"Enhanced RAG evaluation dataset with {len(local_authorities)} LAs and {len(queries)} queries"
                )
                return dataset
            except Exception as e:
                logger.error(f"Error using enhanced evaluation dataset creation: {str(e)}")
                # Fall back to original method
        
        # Original dataset creation method as fallback
        try:
            # Create a dataset name with timestamp
            timestamp = int(time.time())
            dataset_name = f"social_care_rag_test_{name_suffix}_{timestamp}"
            
            # Create dataset
            dataset = self.client.create_dataset(dataset_name=dataset_name)

            # Use default PROMPT_VARIATIONS if none provided
            if prompt_variations_param is None:
                # Import at the function level to avoid circular imports
                try:
                    from prompt_test import PROMPT_VARIATIONS
                    prompt_variations = list(PROMPT_VARIATIONS.keys())
                except ImportError:
                    prompt_variations = ["default"]
            else:
                prompt_variations = prompt_variations_param
            
           # Add examples - one for each LA + query + prompt variation combination
            examples = []
            for la in local_authorities:
                for query in queries:
                    formatted_query = query["text"].format(LA=la)
                    for prompt_name in prompt_variations:
                        # Create a placeholder answer for reference output
                        placeholder_answer = f"This is a placeholder answer for {la} regarding {query['id']}."
                        examples.append({
                            "inputs": {
                                "query": formatted_query,
                                "question": formatted_query,  # Add both formats for compatibility
                                "metadata": {
                                    "local_authority": la,
                                    "query_id": query["id"],
                                    "prompt_variation": prompt_name,
                                    "query_type": query["id"].split("_")[0]
                                }
                            },
                            # Add empty outputs to ensure dataset structure is valid
                            "outputs": {
                                "answer": placeholder_answer
                            }
                        })
            
            # Add examples to dataset
            if examples:
                self.client.create_examples(
                    dataset_id=dataset.id,
                    examples=examples
                )
                
            logger.info(f"Created dataset with {len(examples)} examples")
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating evaluation dataset: {str(e)}")
            return None
    
    def setup_evaluators(self):
        """Set up evaluators using enhanced evaluation system if available."""
        # Try to use enhanced evaluation system first
        if self.enhanced_evaluation:
            try:
                enhanced_evaluators = self.enhanced_evaluation.create_enhanced_evaluators()
                if enhanced_evaluators:
                    logger.info(f"Using enhanced evaluators: {', '.join(enhanced_evaluators.keys())}")
                    return enhanced_evaluators
            except Exception as e:
                logger.error(f"Error setting up enhanced evaluators: {str(e)}")
                logger.info("Falling back to basic evaluators")
        
        # Fallback to basic evaluators
        if not self.client:
            logger.warning("Cannot set up evaluators without LangSmith client")
            return {}
    
        try:
            # Import required modules
            from langsmith import wrappers
            from openai import OpenAI
            from pydantic import BaseModel, Field

            # Set up OpenAI client with wrapper for LangSmith tracing
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error("OpenAI API key not found. Cannot create evaluators.")
                return {}

            openai_client = OpenAI(api_key=openai_api_key)
            wrapped_openai = wrappers.wrap_openai(openai_client)

            # Define Pydantic models for structured outputs
            class RelevanceResponse(BaseModel):
                score: float = Field(description="Score from 0.0 to 1.0")
                explanation: str = Field(description="Explanation for the score")

            class CompletenessResponse(BaseModel):
                score: float = Field(description="Score from 0.0 to 1.0")
                explanation: str = Field(description="Explanation for the score")

            class AccuracyResponse(BaseModel):
                score: float = Field(description="Score from 0.0 to 1.0")
                explanation: str = Field(description="Explanation for the score")

            # Define the relevance evaluator
            def relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
                """Evaluate the relevance of the response to the query."""
                instructions = """
                You are evaluating the relevance of a response to a query about UK social care services.

                Evaluate how well the response addresses the specific query. 
                Consider whether the response directly answers the question asked and provides the information requested.

                Score from 0.0 to 1.0, where:
                - 0.0: Response is completely irrelevant to the query
                - 0.3: Response is tangentially related but doesn't address the query
                - 0.5: Response partially addresses the query but misses key information
                - 0.7: Response addresses the query but could be more specific
                - 1.0: Response directly and fully addresses the query
                """

                question = inputs.get("question", inputs.get("query", ""))
                answer = outputs.get("answer", "")

                msg = f"Query: {question}\n\nResponse: {answer}"

                response = wrapped_openai.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": msg}
                    ],
                    response_format=RelevanceResponse,
                    temperature=0.1
                )

                # Extract structured response
                parsed_response = response.choices[0].message.parsed

                return {
                    "key": "relevance",
                    "score": parsed_response.score,
                    "reasoning": parsed_response.explanation
                }

            # Define the completeness evaluator
            def completeness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
                """Evaluate the completeness of the response."""
                instructions = """
                You are evaluating the completeness of a response to a query about UK social care services.

                Evaluate how complete the response is. 
                Consider whether the response provides all the information that would be necessary to fully answer the query.

                Score from 0.0 to 1.0, where:
                - 0.0: Response is missing all key information
                - 0.3: Response has major gaps in information
                - 0.5: Response provides partial information but lacks important details
                - 0.7: Response is mostly complete with minor omissions
                - 1.0: Response is comprehensive and covers all necessary information
                """

                question = inputs.get("question", inputs.get("query", ""))
                answer = outputs.get("answer", "")

                msg = f"Query: {question}\n\nResponse: {answer}"

                response = wrapped_openai.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": msg}
                    ],
                    response_format=CompletenessResponse,
                    temperature=0.1
                )

                # Extract structured response
                parsed_response = response.choices[0].message.parsed

                return {
                    "key": "completeness",
                    "score": parsed_response.score,
                    "reasoning": parsed_response.explanation
                }

            # Define the accuracy evaluator
            def accuracy_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
                """Evaluate the factual accuracy of the response."""
                instructions = """
                You are evaluating the accuracy of a response to a query about UK social care services.

                Evaluate the factual accuracy of the information provided in the response.
                Consider whether the information appears correct and properly sourced.

                Score from 0.0 to 1.0, where:
                - 0.0: Response contains completely incorrect information
                - 0.3: Response contains mostly incorrect information
                - 0.5: Response contains a mix of correct and incorrect information
                - 0.7: Response contains mostly correct information with minor errors
                - 1.0: Response contains completely accurate information
                """

                question = inputs.get("question", inputs.get("query", ""))
                answer = outputs.get("answer", "")

                msg = f"Query: {question}\n\nResponse: {answer}"

                response = wrapped_openai.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": msg}
                    ],
                    response_format=AccuracyResponse,
                    temperature=0.1
                )

                # Extract structured response
                parsed_response = response.choices[0].message.parsed

                return {
                    "key": "accuracy",
                    "score": parsed_response.score,
                    "reasoning": parsed_response.explanation
                }

            # Define simple citation evaluator (rule-based, not LLM-based)
            def citation_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
                """Evaluate the citation quality of the response."""
                answer = outputs.get("answer", "")

                # Check for citation patterns
                citation_patterns = [
                    'according to', 'as stated in', 'as mentioned in', 
                    'from the', 'the document', 'document', 'source'
                ]

                # Count citations
                citation_count = sum(1 for pattern in citation_patterns if pattern.lower() in answer.lower())

                # Score based on citation count
                if citation_count >= 3:
                    score = 1.0
                    explanation = "Response includes multiple citations"
                elif citation_count >= 1:
                    score = 0.7
                    explanation = "Response includes some citations"
                else:
                    score = 0.0
                    explanation = "Response lacks citations"

                return {
                    "key": "citation_score",
                    "score": score,
                    "reasoning": explanation,
                    "metadata": {
                        "citation_count": citation_count
                    }
                }

            # Create evaluators dictionary
            evaluators = {
                "relevance": relevance_evaluator,
                "completeness": completeness_evaluator,
                "accuracy": accuracy_evaluator,
                "citations": citation_evaluator
            }

            logger.info(f"Created basic evaluators: {', '.join(evaluators.keys())}")
            return evaluators

        except Exception as e:
            logger.error(f"Error setting up basic evaluators: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_evaluation_experiment(self, target_function, dataset_name, evaluators=None):
        """Run an evaluation experiment using enhanced evaluation if available."""
        if not self.client:
            logger.warning("LangSmith client not available. Cannot run evaluation.")
            return None

        # Use enhanced evaluation system if available
        if self.enhanced_evaluation:
            try:
                if evaluators is None:
                    evaluators = self.enhanced_evaluation.create_enhanced_evaluators()
                
                # Get dataset
                if isinstance(dataset_name, str):
                    datasets = self.client.list_datasets(dataset_name_contains=dataset_name)
                    dataset_ids = [d.id for d in datasets if dataset_name in d.name]
                    if not dataset_ids:
                        logger.error(f"No dataset found with name containing '{dataset_name}'")
                        return None
                    dataset_id = dataset_ids[0]
                else:
                    dataset_id = dataset_name.id if hasattr(dataset_name, 'id') else dataset_name

                # Run enhanced evaluation
                experiment_results = self.enhanced_evaluation.run_enhanced_evaluation(
                    target_function=target_function,
                    dataset=dataset_id,
                    evaluators=evaluators,
                    experiment_prefix="enhanced_social_care_rag"
                )
                
                if experiment_results:
                    logger.info("Enhanced evaluation completed successfully")
                    
                    # Analyze results
                    analysis = self.enhanced_evaluation.analyze_evaluation_results(experiment_results)
                    logger.info(f"Evaluation analysis: {analysis.get('overall_stats', {})}")
                
                return experiment_results
                
            except Exception as e:
                logger.error(f"Error with enhanced evaluation, falling back to basic: {str(e)}")

        # Fallback to basic evaluation
        try:
            # Use pre-built evaluators if none specified
            if not evaluators:
                evaluators = self.setup_evaluators()
                if not evaluators:
                    logger.error("No evaluators available for evaluation")
                    return None

            # Get dataset by name or ID
            if isinstance(dataset_name, str):
                datasets = self.client.list_datasets(dataset_name_contains=dataset_name)
                dataset_ids = [d.id for d in datasets if dataset_name in d.name]

                if not dataset_ids:
                    logger.error(f"No dataset found with name containing '{dataset_name}'")
                    return None

                dataset_id = dataset_ids[0]
            else:
                # Assume it's a dataset object with an id attribute
                dataset_id = dataset_name.id

            # Generate a unique experiment name
            timestamp = int(time.time())
            experiment_name = f"eval_{dataset_name}_{timestamp}"

            # Import the evaluate function
            from langsmith import evaluate

            # Run evaluation using the newer evaluate function
            experiment_results = evaluate(
                target_function,
                data=dataset_id,
                evaluators=list(evaluators.values()),
                experiment_prefix=experiment_name,
                max_concurrency=4
            )

            logger.info(f"Basic evaluation experiment completed: {experiment_name}")
            return experiment_results

        except Exception as e:
            logger.error(f"Error running basic evaluation experiment: {str(e)}")
            import traceback
            traceback.print_exc()
            return None