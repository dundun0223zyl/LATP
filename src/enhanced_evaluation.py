# enhanced_evaluation.py
"""
Enhanced evaluation system that integrates with LangSmith's latest evaluation framework.
This follows the SOTA patterns from the official LangSmith RAG evaluation tutorial.
"""

import os
import time
import logging
import re
from pathlib import Path
from datetime import datetime
import traceback
from typing import Dict, Any, List, Optional

from langsmith import Client, wrappers
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated, TypedDict

# Set up logging
logger = logging.getLogger("enhanced_evaluation")

class EnhancedEvaluationSystem:
    def __init__(self, api_key=None, project_name="social_care_rag_enhanced", openai_api_key=None):
        """Initialize the enhanced evaluation system with latest LangSmith patterns."""
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.project_name = project_name
        self.client = None
        
        if self.api_key:
            try:
                self.client = Client(api_key=self.api_key)
                logger.info(f"LangSmith client initialized for project: {project_name}")
            except Exception as e:
                logger.error(f"Error initializing LangSmith client: {str(e)}")
                self.client = None

        # Initialize wrapped OpenAI client for tracing
        if self.openai_api_key:
            try:
                from openai import OpenAI
                openai_client = OpenAI(api_key=self.openai_api_key)
                self.wrapped_openai = wrappers.wrap_openai(openai_client)
                logger.info("OpenAI client wrapped for LangSmith tracing")
            except Exception as e:
                logger.error(f"Error wrapping OpenAI client: {str(e)}")
                self.wrapped_openai = None
        else:
            self.wrapped_openai = None

    def create_enhanced_evaluators(self):
        """Create enhanced evaluators following LangSmith's latest RAG evaluation patterns."""
        if not self.wrapped_openai:
            logger.warning("Cannot create evaluators without OpenAI client")
            return {}

        # Define Pydantic-like schemas using TypedDict for structured outputs
        class CorrectnessGrade(TypedDict):
            explanation: Annotated[str, ..., "Explain your reasoning for the score"]
            correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

        class RelevanceGrade(TypedDict):
            explanation: Annotated[str, ..., "Explain your reasoning for the score"]
            relevant: Annotated[bool, ..., "True if the answer is relevant to the question, False otherwise."]

        class GroundednessGrade(TypedDict):
            explanation: Annotated[str, ..., "Explain your reasoning for the score"]
            grounded: Annotated[bool, ..., "True if the answer is grounded in the documents, False otherwise."]

        class RetrievalRelevanceGrade(TypedDict):
            explanation: Annotated[str, ..., "Explain your reasoning for the score"]
            relevant: Annotated[bool, ..., "True if retrieved documents are relevant to the question, False otherwise."]

        class BudgetSpecificGrade(TypedDict):
            explanation: Annotated[str, ..., "Explain your reasoning for the score"]
            contains_budget_info: Annotated[bool, ..., "True if answer contains specific budget information, False otherwise."]
            budget_accuracy: Annotated[bool, ..., "True if budget figures are accurate and properly cited, False otherwise."]

        # Create LLM instances with structured output
        try:
            correctness_llm = ChatOpenAI(
                model="gpt-4o", 
                temperature=0,
                openai_api_key=self.openai_api_key
            ).with_structured_output(CorrectnessGrade, method="json_schema", strict=True)

            relevance_llm = ChatOpenAI(
                model="gpt-4o", 
                temperature=0,
                openai_api_key=self.openai_api_key
            ).with_structured_output(RelevanceGrade, method="json_schema", strict=True)

            groundedness_llm = ChatOpenAI(
                model="gpt-4o", 
                temperature=0,
                openai_api_key=self.openai_api_key
            ).with_structured_output(GroundednessGrade, method="json_schema", strict=True)

            retrieval_relevance_llm = ChatOpenAI(
                model="gpt-4o", 
                temperature=0,
                openai_api_key=self.openai_api_key
            ).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)

            budget_specific_llm = ChatOpenAI(
                model="gpt-4o", 
                temperature=0,
                openai_api_key=self.openai_api_key
            ).with_structured_output(BudgetSpecificGrade, method="json_schema", strict=True)

        except Exception as e:
            logger.error(f"Error creating LLM instances: {str(e)}")
            return {}

        # Enhanced evaluator functions following LangSmith patterns
        def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
            """Enhanced correctness evaluator for RAG responses vs reference answers."""
            if not reference_outputs or 'answer' not in reference_outputs:
                return {"key": "correctness", "score": None, "reasoning": "No reference answer provided"}

            correctness_instructions = """You are a teacher grading a quiz about UK social care services.
            You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.

            Grading criteria:
            (1) Grade based ONLY on factual accuracy relative to the ground truth answer
            (2) For budget queries, exact figures must match or be within reasonable rounding
            (3) Allow additional information if factually accurate
            (4) Check for conflicting statements
            (5) Consider terminology variations (e.g., "adult social care" vs "social services")

            Score True if the student answer is factually correct relative to ground truth.
            Score False if there are factual errors or contradictions."""

            question = inputs.get("question", inputs.get("query", ""))
            ground_truth = reference_outputs.get("answer", "")
            student_answer = outputs.get("answer", "")

            prompt = f"""QUESTION: {question}
            GROUND TRUTH ANSWER: {ground_truth}
            STUDENT ANSWER: {student_answer}"""

            try:
                grade = correctness_llm.invoke([
                    {"role": "system", "content": correctness_instructions},
                    {"role": "user", "content": prompt}
                ])
                return {
                    "key": "correctness",
                    "score": 1.0 if grade["correct"] else 0.0,
                    "reasoning": grade["explanation"]
                }
            except Exception as e:
                logger.error(f"Error in correctness evaluation: {str(e)}")
                return {"key": "correctness", "score": None, "reasoning": f"Evaluation error: {str(e)}"}

        def relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
            """Enhanced relevance evaluator for response vs query."""
            relevance_instructions = """You are evaluating if a response about UK social care services addresses the user's question.

            Evaluation criteria:
            (1) Does the response directly address the question asked?
            (2) Is the response helpful for someone needing this information?
            (3) For budget questions, does it attempt to provide financial information?
            (4) Is the response concise and relevant?

            Score True if the response meaningfully addresses the question.
            Score False if the response is off-topic or unhelpful."""

            question = inputs.get("question", inputs.get("query", ""))
            answer = outputs.get("answer", "")

            prompt = f"""QUESTION: {question}
            STUDENT ANSWER: {answer}"""

            try:
                grade = relevance_llm.invoke([
                    {"role": "system", "content": relevance_instructions},
                    {"role": "user", "content": prompt}
                ])
                return {
                    "key": "relevance",
                    "score": 1.0 if grade["relevant"] else 0.0,
                    "reasoning": grade["explanation"]
                }
            except Exception as e:
                logger.error(f"Error in relevance evaluation: {str(e)}")
                return {"key": "relevance", "score": None, "reasoning": f"Evaluation error: {str(e)}"}

        def groundedness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
            """Enhanced groundedness evaluator for response vs retrieved documents."""
            groundedness_instructions = """You are checking if a response about UK social care services is grounded in the provided documents.

            Evaluation criteria:
            (1) Is the response supported by information in the documents?
            (2) Does the response avoid hallucinating facts not in the documents?
            (3) For budget figures, are they directly mentioned in the documents?
            (4) Are citations and sources properly referenced?

            Score True if the response is well-grounded in the provided facts.
            Score False if the response contains information not supported by the documents."""

            # Extract documents from outputs
            documents = outputs.get("documents", outputs.get("retrieved_documents", []))
            
            if not documents:
                return {"key": "groundedness", "score": None, "reasoning": "No documents provided for groundedness check"}

            # Format documents
            if isinstance(documents, list) and len(documents) > 0:
                if isinstance(documents[0], dict):
                    doc_string = "\n\n".join(doc.get('content', str(doc)) for doc in documents)
                else:
                    doc_string = "\n\n".join(str(doc) for doc in documents)
            else:
                doc_string = str(documents)

            answer = outputs.get("answer", "")

            prompt = f"""FACTS: {doc_string}
            STUDENT ANSWER: {answer}"""

            try:
                grade = groundedness_llm.invoke([
                    {"role": "system", "content": groundedness_instructions},
                    {"role": "user", "content": prompt}
                ])
                return {
                    "key": "groundedness",
                    "score": 1.0 if grade["grounded"] else 0.0,
                    "reasoning": grade["explanation"]
                }
            except Exception as e:
                logger.error(f"Error in groundedness evaluation: {str(e)}")
                return {"key": "groundedness", "score": None, "reasoning": f"Evaluation error: {str(e)}"}

        def retrieval_relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
            """Enhanced retrieval relevance evaluator for retrieved docs vs query."""
            retrieval_instructions = """You are evaluating if retrieved documents are relevant to a query about UK social care services.

            Evaluation criteria:
            (1) Do the documents contain information related to the question?
            (2) For budget questions, do documents contain financial information?
            (3) Are the documents from appropriate sources (council websites, budget documents)?
            (4) Do documents contain keywords or concepts related to the query?

            Be generous - if documents have ANY relevant information, score as relevant.
            Score False only if documents are completely unrelated to the question."""

            question = inputs.get("question", inputs.get("query", ""))
            documents = outputs.get("documents", outputs.get("retrieved_documents", []))

            if not documents:
                return {"key": "retrieval_relevance", "score": None, "reasoning": "No documents provided for retrieval evaluation"}

            # Format documents
            if isinstance(documents, list) and len(documents) > 0:
                if isinstance(documents[0], dict):
                    doc_string = "\n\n".join(doc.get('content', str(doc)) for doc in documents)
                else:
                    doc_string = "\n\n".join(str(doc) for doc in documents)
            else:
                doc_string = str(documents)

            prompt = f"""FACTS: {doc_string}
            QUESTION: {question}"""

            try:
                grade = retrieval_relevance_llm.invoke([
                    {"role": "system", "content": retrieval_instructions},
                    {"role": "user", "content": prompt}
                ])
                return {
                    "key": "retrieval_relevance",
                    "score": 1.0 if grade["relevant"] else 0.0,
                    "reasoning": grade["explanation"]
                }
            except Exception as e:
                logger.error(f"Error in retrieval relevance evaluation: {str(e)}")
                return {"key": "retrieval_relevance", "score": None, "reasoning": f"Evaluation error: {str(e)}"}

        def budget_specific_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
            """Custom evaluator specifically for budget-related queries."""
            # Check if this is a budget-related query
            question = inputs.get("question", inputs.get("query", "")).lower()
            budget_keywords = ['budget', 'expenditure', 'spending', 'allocation', 'cost', 'funding', 'financial']
            
            if not any(keyword in question for keyword in budget_keywords):
                return {"key": "budget_specific", "score": None, "reasoning": "Not a budget-related query"}

            budget_instructions = """You are evaluating a response to a budget-related question about UK local authority finances.

            Evaluation criteria:
            (1) Does the response contain specific budget information (figures, amounts, percentages)?
            (2) Are budget figures properly formatted and clearly presented?
            (3) Are sources and time periods clearly specified for budget data?
            (4) Does the response distinguish between different types of budgets (total vs social care)?

            Rate both:
            - contains_budget_info: True if response has specific budget data
            - budget_accuracy: True if budget data is well-presented and sourced"""

            answer = outputs.get("answer", "")

            prompt = f"""BUDGET QUESTION: {question}
            RESPONSE: {answer}"""

            try:
                grade = budget_specific_llm.invoke([
                    {"role": "system", "content": budget_instructions},
                    {"role": "user", "content": prompt}
                ])
                
                # Calculate combined score
                has_info = grade["contains_budget_info"]
                is_accurate = grade["budget_accuracy"]
                combined_score = 0.0
                
                if has_info and is_accurate:
                    combined_score = 1.0
                elif has_info:
                    combined_score = 0.5
                
                return {
                    "key": "budget_specific",
                    "score": combined_score,
                    "reasoning": grade["explanation"],
                    "metadata": {
                        "contains_budget_info": has_info,
                        "budget_accuracy": is_accurate
                    }
                }
            except Exception as e:
                logger.error(f"Error in budget-specific evaluation: {str(e)}")
                return {"key": "budget_specific", "score": None, "reasoning": f"Evaluation error: {str(e)}"}

        # Citation evaluator (rule-based, not LLM-based for efficiency)
        def citation_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
            """Rule-based evaluator for citation quality."""
            answer = outputs.get("answer", "")

            # Check for citation patterns
            citation_patterns = [
                'according to', 'as stated in', 'as mentioned in', 
                'from the', 'the document', 'source:', 'url:', 'http'
            ]

            # Count citations
            citation_count = sum(1 for pattern in citation_patterns if pattern.lower() in answer.lower())

            # Check for specific formatting
            has_bold_formatting = '**' in answer
            has_proper_structure = any(marker in answer.lower() for marker in ['summary:', 'answer:', 'according to'])

            # Calculate score
            if citation_count >= 2 and has_bold_formatting:
                score = 1.0
                explanation = "Response includes multiple citations and proper formatting"
            elif citation_count >= 1:
                score = 0.7
                explanation = "Response includes some citations"
            elif has_proper_structure:
                score = 0.3
                explanation = "Response has good structure but lacks citations"
            else:
                score = 0.0
                explanation = "Response lacks proper citations and structure"

            return {
                "key": "citation_quality",
                "score": score,
                "reasoning": explanation,
                "metadata": {
                    "citation_count": citation_count,
                    "has_formatting": has_bold_formatting
                }
            }

        # Return all evaluators
        evaluators = {
            "correctness": correctness_evaluator,
            "relevance": relevance_evaluator,
            "groundedness": groundedness_evaluator,
            "retrieval_relevance": retrieval_relevance_evaluator,
            "budget_specific": budget_specific_evaluator,
            "citation_quality": citation_evaluator
        }

        logger.info(f"Created {len(evaluators)} enhanced evaluators")
        return evaluators

    def create_evaluation_dataset(self, local_authorities: List[str], queries: List[Dict], 
                                name_suffix: str = "", description: str = None) -> Optional[Any]:
        """Create evaluation dataset following latest LangSmith patterns."""
        if not self.client:
            logger.warning("LangSmith client not available. Cannot create dataset.")
            return None

        try:
            # Generate unique dataset name
            timestamp = int(time.time())
            dataset_name = f"social_care_rag_enhanced_{name_suffix}_{timestamp}" if name_suffix else f"social_care_rag_enhanced_{timestamp}"
            
            # Create dataset with proper description
            dataset_description = description or f"Enhanced evaluation dataset for social care RAG system with {len(local_authorities)} LAs and {len(queries)} queries"
            
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description=dataset_description
            )

            # Prepare examples
            examples = []
            for la in local_authorities:
                for query in queries:
                    formatted_query = query["text"].format(LA=la)
                    
                    example = {
                        "inputs": {
                            "question": formatted_query,
                            "query": formatted_query,  # Alternative key for compatibility
                        },
                        "outputs": {
                            "answer": query.get("expected_answer", ""),  # Reference answer if available
                        },
                        "metadata": {
                            "local_authority": la,
                            "query_id": query["id"],
                            "query_type": query.get("type", query["id"].split("_")[0]),
                            "expected_info_type": query.get("expected_info_type", "general"),
                            "is_budget_query": any(term in formatted_query.lower() 
                                                 for term in ["budget", "expenditure", "spending", "cost", "funding"]),
                            "created_at": datetime.now().isoformat()
                        }
                    }
                    examples.append(example)

            # Create examples in bulk
            if examples:
                self.client.create_examples(
                    dataset_id=dataset.id,
                    examples=examples
                )
                
            logger.info(f"Created dataset '{dataset_name}' with {len(examples)} examples")
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating evaluation dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def run_enhanced_evaluation(self, target_function, dataset, evaluators=None, experiment_prefix="enhanced_eval"):
        """Run evaluation using latest LangSmith patterns."""
        if not self.client:
            logger.warning("LangSmith client not available. Cannot run evaluation.")
            return None

        if evaluators is None:
            evaluators = self.create_enhanced_evaluators()

        try:
            # Import the latest evaluate function
            from langsmith import evaluate

            # Get dataset ID
            dataset_id = dataset.id if hasattr(dataset, 'id') else dataset

            # Run evaluation with all evaluators
            experiment_results = evaluate(
                target_function,
                data=dataset_id,
                evaluators=list(evaluators.values()),
                experiment_prefix=experiment_prefix,
                max_concurrency=3,  # Limit concurrency to avoid rate limits
                metadata={
                    "evaluation_type": "enhanced_social_care_rag",
                    "evaluator_count": len(evaluators),
                    "evaluators_used": list(evaluators.keys()),
                    "timestamp": datetime.now().isoformat()
                }
            )

            logger.info(f"Evaluation experiment completed: {experiment_prefix}")
            return experiment_results

        except Exception as e:
            logger.error(f"Error running enhanced evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_evaluation_results(self, experiment_results) -> Dict[str, Any]:
        """Analyze evaluation results and provide insights."""
        if not experiment_results:
            return {"error": "No experiment results provided"}

        try:
            # Convert to DataFrame for analysis
            df = experiment_results.to_pandas() if hasattr(experiment_results, 'to_pandas') else None
            
            if df is None:
                return {"error": "Could not convert results to DataFrame"}

            analysis = {
                "overall_stats": {
                    "total_examples": len(df),
                    "avg_scores": {},
                    "score_distribution": {}
                },
                "by_local_authority": {},
                "by_query_type": {},
                "insights": []
            }

            # Calculate average scores for each metric
            metric_columns = [col for col in df.columns if col.endswith('_score') or col in ['correctness', 'relevance', 'groundedness']]
            
            for metric in metric_columns:
                if metric in df.columns:
                    scores = df[metric].dropna()
                    if len(scores) > 0:
                        analysis["overall_stats"]["avg_scores"][metric] = {
                            "mean": float(scores.mean()),
                            "std": float(scores.std()),
                            "min": float(scores.min()),
                            "max": float(scores.max())
                        }

            # Analyze by local authority if metadata available
            if 'local_authority' in df.columns:
                for la in df['local_authority'].unique():
                    la_data = df[df['local_authority'] == la]
                    la_scores = {}
                    for metric in metric_columns:
                        if metric in la_data.columns:
                            scores = la_data[metric].dropna()
                            if len(scores) > 0:
                                la_scores[metric] = float(scores.mean())
                    analysis["by_local_authority"][la] = la_scores

            # Generate insights
            insights = []
            
            # Budget-specific insights
            if 'budget_specific_score' in df.columns:
                budget_scores = df['budget_specific_score'].dropna()
                if len(budget_scores) > 0:
                    avg_budget_score = budget_scores.mean()
                    if avg_budget_score < 0.5:
                        insights.append("Budget-specific queries show low performance - consider improving budget information retrieval")
                    elif avg_budget_score > 0.8:
                        insights.append("Budget-specific queries perform well - good budget information coverage")

            # Retrieval insights
            if 'retrieval_relevance_score' in df.columns:
                retrieval_scores = df['retrieval_relevance_score'].dropna()
                if len(retrieval_scores) > 0:
                    avg_retrieval = retrieval_scores.mean()
                    if avg_retrieval < 0.6:
                        insights.append("Document retrieval relevance is low - consider improving search strategy")

            analysis["insights"] = insights

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing evaluation results: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}


# Integration function for your existing system
def integrate_enhanced_evaluation(rag_system, tracing_manager):
    """
    Integration function to add enhanced evaluation to your existing RAG system.
    
    Usage:
        enhanced_eval = integrate_enhanced_evaluation(rag_system, tracing_manager)
        
        # Create dataset
        dataset = enhanced_eval.create_evaluation_dataset(
            local_authorities=['Manchester', 'Birmingham'],
            queries=STANDARD_QUERIES
        )
        
        # Run evaluation
        def target_fn(inputs):
            return rag_system.process_query(inputs['question'])
            
        results = enhanced_eval.run_enhanced_evaluation(target_fn, dataset)
    """
    enhanced_eval = EnhancedEvaluationSystem(
        api_key=tracing_manager.api_key if tracing_manager else None,
        project_name=tracing_manager.project_name if tracing_manager else "social_care_rag_enhanced",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    return enhanced_eval


# Example usage for your prompt testing
def enhance_prompt_testing_evaluation(prompt_tester_instance):
    """Enhance your existing prompt testing with better evaluation metrics."""
    
    # Create enhanced evaluator
    enhanced_eval = EnhancedEvaluationSystem()
    
    # Get enhanced evaluators
    evaluators = enhanced_eval.create_enhanced_evaluators()
    
    # Replace the evaluators in your tracing manager
    if hasattr(prompt_tester_instance, 'tracing_manager') and prompt_tester_instance.tracing_manager:
        prompt_tester_instance.tracing_manager.enhanced_evaluators = evaluators
        logger.info("Enhanced evaluators integrated into prompt testing system")
    
    return enhanced_eval