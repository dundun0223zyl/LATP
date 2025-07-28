import os
import time
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
#from langchain_core.tracers import tracing_enabled, traceable
from langsmith.run_helpers import traceable, get_current_run_tree
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain.callbacks.tracers import LangChainTracer

from tenacity import retry, wait_random_exponential, stop_after_attempt

from tracing import TracingManager

# Configure logger
logger = logging.getLogger("social_care_rag")

class RAGSystem:
    def __init__(self, vector_db, embedding_model="all-MiniLM-L6-v2", openai_api_key=None, langsmith_api_key=None):
        """Initialize the RAG system with vector database and embedding model."""
        self.vector_db = vector_db
        
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Initialized embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
        
        # Set up OpenAI configuration
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize tracing
        self.tracing_manager = TracingManager(api_key=langsmith_api_key)
        self.prompt_version = "v1.0"  # Track prompt versions
        self.prompt_template = """
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
        """
        
        if self.openai_api_key:
            try:
                # Create a tracer for this session
                tracer = self.tracing_manager.get_tracer_callback()
                
                # Set up the LangChain ChatOpenAI client with tracing
                self.client = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.3,
                    openai_api_key=self.openai_api_key,
                    max_tokens=1000,
                    callbacks=[tracer] if tracer else None
                )
                self.llm_available = True
                logger.info("LangChain OpenAI client configured successfully")
            except Exception as e:
                logger.error(f"Error setting up LangChain OpenAI client: {str(e)}")
                self.llm_available = False
                self.client = None
        else:
            self.llm_available = False
            self.client = None
            logger.warning("No OpenAI API key provided. Only retrieval will be available.")

    @traceable(run_type="chain", name="RAG Query Processing")
    def process_query(self, query, top_k=10, prompt_variation_name=None, local_authority=None, query_id=None):
        """Process a query through the RAG system with tracing."""
        
        start_time = time.time()

        try:
            # Get current run for adding metadata
            current_run = get_current_run_tree()
            
            # Add metadata to the current run if available
            if current_run:
                # Set a more descriptive name
                query_short = query[:30] + "..." if len(query) > 30 else query
                run_name = f"LA:{local_authority or 'Unknown'}"
                if query_id:
                    run_name += f" | {query_id}"
                run_name += f" | {prompt_variation_name or 'default'}"
                run_name += f" | {query_short}"
                
                # Update the run name
                current_run.name = run_name
                
                metadata = {
                    "query_text": query[:100] + "..." if len(query) > 100 else query,
                    "prompt_variation": prompt_variation_name or "default",
                    "local_authority": local_authority or "Unknown",
                    "query_id": query_id or "Unknown",
                    "top_k": top_k,
                    "timestamp": datetime.now().isoformat()
                }
                current_run.metadata.update(metadata)
                
                # Add tags to the current run
                tags = [
                    f"prompt_variation:{prompt_variation_name or 'default'}",
                    f"local_authority:{local_authority or 'Unknown'}",
                    f"query_id:{query_id or 'Unknown'}",
                    "type:query_processing"
                ]
                current_run.tags.extend(tags)
                
                logger.info(f"Added metadata and tags to current run: {run_name}")
            
            # 1. Generate embedding for the query
            query_embedding = self.embedding_model.encode(query).tolist()

            # 2. Retrieve relevant documents
            retrieved_docs = self.vector_db.similarity_search(
                query_embedding, 
                top_k=top_k
            )
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query")

            # Update run with document count if available
            if current_run:
                current_run.metadata["documents_retrieved"] = len(retrieved_docs)

            # 3. Format context from retrieved documents
            context = self._format_context(retrieved_docs)

            # 4. Generate answer using LLM if available
            answer = "LLM generation not available. Please check the retrieved context."

            if self.llm_available and self.client:
                # Use the current prompt template (which might be the default or overridden)
                current_template = self.prompt_template
                logger.info(f"Using prompt template for variation: {prompt_variation_name or 'default'}")
                
                # Generate answer
                answer = self._generate_answer(
                    query=query, 
                    context=context, 
                    retrieved_docs=retrieved_docs, 
                    llm=self.client, 
                    prompt_template=current_template,
                    prompt_variation_name=prompt_variation_name
                )
                logger.info(f"Generated answer of length {len(answer)}")

            end_time = time.time()
            duration = end_time - start_time

            # Record metrics
            metrics = {
                "duration_seconds": duration,
                "retrieved_docs_count": len(retrieved_docs),
                "query_length": len(query),
                "answer_length": len(answer)
            }

            # Add RAG-specific metrics
            rag_metrics = self._calculate_rag_metrics(query, answer, context, retrieved_docs)
            metrics.update(rag_metrics)

            # Prepare result
            result = {
                'query': query,
                'retrieved_documents': retrieved_docs,
                'context': context,
                'answer': answer,
                'metrics': metrics,
                'local_authority': local_authority,
                'query_id': query_id
            }

            # Add run_id if available - Convert UUID to string
            if current_run:
                result['run_id'] = str(current_run.id) if hasattr(current_run, 'id') else None
                
                if hasattr(current_run, 'id'):
                    # Generate a trace URL with string UUID
                    trace_url = f"https://eu.smith.langchain.com/o/me/runs/{str(current_run.id)}"
                    result['trace'] = {
                        'run_id': str(current_run.id),
                        'project': self.tracing_manager.project_name,
                        'url': trace_url
                    }
                
                logger.info(f"Added run_id: {str(current_run.id) if hasattr(current_run, 'id') else None} to result")
            
            return result

        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            return {
                'query': query,
                'error': str(e),
                'retrieved_documents': [],
                'context': "",
                'answer': f"Error processing your query: {str(e)}",
                'local_authority': local_authority,
                'query_id': query_id
            }

    def _format_context(self, retrieved_docs):
        """Format retrieved documents into a context string for the LLM."""
        context_parts = []

        for i, doc in enumerate(retrieved_docs):
            # Get more descriptive source information
            source_name = doc['metadata'].get('source', 'Unknown source')
            source_url = doc['metadata'].get('source_url', '')  # Get source URL
            doc_type = doc['metadata'].get('type', 'Unknown type')

            # Clean up the source name to be more descriptive
            clean_source = source_name.replace('-', ' ').replace('_', ' ')
            if '.' in clean_source:
                clean_source = clean_source.split('.')[0]  # Remove file extension

            content = doc['content']

            # Format each document with more user-friendly source information
            source_info = f"{clean_source} ({doc_type})"
            if source_url:
                source_info += f" - URL: {source_url}"

            # Add page count info for PDFs
            if doc_type == 'pdf' and 'page_count' in doc['metadata']:
                source_info += f" - {doc['metadata']['page_count']} pages"

            doc_context = f"[Document {i+1}] From {source_info}:\n{content}\n"
            context_parts.append(doc_context)

        return "\n".join(context_parts)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def _generate_answer(self, query, context, retrieved_docs, llm, prompt_template=None, prompt_variation_name=None):
        """Generate an answer using the LLM with tracing."""
        try:
            # Use provided template or default
            template = prompt_template or self.prompt_template
            
            # Log which template is being used
            logger.info(f"Using prompt template: {prompt_variation_name or 'default'}")
            
            # Log the prompt version if tracing is enabled
            if self.tracing_manager and self.tracing_manager.client:
                # Create a unique name for this prompt version
                timestamp = int(time.time())
                clean_name = prompt_variation_name.lower().replace(' ', '_') if prompt_variation_name else "default"
                version_name = f"prompt_{clean_name}_{timestamp}"
                
                logger.info(f"Logging prompt version: {version_name}")
                
                self.tracing_manager.log_prompt_version(
                    prompt_template=template,
                    version_name=version_name,
                    description=f"RAG prompt: {prompt_variation_name or 'default'}"
                )
            
            # Create message templates
            system_message = SystemMessagePromptTemplate.from_template(
                "You are a helpful assistant specialized in social care services in the UK. Answer based only on the provided context."
            )
            
            # Create a template with the actual values already substituted
            human_template = template.replace("{query}", query).replace("{context}", context)
            human_message = HumanMessagePromptTemplate.from_template(human_template)
            
            # Create the chat prompt template
            chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
            
            # Format the messages
            formatted_messages = chat_prompt.format_messages()
            
            # Get current run for tracing
            current_run = get_current_run_tree()
            if current_run:
                current_run.metadata["prompt_type"] = "chat"
                current_run.metadata["prompt_template_name"] = prompt_variation_name or "default"
            
            # Call the LLM
            response = llm.invoke(formatted_messages)
            
            # Extract the answer text
            answer = response.content
            
            logger.info("Generated answer with LLM")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"

    def _calculate_rag_metrics(self, query, answer, context, retrieved_docs):
        """Calculate RAG-specific metrics for the response."""
        metrics = {}

        try:
            # Create a set of stop words for filtering
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are'}

            # 1. Context utilization score - how much of the context is reflected in the answer
            if context and answer:
                # Convert to lowercase and tokenize by splitting on whitespace and punctuation
                context_tokens = set(re.findall(r'\b\w+\b', context.lower()))
                answer_tokens = set(re.findall(r'\b\w+\b', answer.lower()))

                # Filter out stop words
                context_tokens = context_tokens - stop_words
                answer_tokens = answer_tokens - stop_words

                # Calculate overlap
                if context_tokens:
                    utilization = len(context_tokens.intersection(answer_tokens)) / len(context_tokens)
                    metrics['context_utilization'] = utilization

            # 2. Citation detection - checks if answer references sources
            citation_patterns = ['according to', 'as stated in', 'as mentioned in', 'from the', 'the document']
            citations_found = sum(1 for pattern in citation_patterns if pattern in answer.lower())
            metrics['citations_count'] = citations_found
            metrics['has_citations'] = citations_found > 0

            # 3. Response relevance to query
            query_terms = set(re.findall(r'\b\w+\b', query.lower())) - stop_words
            if query_terms:
                query_term_presence = sum(1 for term in query_terms if term in answer.lower())
                metrics['query_relevance'] = query_term_presence / len(query_terms)

        except Exception as e:
            logger.error(f"Error calculating RAG metrics: {str(e)}")

        return metrics

    def _evaluate_relevance(self, query, response, retrieved_docs):
        """Evaluate if the response is relevant to the query."""
        # Simple relevance check - Are query terms in the response?
        query_terms = set(query.lower().split())
        response_lower = response.lower()

        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are'}
        query_terms = query_terms - common_words

        # Count matching terms
        matching_terms = sum(1 for term in query_terms if term in response_lower)
        relevance_score = matching_terms / len(query_terms) if query_terms else 0

        # Check if response references context
        context_relevance = 0
        if retrieved_docs:
            key_phrases = []
            for doc in retrieved_docs[:3]:  # Check top 3 docs
                # Extract key phrases from each doc
                doc_content = doc.get('content', '').lower()
                doc_phrases = [s.strip() for s in doc_content.split('.') if len(s.strip()) > 20]
                key_phrases.extend(doc_phrases[:2])  # Add top 2 phrases from each doc

            # Check if response contains key phrases or fragments
            phrase_matches = 0
            for phrase in key_phrases:
                words = phrase.split()
                if len(words) >= 4:  # Only meaningful phrases
                    # Check for 4-word segments in the phrase
                    for i in range(len(words) - 3):
                        segment = ' '.join(words[i:i+4])
                        if segment in response_lower:
                            phrase_matches += 1
                            break

            context_relevance = min(1.0, phrase_matches / max(1, len(key_phrases)))

        # Combined score (70% query relevance, 30% context relevance)
        final_score = (0.7 * relevance_score) + (0.3 * context_relevance)

        if final_score >= 0.8:
            comment = "Response is highly relevant to the query and uses context effectively."
        elif final_score >= 0.5:
            comment = "Response is somewhat relevant but could better address the query or use context."
        else:
            comment = "Response lacks relevance to the query or doesn't effectively use the provided context."

        return final_score, comment

    def _evaluate_completeness(self, query, response):
        """Evaluate if the response completely answers the query."""
        # Simple completeness check based on response length and structure
        min_expected_length = 100  # Minimum characters for a complete response

        # Longer responses tend to be more complete (up to a point)
        length_score = min(1.0, len(response) / min_expected_length)

        # Check for structural elements that suggest completeness
        has_explanation = any(marker in response.lower() for marker in 
                             ["because", "since", "as a result", "due to", "therefore"])
        has_examples = any(marker in response.lower() for marker in 
                          ["for example", "such as", "for instance", "e.g."])
        has_conclusion = any(marker in response.lower() for marker in 
                            ["in conclusion", "to summarize", "overall", "in summary"])

        # Calculate structure score
        structure_points = sum([has_explanation, has_examples, has_conclusion])
        structure_score = structure_points / 3.0

        # Combined score (60% length, 40% structure)
        final_score = (0.6 * length_score) + (0.4 * structure_score)

        if final_score >= 0.8:
            comment = "Response is comprehensive and well-structured."
        elif final_score >= 0.5:
            comment = "Response addresses the query but could be more detailed or better structured."
        else:
            comment = "Response is incomplete or lacks sufficient detail."

        return final_score, comment

    def _evaluate_accuracy(self, response, retrieved_docs):
        """Evaluate if the response accurately reflects the retrieved documents."""
        # Extract named entities from response (simplified)
        response_lower = response.lower()

        # Extract possible entities (naive approach)
        response_capitalized_words = [word for word in response.split() 
                                     if word[0].isupper() and len(word) > 1]

        # Create a combined context from retrieved docs
        combined_context = ""
        for doc in retrieved_docs:
            combined_context += doc.get('content', '') + " "

        context_lower = combined_context.lower()

        # Check if capitalized words from response appear in context
        entity_matches = 0
        for entity in response_capitalized_words:
            if entity.lower() in context_lower:
                entity_matches += 1

        entity_score = entity_matches / max(1, len(response_capitalized_words))

        # Check for factual statements and qualifying language
        has_qualifiers = any(term in response_lower for term in 
                           ["might", "may", "could be", "possibly", "suggests"])

        has_citations = any(term in response_lower for term in 
                          ["according to", "as stated in", "the document mentions", 
                           "as shown in", "as per"])

        accuracy_points = sum([not has_qualifiers, has_citations])
        accuracy_modifier = accuracy_points / 2.0

        # Final score (70% entity matching, 30% statement quality)
        final_score = (0.7 * entity_score) + (0.3 * accuracy_modifier)

        if final_score >= 0.8:
            comment = "Response appears to accurately reflect the information in the retrieved documents."
        elif final_score >= 0.5:
            comment = "Response is generally accurate but may contain some unsupported information."
        else:
            comment = "Response may contain significant information not supported by the retrieved documents."

        return final_score, comment