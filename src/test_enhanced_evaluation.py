# test_enhanced_evaluation.py
"""
Simple test script to verify the enhanced evaluation system is working.
Use this to test before running the full prompt testing.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_evaluation_import():
    """Test if enhanced evaluation can be imported."""
    try:
        from enhanced_evaluation import EnhancedEvaluationSystem
        logger.info("✅ Enhanced evaluation system imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Cannot import enhanced evaluation: {e}")
        return False

def test_enhanced_tracing_import():
    """Test if enhanced tracing can be imported."""
    try:
        from enhanced_tracing import TracingManager
        logger.info("✅ Enhanced tracing system imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Cannot import enhanced tracing: {e}")
        return False

def test_api_keys():
    """Test if required API keys are available."""
    openai_key = os.getenv("OPENAI_API_KEY")
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    
    if openai_key:
        logger.info("✅ OpenAI API key found")
    else:
        logger.warning("⚠️ OpenAI API key not found")
    
    if langsmith_key:
        logger.info("✅ LangSmith API key found")
    else:
        logger.warning("⚠️ LangSmith API key not found")
    
    return bool(openai_key and langsmith_key)

def test_enhanced_evaluation_creation():
    """Test creating an enhanced evaluation system instance."""
    try:
        from enhanced_evaluation import EnhancedEvaluationSystem
        
        eval_system = EnhancedEvaluationSystem(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        if eval_system.client:
            logger.info("✅ Enhanced evaluation system created with LangSmith client")
        else:
            logger.warning("⚠️ Enhanced evaluation system created but no LangSmith client")
        
        if eval_system.wrapped_openai:
            logger.info("✅ OpenAI client wrapped for tracing")
        else:
            logger.warning("⚠️ OpenAI client not wrapped")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error creating enhanced evaluation system: {e}")
        return False

def test_evaluators_creation():
    """Test creating evaluators."""
    try:
        from enhanced_evaluation import EnhancedEvaluationSystem
        
        eval_system = EnhancedEvaluationSystem(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        evaluators = eval_system.create_enhanced_evaluators()
        
        if evaluators:
            logger.info(f"✅ Created {len(evaluators)} evaluators: {', '.join(evaluators.keys())}")
            return True
        else:
            logger.error("❌ No evaluators created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error creating evaluators: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_evaluation():
    """Test running a simple evaluation."""
    try:
        from enhanced_evaluation import EnhancedEvaluationSystem
        
        eval_system = EnhancedEvaluationSystem(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        evaluators = eval_system.create_enhanced_evaluators()
        
        if not evaluators:
            logger.error("❌ No evaluators available for testing")
            return False
        
        # Test data
        test_inputs = {"question": "What is the budget for social care in Manchester?"}
        test_outputs = {"answer": "The social care budget for Manchester is £50 million according to the council documents."}
        test_reference = {"answer": "Manchester's social care budget is £50 million for 2024."}
        
        # Test each evaluator
        results = {}
        for name, evaluator in evaluators.items():
            try:
                if name == "correctness":
                    result = evaluator(test_inputs, test_outputs, test_reference)
                else:
                    result = evaluator(test_inputs, test_outputs)
                
                results[name] = result
                logger.info(f"✅ {name}: Score = {result.get('score', 'N/A')}")
                
            except Exception as e:
                logger.error(f"❌ Error testing {name}: {e}")
                results[name] = {"error": str(e)}
        
        return len(results) > 0
        
    except Exception as e:
        logger.error(f"❌ Error in simple evaluation test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_database_exists(las=["manchester"]):
    """Test if vector databases exist for testing."""
    vector_db_base = Path("./output")
    
    existing_dbs = []
    for la in las:
        db_path = vector_db_base / f"{la.lower()}_db"
        if db_path.exists():
            existing_dbs.append(la)
            logger.info(f"✅ Vector DB found for {la}: {db_path}")
        else:
            logger.warning(f"⚠️ Vector DB not found for {la}: {db_path}")
    
    return existing_dbs

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING ENHANCED EVALUATION SYSTEM")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Import tests
    total_tests += 1
    if test_enhanced_evaluation_import():
        tests_passed += 1
    
    total_tests += 1
    if test_enhanced_tracing_import():
        tests_passed += 1
    
    # Test 2: API keys
    total_tests += 1
    if test_api_keys():
        tests_passed += 1
    
    # Test 3: System creation
    total_tests += 1
    if test_enhanced_evaluation_creation():
        tests_passed += 1
    
    # Test 4: Evaluators creation
    total_tests += 1
    if test_evaluators_creation():
        tests_passed += 1
    
    # Test 5: Simple evaluation
    total_tests += 1
    if test_simple_evaluation():
        tests_passed += 1
    
    # Test 6: Vector databases
    total_tests += 1
    existing_dbs = test_vector_database_exists()
    if existing_dbs:
        tests_passed += 1
        print(f"✅ Found vector databases for: {', '.join(existing_dbs)}")
    
    print("=" * 60)
    print(f"TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Enhanced evaluation system is ready to use.")
        print("\nNext steps:")
        print("1. Use enhanced_prompt_test.py for full prompt testing")
        print("2. Or modify your existing main.py to use enhanced_tracing.py")
        print("\nExample command:")
        if existing_dbs:
            print(f"python enhanced_prompt_test.py --las {' '.join(existing_dbs)} --run_evaluation --langsmith_key YOUR_KEY")
    elif tests_passed >= 4:
        print("⚠️ Most tests passed. System should work but may have limited functionality.")
        print("Check the warnings above for missing features.")
    else:
        print("❌ Many tests failed. Please check your setup:")
        print("1. Ensure all .py files are in the same directory")
        print("2. Check API keys in .env file")
        print("3. Install required packages: pip install langsmith langchain-openai")

if __name__ == "__main__":
    main()