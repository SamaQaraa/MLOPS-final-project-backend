#!/usr/bin/env python3
"""
Test runner script for the gesture recognition project.
This script runs all unit tests and generates a test report.
"""

import unittest
import sys
import os
from io import StringIO
from test import test_app , test_predict

def run_all_tests():
    """Run all unit tests and return results"""
    
    # Discover and load all test modules
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load tests from test files
    suite = unittest.TestSuite()
    
    # Try to load test modules
    try:
        # Load predict tests
        predict_tests = loader.loadTestsFromName('test.test_predict')
        suite.addTests(predict_tests)
        print("✓ Loaded predict tests")
    except ImportError as e:
        print(f"⚠ Warning: Could not load predict tests: {e}")
    
    try:
        # Load app tests
        app_tests = loader.loadTestsFromName('test.test_app')
        suite.addTests(app_tests)
        print("✓ Loaded app tests")
    except ImportError as e:
        print(f"⚠ Warning: Could not load app tests: {e}")
    
    # Run the tests
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        buffer=True
    )
    
    print("\n" + "="*50)
    print("RUNNING UNIT TESTS")
    print("="*50)
    
    result = runner.run(suite)
    
    # Print results
    print(stream.getvalue())
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Return success status
    return result.wasSuccessful()

def check_dependencies():
    """Check if required dependencies are available"""
    dependencies = [
        ('fastapi', 'FastAPI'),
        ('numpy', 'NumPy'),
        ('unittest.mock', 'Mock (built-in)'),
    ]
    
    missing = []
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {display_name} is available")
        except ImportError:
            missing.append(display_name)
            print(f"✗ {display_name} is missing")
    
    return len(missing) == 0

def main():
    """Main function to run tests"""
    print("Gesture Recognition API - Unit Test Suite")
    print("="*50)
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        print("\n⚠ Some dependencies are missing. Tests may fail.")
        print("Please install required packages:\n")
        print("pip install fastapi numpy pytest")
        return False
    
    # Check if model file exists
    model_path = "model/model_svm.pkl"
    if os.path.exists(model_path):
        print(f"✓ Model file found: {model_path}")
    else:
        print(f"⚠ Model file not found: {model_path}")
        print("Some integration tests will be skipped.")
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)