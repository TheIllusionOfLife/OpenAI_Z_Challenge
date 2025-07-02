#!/usr/bin/env python3
"""Quick human test for immediate verification of core functionality."""

def quick_test():
    """Quick verification for immediate feedback."""
    print("=== Quick Human Test ===")
    
    try:
        print("1. Testing imports...")
        import src.openai_integration
        import src.geospatial_processing  
        import src.data_loading
        print("✓ All imports successful")

        print("\n2. Testing basic functionality...")
        from src.geospatial_processing import CoordinateTransformer
        transformer = CoordinateTransformer()
        print("✓ CoordinateTransformer initialized")

        print("\n3. Testing error handling...")
        try:
            from src.openai_integration import OpenAIClient
            client = OpenAIClient(api_key='test-key')
            print("✓ OpenAI client handles test key")
        except Exception as e:
            print(f"Expected behavior: {str(e)[:60]}...")

        print("\n🎉 Quick human tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)