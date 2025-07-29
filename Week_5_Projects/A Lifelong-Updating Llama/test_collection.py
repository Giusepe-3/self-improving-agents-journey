"""
Test script for SEAL-drip data collection
Run this to verify everything is set up correctly before doing full collection.
"""

import os
import sys
import json
from datetime import datetime

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import collect

def test_config():
    """Test that configuration is loaded correctly"""
    print("🔧 Testing configuration...")
    
    assert hasattr(config, 'MODEL_NAME'), "MODEL_NAME not found in config"
    assert hasattr(config, 'DATA_DIR'), "DATA_DIR not found in config"
    assert hasattr(config, 'MAX_DAILY_ITEMS'), "MAX_DAILY_ITEMS not found in config"
    
    print(f"  ✅ Model: {config.MODEL_NAME}")
    print(f"  ✅ Data directory: {config.DATA_DIR}")
    print(f"  ✅ Max daily items: {config.MAX_DAILY_ITEMS}")
    
    # Test directory creation
    config.ensure_directories()
    for dir_path in config.REQUIRED_DIRS:
        assert os.path.exists(dir_path), f"Directory {dir_path} was not created"
    
    print("  ✅ All directories created successfully")

def test_single_collector(collector_name: str, collector_func, max_items: int = 3):
    """Test a single data collector with limited items"""
    print(f"\n📡 Testing {collector_name} collector (max {max_items} items)...")
    
    try:
        # Temporarily reduce limits for testing
        original_hackernews_max = config.HACKERNEWS_MAX_ITEMS
        original_arxiv_max = config.ARXIV_MAX_ITEMS
        
        config.HACKERNEWS_MAX_ITEMS = max_items
        config.ARXIV_MAX_ITEMS = max_items
        
        data = collector_func()
        
        # Restore original limits
        config.HACKERNEWS_MAX_ITEMS = original_hackernews_max
        config.ARXIV_MAX_ITEMS = original_arxiv_max
        
        if data:
            print(f"  ✅ Collected {len(data)} items")
            print(f"  📋 Sample item keys: {list(data[0].keys()) if data else 'None'}")
            
            # Test saving
            filepath = collect.save_data(data, f"test_{collector_name}")
            assert os.path.exists(filepath), f"File {filepath} was not created"
            print(f"  ✅ Successfully saved to {filepath}")
            
            return True
        else:
            print(f"  ⚠️  No data collected (this might be normal)")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_all_collectors():
    """Test all data collectors with reduced limits"""
    print("\n🧪 Testing all data collectors...")
    
    collectors = [
        ('hackernews', collect.collect_hackernews),
        ('arxiv', collect.collect_arxiv),
        ('wikipedia', collect.collect_wikipedia)
    ]
    
    results = {}
    for name, func in collectors:
        success = test_single_collector(name, func, max_items=3)
        results[name] = success
    
    return results

def test_data_format():
    """Test that saved data has the expected format"""
    print("\n📊 Testing data format...")
    
    # Look for any test files we created
    test_files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.startswith('test_')]
    
    if not test_files:
        print("  ⚠️  No test files found to check format")
        return False
    
    for filename in test_files[:1]:  # Just check the first one
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        print(f"  🔍 Checking format of {filename}")
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Check if first line is valid JSON
                    first_item = json.loads(lines[0])
                    print(f"    ✅ Valid JSON format")
                    print(f"    ✅ Item has {len(first_item)} fields")
                    
                    # Check for required timestamp
                    if 'collected_at' in first_item:
                        print(f"    ✅ Timestamp field present")
                    else:
                        print(f"    ⚠️  No timestamp field found")
                else:
                    print(f"    ⚠️  File is empty")
                    
        except json.JSONDecodeError as e:
            print(f"    ❌ Invalid JSON format: {e}")
            return False
        except Exception as e:
            print(f"    ❌ Error reading file: {e}")
            return False
    
    return True

def cleanup_test_files():
    """Remove test files created during testing"""
    print("\n🧹 Cleaning up test files...")
    
    if not os.path.exists(config.RAW_DATA_DIR):
        return
    
    test_files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.startswith('test_')]
    
    for filename in test_files:
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        try:
            os.remove(filepath)
            print(f"  🗑️  Removed {filename}")
        except Exception as e:
            print(f"  ❌ Could not remove {filename}: {e}")

def main():
    """Run all tests"""
    print("🚀 SEAL-drip Data Collection Test Suite")
    print("=" * 50)
    
    try:
        # Test configuration
        test_config()
        
        # Test collectors
        results = test_all_collectors()
        
        # Test data format
        test_data_format()
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 Test Results Summary:")
        
        working_collectors = sum(1 for success in results.values() if success)
        total_collectors = len(results)
        
        print(f"  Working collectors: {working_collectors}/{total_collectors}")
        
        for name, success in results.items():
            status = "✅" if success else "❌"
            print(f"    {status} {name}")
        
        if working_collectors == total_collectors:
            print("\n🎉 All tests passed! Your collection setup is ready.")
        elif working_collectors > 0:
            print(f"\n⚠️  {working_collectors} out of {total_collectors} collectors working.")
            print("   This is often normal - some APIs may be temporarily unavailable.")
        else:
            print("\n❌ No collectors working. Check your internet connection and try again.")
        
        # Cleanup
        cleanup_test_files()
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        cleanup_test_files()
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        cleanup_test_files()

if __name__ == "__main__":
    main() 