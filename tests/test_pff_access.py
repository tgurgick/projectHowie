#!/usr/bin/env python3
"""
Test PFF API access and available data
"""

import requests
import json
import sys

def test_pff_api(api_key: str, base_url: str = "https://api.profootballfocus.com/v1"):
    """Test PFF API access and get available endpoints"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print("🔍 Testing PFF API Access...")
    print(f"Base URL: {base_url}")
    print(f"API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
    print("-" * 50)
    
    # Test 1: Basic connectivity
    try:
        response = requests.get(f"{base_url}/", headers=headers)
        print(f"✅ Basic connectivity: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ Basic connectivity failed: {e}")
        return False
    
    # Test 2: Available endpoints
    endpoints_to_test = [
        "/seasons",
        "/stats/routes", 
        "/stats/receiving",
        "/players",
        "/teams"
    ]
    
    print("\n🔍 Testing available endpoints:")
    for endpoint in endpoints_to_test:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.get(url, headers=headers)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"   {status} {endpoint}: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list):
                        print(f"      Records: {len(data)}")
                    elif isinstance(data, dict):
                        print(f"      Keys: {list(data.keys())[:5]}...")
                except:
                    print(f"      Response: {response.text[:100]}...")
                    
        except Exception as e:
            print(f"   ❌ {endpoint}: {e}")
    
    # Test 3: Route data specifically
    print("\n🔍 Testing route data access:")
    try:
        url = f"{base_url}/stats/routes"
        params = {"season": 2024, "format": "json"}
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Route data available: {len(data)} records")
            if data:
                print(f"   Sample columns: {list(data[0].keys())[:10]}...")
        else:
            print(f"❌ Route data not available: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Route data test failed: {e}")
    
    print("\n" + "="*50)
    print("📋 Next Steps:")
    print("1. If API access works, we can integrate route data")
    print("2. If not, we can explore CSV export options")
    print("3. Or implement derived route metrics from existing data")
    
    return True

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python test_pff_access.py <your_pff_api_key>")
        print("Example: python test_pff_access.py pff_1234567890abcdef")
        sys.exit(1)
    
    api_key = sys.argv[1]
    test_pff_api(api_key)

if __name__ == "__main__":
    main()
