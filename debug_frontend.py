#!/usr/bin/env python3
"""
Debug script to test frontend JavaScript functions
"""

import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

def test_frontend_issues():
    print("🔍 Testing Frontend Issues")
    
    # Chrome options for debugging
    chrome_options = Options()
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-features=VizDisplayCompositor")
    
    try:
        # Start Chrome driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("http://localhost:8080")
        
        print("📱 Page loaded, checking for JavaScript errors...")
        
        # Check for JavaScript errors in console
        logs = driver.get_log('browser')
        if logs:
            print("⚠️  JavaScript Console Errors Found:")
            for log in logs:
                if log['level'] in ['SEVERE', 'ERROR']:
                    print(f"  {log['level']}: {log['message']}")
        else:
            print("✅ No JavaScript errors found initially")
        
        # Test if services are available by checking elements
        try:
            # Wait for main app to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[ng-app='DocumentProcessorApp']"))
            )
            print("✅ AngularJS app loaded successfully")
        except:
            print("❌ AngularJS app failed to load")
            
        # Take a screenshot for debugging
        driver.save_screenshot("frontend_debug.png")
        print("📷 Screenshot saved as frontend_debug.png")
        
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    finally:
        if 'driver' in locals():
            driver.quit()

if __name__ == "__main__":
    # Check if frontend is accessible first
    try:
        response = requests.get("http://localhost:8080", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            test_frontend_issues()
        else:
            print(f"❌ Frontend returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot reach frontend: {e}")