"""
Security Check Script for AQI Prediction System
Verifies that sensitive data is properly secured
"""

import os
import re
from pathlib import Path

def check_security():
    print("=" * 60)
    print("🔒 SECURITY CHECK FOR AQI PREDICTION SYSTEM")
    print("=" * 60)
    
    issues_found = []
    warnings = []
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✅ .env file found")
        
        # Check .env file size
        env_size = os.path.getsize('.env')
        if env_size > 1000:  # More than 1KB might indicate large secrets
            warnings.append(f".env file is large ({env_size} bytes) - check for unnecessary data")
        else:
            print("✅ .env file size is appropriate")
    else:
        issues_found.append("❌ .env file not found")
    
    # Check if .gitignore exists
    if os.path.exists('.gitignore'):
        print("✅ .gitignore file found")
        
        # Check if .env is in .gitignore
        with open('.gitignore', 'r') as f:
            gitignore_content = f.read()
            if '.env' in gitignore_content:
                print("✅ .env is properly ignored in .gitignore")
            else:
                issues_found.append("❌ .env is not in .gitignore")
    else:
        issues_found.append("❌ .gitignore file not found")
    
    # Check config.py for hardcoded secrets
    if os.path.exists('config.py'):
        with open('config.py', 'r') as f:
            config_content = f.read()
            
        # Check for hardcoded API keys
        hardcoded_patterns = [
            r'API_KEY\s*=\s*["\'][^"\']+["\']',
            r'USERNAME\s*=\s*["\'][^"\']+["\']',
            r'PASSWORD\s*=\s*["\'][^"\']+["\']',
            r'SECRET\s*=\s*["\'][^"\']+["\']',
            r'TOKEN\s*=\s*["\'][^"\']+["\']'
        ]
        
        for pattern in hardcoded_patterns:
            matches = re.findall(pattern, config_content)
            if matches:
                for match in matches:
                    if 'YOUR_' not in match and 'PLACEHOLDER' not in match:
                        issues_found.append(f"❌ Hardcoded secret found: {match}")
        
        # Check if environment variables are used
        if 'os.getenv(' in config_content:
            print("✅ Environment variables are properly used in config.py")
        else:
            issues_found.append("❌ Environment variables not used in config.py")
    else:
        issues_found.append("❌ config.py not found")
    
    # Check for other sensitive files
    sensitive_files = [
        '.env.local', '.env.production', '.env.staging',
        'secrets.json', 'credentials.json', 'config_secrets.py',
        'id_rsa', 'id_ed25519', '*.pem', '*.key'
    ]
    
    for pattern in sensitive_files:
        if '*' in pattern:
            # Handle wildcard patterns
            for file in Path('.').glob(pattern):
                if file.is_file():
                    warnings.append(f"⚠️  Sensitive file found: {file}")
        else:
            if os.path.exists(pattern):
                warnings.append(f"⚠️  Sensitive file found: {pattern}")
    
    # Check for large data files that might contain sensitive info
    data_dir = Path('data')
    if data_dir.exists():
        for file in data_dir.iterdir():
            if file.is_file() and file.stat().st_size > 10 * 1024 * 1024:  # 10MB
                warnings.append(f"⚠️  Large data file: {file} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Check for model files that might contain sensitive data
    models_dir = Path('models')
    if models_dir.exists():
        for file in models_dir.iterdir():
            if file.is_file() and file.stat().st_size > 100 * 1024 * 1024:  # 100MB
                warnings.append(f"⚠️  Large model file: {file} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Summary
    print("\n" + "=" * 60)
    print("🔍 SECURITY CHECK RESULTS")
    print("=" * 60)
    
    if not issues_found and not warnings:
        print("🎉 PERFECT! No security issues found!")
        print("✅ All sensitive data is properly secured")
    else:
        if issues_found:
            print("\n❌ CRITICAL ISSUES FOUND:")
            for issue in issues_found:
                print(f"  {issue}")
        
        if warnings:
            print("\n⚠️  WARNINGS:")
            for warning in warnings:
                print(f"  {warning}")
    
    # Recommendations
    print("\n💡 SECURITY RECOMMENDATIONS:")
    print("-" * 40)
    print("1. ✅ Never commit .env files to version control")
    print("2. ✅ Use environment variables for all secrets")
    print("3. ✅ Keep .env.example updated with required variables")
    print("4. ✅ Regularly rotate API keys and passwords")
    print("5. ✅ Use strong, unique passwords for each service")
    print("6. ✅ Enable 2FA on all accounts when possible")
    print("7. ✅ Monitor API usage for unusual activity")
    print("8. ✅ Use least privilege principle for API access")
    
    # Check if .env.example exists
    if os.path.exists('.env.example'):
        print("\n✅ .env.example file found - good for team collaboration")
    else:
        print("\n⚠️  Consider creating .env.example for team collaboration")
    
    print("\n" + "=" * 60)
    
    return len(issues_found) == 0

if __name__ == "__main__":
    is_secure = check_security()
    if is_secure:
        print("🎉 System is SECURE!")
    else:
        print("❌ Security issues need to be addressed!") 