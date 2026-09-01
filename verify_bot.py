#!/usr/bin/env python
"""
Telegram Bot Verification Script
Tests bot token and verifies connection before starting the bot
"""

import os
import sys
import logging
import asyncio

# Load environment variables from .env
def load_env():
    """Load .env file manually"""
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()

load_env()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_bot():
    """Verify Telegram bot token and connection"""
    
    # Load token
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not token:
        logger.error("Token not found in .env")
        logger.info("Add this to .env:")
        logger.info("   TELEGRAM_BOT_TOKEN=your_token_here")
        return False
    
    if token == "YOUR_TELEGRAM_BOT_TOKEN" or token.startswith("YOUR_"):
        logger.error("Token still contains placeholder")
        logger.info("Replace with real token from @BotFather")
        return False
    
    logger.info("Token found: {}...".format(token[:20]))
    
    # Test connection
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            url = "https://api.telegram.org/bot{}/getMe".format(token)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("ok"):
                        bot_info = data.get("result", {})
                        logger.info("Bot connection successful!")
                        logger.info("   Bot Name: {}".format(bot_info.get('first_name')))
                        logger.info("   Bot ID: {}".format(bot_info.get('id')))
                        logger.info("   Username: @{}".format(bot_info.get('username')))
                        return True
                    else:
                        logger.error("Bot error: {}".format(data.get('description')))
                        return False
                else:
                    logger.error("HTTP {}: Connection failed".format(resp.status))
                    return False
    
    except asyncio.TimeoutError:
        logger.error("Connection timeout (10s)")
        logger.info("   Check internet connection")
        return False
    except Exception as e:
        logger.error("Connection error: {}".format(e))
        return False

async def verify_dependencies():
    """Check required dependencies"""
    
    logger.info("\nChecking dependencies...")
    
    required = [
        ("telegram", "python-telegram-bot"),
        ("torch", "torch"),
        ("fastapi", "fastapi"),
        ("sqlalchemy", "sqlalchemy"),
        ("librosa", "librosa"),
        ("whisper", "openai-whisper"),
    ]
    
    missing = []
    for module, package in required:
        try:
            __import__(module)
            logger.info("   [OK] {}".format(package))
        except ImportError:
            logger.warning("   [MISS] {} (MISSING)".format(package))
            missing.append(package)
    
    if missing:
        logger.error("\nMissing packages: {}".format(", ".join(missing)))
        logger.info("   Run: pip install -r requirements.txt")
        return False
    
    return True

async def verify_database():
    """Check database initialization"""
    
    logger.info("\nChecking database...")
    
    try:
        from models import SessionLocal, Video
        session = SessionLocal()
        count = session.query(Video).count()
        session.close()
        logger.info("   [OK] Database connected ({} videos)".format(count))
        return True
    except Exception as e:
        logger.warning("   [ERR] Database error: {}".format(e))
        logger.info("   Run: python init.py")
        return False

async def verify_models():
    """Check if ML models are available"""
    
    logger.info("\nChecking ML models...")
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("   [OK] PyTorch ({})".format(device))
        
        if torch.cuda.is_available():
            logger.info("     GPU: {}".format(torch.cuda.get_device_name(0)))
            logger.info("     Memory: {:.1f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1024**3))
    except Exception as e:
        logger.warning("   [ERR] PyTorch error: {}".format(e))
    
    return True

async def main():
    """Run all verification checks"""
    
    print("\n" + "="*60)
    print("TELEGRAM BOT VERIFICATION")
    print("="*60)
    
    checks = [
        ("Telegram Connection", verify_bot),
        ("Dependencies", verify_dependencies),
        ("Database", verify_database),
        ("ML Models", verify_models),
    ]
    
    results = []
    for name, check in checks:
        try:
            result = await check()
            results.append((name, result))
        except Exception as e:
            logger.error(f"❌ {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print("{:8} - {}".format(status, name))
    
    print("="*60)
    
    if passed == total:
        print("\nAll checks passed! ({}/{})".format(passed, total))
        print("\nReady to start bot:")
        print("   python telegram_bot.py")
        return 0
    else:
        print("\n{} check(s) failed. Please fix above issues.".format(total - passed))
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
