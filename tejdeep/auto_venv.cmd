@echo off
cd /d %CD%
echo 🔍 Current directory: %CD%

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    echo 🐍 Activated local .venv
) else if exist "E:\\tejcsDevWork\\hackathon2025\\tejdeep\\.venv\\Scripts\\activate.bat" (
    call "E:\\tejcsDevWork\\hackathon2025\\tejdeep\\.venv\\Scripts\\activate.bat"
     call cd tejdeep
    echo 🐍 Activated fallback tejdeep .venv
) else (
    echo ⚠️ No virtual environment found.
)
