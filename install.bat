@echo off
setlocal
pushd "%~dp0"

call :execute_installation
set EXIT_CODE=%errorlevel%

call deactivate 2>nul
popd

if "%~1" neq "nopause" pause
exit /b %EXIT_CODE%

:execute_installation
if not exist .venv (
    echo Creating virtual environment for stt-tts...
    py -3.12 -m venv .venv 2>nul || python -m venv .venv || exit /b 1
)

call .venv\Scripts\activate.bat
call pip install --upgrade pip
call pip install -r requirements.txt || exit /b 1

echo Installation for stt-tts complete


if not exist .venv_tools (
    echo Creating virtual environment for stt-tts preparation tools...
    py -3.12 -m venv .venv_tools 2>nul || python -m venv .venv_tools || exit /b 1
)

call .venv_tools\Scripts\activate.bat
call pip install --upgrade pip
call pip install -r requirements-tools.txt || exit /b 1

python install.py || exit /b 1

echo Installation for stt-tts preparation tools complete
goto :eof