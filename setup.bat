@echo off

REM Making an virtual env

if exist ".\.venv\" (
    echo .venv is aleardy installed!
) else (
    echo Installing .venv
    python -m venv .venv
    echo VENV INSTALLED!
)

call :Verify_package numpy
call :Verify_package tensorflow
call :Verify_package keras
call :Verify_package tkinter
call :Verify_package unidecode

echo REMEMBER to check the config.toml file!

pause

.\.venv\Scripts\python.exe .\WinVersion\Connect_NN.py

EXIT /B %ERRORLEVEL%

:Verify_package

echo Checking %~1

.\.venv\Scripts\python.exe -c "import %~1" 2>nul

if %errorlevel% neq 0 (
    echo Installing %~1
    .\.venv\Scripts\pip.exe install %~1
    if %errorlevel% neq 0 (
        echo Error when installing %~1, you may need to check it manualy.
        echo Aborting execution
        EXIT /B 1
        exit /b
    ) else (
        echo Success!
    )
) else (
    echo %~1 installed
)
EXIT /B 0