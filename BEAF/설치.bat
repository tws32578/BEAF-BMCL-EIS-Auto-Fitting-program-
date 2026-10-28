@echo off
chcp 65001 > nul
cd /d "%~dp0"
set "CONDA=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not exist "%CONDA%" (
  echo [BEAF] Anaconda 를 찾지 못했습니다: %CONDA%
  pause
  exit /b 1
)
"%CONDA%" create -y -n beaf python=3.12
set "PY=%USERPROFILE%\anaconda3\envs\beaf\python.exe"
"%PY%" -m pip install -r requirements.txt -c constraints.txt
echo [BEAF] 설치 완료. 실행.bat 을 더블클릭하세요.
pause
