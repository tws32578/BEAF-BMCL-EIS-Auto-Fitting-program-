@echo off
chcp 65001 > nul
cd /d "%~dp0"
set "PY=%USERPROFILE%\anaconda3\envs\beaf\python.exe"
if not exist "%PY%" (
  echo [BEAF] beaf 환경이 없습니다. 먼저 설치.bat 을 실행하세요.
  pause
  exit /b 1
)
echo [BEAF] 서버 시작: http://127.0.0.1:8765   (이 창을 닫으면 종료)
"%PY%" -m beaf.server
pause
