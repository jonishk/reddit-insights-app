@echo off
rem -------------------------------------------------
rem 1. Move to the directory where the script lives
cd /d "%~dp0"

rem 2. Activate the conda environment (adjust name if needed)
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" reditbot

rem 3. Launch the app without blocking the console
start "" /b python app.py

rem 4. Simple “loading” animation (10 seconds)
powershell -NoProfile -Command ^
    "for($i=0;$i -lt 10;$i++){ Write-Host -NoNewline '.'; Start-Sleep -Seconds 1 };" ^
    "Write-Host ' '"

rem 5. Open the app in the default browser
start "" http://localhost:8080/