@echo off
echo ========================================================
echo      AUTO HIRE PRO - DEPLOYMENT SCRIPT
echo ========================================================
echo.
echo 1. Adding all changes...
"C:\Program Files\Git\cmd\git.exe" add .

echo.
echo 2. Committing changes...
set /p commit_msg="Enter commit message (Press Enter for 'Update'): "
if "%commit_msg%"=="" set commit_msg=Update
"C:\Program Files\Git\cmd\git.exe" commit -m "%commit_msg%"

echo.
echo 3. Pushing to GitHub...
"C:\Program Files\Git\cmd\git.exe" push

echo.
echo ========================================================
echo      DONE! Your changes are on their way to the cloud.
echo ========================================================
echo.
pause
