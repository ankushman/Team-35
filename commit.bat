@echo off
echo Adding updated files...
git add .

echo Committing...
git commit -m "feat: interactive simulation toggle from dashboard to core loop"

echo Pushing to GitHub...
git push origin main

echo Done!
pause
