@echo off
echo Pulling remote changes...
git pull origin main --allow-unrelated-histories --no-rebase -X ours --no-edit

echo Adding files...
git add .

echo Committing...
git commit -m "fix: merge remote changes"

echo Pushing...
git push -u origin main

echo Done!
pause
