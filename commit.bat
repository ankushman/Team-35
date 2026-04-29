@echo off
echo Configuring git user...
git config user.name "ankushman"
git config user.email "ankushman@users.noreply.github.com"

echo Adding files...
git add .

echo Committing changes...
git commit -m "feat: UI/UX overhaul with circular gauges and dark/light mode"

echo Renaming branch to main...
git branch -M main

echo Pulling remote changes to sync...
git pull origin main --rebase

echo Pushing to GitHub...
git push -u origin main

echo Done!
pause
