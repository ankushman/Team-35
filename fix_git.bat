@echo off
echo Aborting the broken rebase...
git rebase --abort

echo Pulling remote changes (allowing unrelated histories and keeping local files)...
git pull origin main -X ours --allow-unrelated-histories --no-rebase -m "Merge remote changes"

echo Adding files...
git add .

echo Committing any remaining changes...
git commit -m "fix: resolve conflicts and push UI updates"

echo Pushing to GitHub...
git push -u origin main

echo Done!
pause
