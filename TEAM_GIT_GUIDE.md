# Team Git Workflow Guide
**Complete guide for 4-person team collaboration**

## 🚀 **Step 1: Create GitHub Repository**

### **For Matin (Repository Owner):**
1. Go to [GitHub.com](https://github.com) and sign in
2. Click **"New Repository"** (green button)
3. Repository settings:
   - **Name**: `song-mood-classifier`
   - **Description**: "AI-powered song mood classification using machine learning"
   - **Visibility**: **Public** (for portfolio)
   - **Initialize**: ❌ **Don't** check "Add README" (we already have one)
4. Click **"Create Repository"**

### **Connect Local to GitHub:**
```bash
# In your project directory
git remote add origin https://github.com/YOUR_USERNAME/song-mood-classifier.git
git branch -M main
git push -u origin main
git push -u origin develop
```

---

## 👥 **Step 2: Team Member Setup**

### **For Each Team Member:**
```bash
# 1. Clone the repository
git clone https://github.com/MatinMeraj/Spotify-Song-Classifier.git
cd song-mood-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your feature branch
git checkout -b feature/your-name-your-feature
# Examples:
# git checkout -b feature/matin-data-collection
# git checkout -b feature/sarah-model-training
# git checkout -b feature/john-visualization
# git checkout -b feature/mike-api-integration
```

---

## 🔄 **Daily Git Workflow**

### **Morning Routine (Start of Day):**
```bash
# 1. Check current status
git status

# 2. Switch to develop branch
git checkout develop

# 3. Pull latest changes
git pull origin develop

# 4. Switch to your feature branch
git checkout feature/your-name-your-feature

# 5. Merge latest develop into your branch
git merge develop
```

### **During Work (Multiple Times Per Day):**
```bash
# 1. Check what you've changed
git status

# 2. Add your changes
git add .                    # Add all changes
# OR
git add filename.py          # Add specific file

# 3. Commit with descriptive message
git commit -m "Add: feature description"
# Examples:
# git commit -m "Add: data preprocessing functions"
# git commit -m "Fix: model training bug"
# git commit -m "Update: visualization code"
```

### **End of Day (Push Your Work):**
```bash
# 1. Push your feature branch
git push origin feature/your-name-your-feature

# 2. Create Pull Request on GitHub (see below)
```

---

## 🌿 **Branching Strategy**

### **Main Branches:**
- **`main`**: Production-ready code (only Matin merges here)
- **`develop`**: Integration branch (team merges here)

### **Feature Branches (One per person):**
- **`feature/matin-data-collection`**: Data loading and preprocessing
- **`feature/sarah-model-training`**: Model training and evaluation
- **`feature/john-visualization`**: Data visualization and analysis
- **`feature/mike-api-integration`**: Spotify API integration

### **Branch Commands:**
```bash
# Create new branch
git checkout -b feature/your-name-feature

# Switch between branches
git checkout branch-name

# List all branches
git branch -a

# Delete local branch (after merging)
git branch -d feature/your-name-feature
```

---

## 🔀 **Pull Request Workflow**

### **Creating Pull Request:**
1. **Push your feature branch:**
   ```bash
   git push origin feature/your-name-feature
   ```

2. **Go to GitHub repository**
3. **Click "Compare & pull request"** (appears after push)
4. **Fill out PR details:**
   - **Title**: "Add: [your feature description]"
   - **Description**: What you implemented
   - **Reviewers**: Assign team members
5. **Click "Create pull request"**

### **Reviewing Pull Request:**
1. **Go to "Pull requests" tab**
2. **Click on the PR**
3. **Review the code changes**
4. **Add comments if needed**
5. **Approve or request changes**

### **Merging Pull Request:**
1. **After approval, click "Merge pull request"**
2. **Delete the feature branch** (GitHub will ask)
3. **Update your local repository:**
   ```bash
   git checkout develop
   git pull origin develop
   ```

---

## 🚨 **Common Git Scenarios**

### **Scenario 1: Someone else pushed to develop**
```bash
# You're working on your feature branch
git checkout develop
git pull origin develop
git checkout feature/your-name-feature
git merge develop
```

### **Scenario 2: Merge conflicts**
```bash
# When you get merge conflicts:
# 1. Open the conflicted file
# 2. Look for <<<<<<< ======= >>>>>>> markers
# 3. Choose which code to keep
# 4. Remove the markers
# 5. Add and commit the resolved file
git add filename.py
git commit -m "Resolve: merge conflict in filename.py"
```

### **Scenario 3: Accidentally committed to wrong branch**
```bash
# If you committed to main instead of your feature branch:
git checkout main
git reset --soft HEAD~1  # Undo last commit but keep changes
git checkout feature/your-name-feature
git add .
git commit -m "Add: your changes"
```

### **Scenario 4: Undo last commit (but keep changes)**
```bash
git reset --soft HEAD~1
```

### **Scenario 5: Undo last commit (and lose changes)**
```bash
git reset --hard HEAD~1
```

---

## 📋 **Commit Message Guidelines**

### **Good Commit Messages:**
```
Add: data preprocessing functions
Fix: model training accuracy issue
Update: visualization colors
Remove: unused import statements
Refactor: code organization
```

### **Bad Commit Messages:**
```
❌ "stuff"
❌ "changes"
❌ "fix"
❌ "update"
❌ "work"
```

### **Commit Message Format:**
```
Action: Description

Examples:
Add: new feature description
Fix: bug description
Update: what was updated
Remove: what was removed
Refactor: what was reorganized
```

---

## 🛠️ **Useful Git Commands**

### **Status and Information:**
```bash
git status                    # Show current status
git log --oneline           # Show commit history
git diff                    # Show unstaged changes
git diff --staged           # Show staged changes
```

### **Undoing Things:**
```bash
git checkout -- filename     # Undo changes to file
git reset HEAD filename     # Unstage file
git reset --soft HEAD~1     # Undo commit, keep changes
git reset --hard HEAD~1     # Undo commit, lose changes
```

### **Branch Management:**
```bash
git branch                  # List local branches
git branch -r              # List remote branches
git branch -a              # List all branches
git checkout -b new-branch # Create and switch to new branch
git branch -d branch-name   # Delete local branch
```

---

## 🎯 **Team Responsibilities**

### **Matin (Data Collection):**
- Kaggle dataset research and loading
- Data preprocessing and cleaning
- Feature engineering
- Data validation

### **Sarah (Model Training):**
- Model selection and training
- Hyperparameter tuning
- Cross-validation
- Model evaluation

### **John (Visualization):**
- Data exploration plots
- Model performance visualizations
- Results presentation
- Dashboard creation

### **Mike (API Integration):**
- Spotify API setup
- Real-time prediction system
- Playlist creation
- Production deployment

---

## 🚀 **Quick Start Checklist**

### **For New Team Members:**
- [ ] Clone repository
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Create feature branch
- [ ] Make first commit
- [ ] Push to GitHub
- [ ] Create first pull request

### **Daily Checklist:**
- [ ] Pull latest changes from develop
- [ ] Work on your feature
- [ ] Commit frequently with good messages
- [ ] Push your work
- [ ] Create pull request when feature is complete

### **Weekly Checklist:**
- [ ] Review and merge pull requests
- [ ] Update documentation
- [ ] Test integrated features
- [ ] Plan next week's work

---

## 🆘 **Emergency Procedures**

### **If You Break Something:**
1. **Don't panic!**
2. **Check git status**: `git status`
3. **See what changed**: `git diff`
4. **Undo if needed**: `git checkout -- filename`
5. **Ask for help** in team chat

### **If You Lose Your Work:**
```bash
# Check git log for recent commits
git log --oneline

# Reset to a previous commit
git reset --hard COMMIT_HASH

# Or check reflog for lost commits
git reflog
```

### **If You're Stuck:**
1. **Check this guide first**
2. **Ask in team chat**
3. **Google the error message**
4. **Ask Matin for help**

---

## 📚 **Additional Resources**

- **GitHub Desktop**: GUI for Git (easier for beginners)
- **VS Code Git Integration**: Built-in Git support
- **GitHub Docs**: https://docs.github.com/
- **Git Tutorial**: https://learngitbranching.js.org/

---

## 🎯 **Remember:**
- **Commit often** (small, frequent commits)
- **Write good commit messages**
- **Pull before you push**
- **Ask for help when stuck**
- **Review each other's code**
- **Keep the main branch clean**

**Happy coding! 🚀**
