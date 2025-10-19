# GitHub Repository Setup - Step by Step

## 🚀 **Step 1: Create GitHub Repository**

### **Go to GitHub.com:**
1. **Sign in** to your GitHub account
2. **Click the "+" icon** in the top right corner
3. **Select "New repository"**

### **Repository Settings:**
- **Repository name**: `song-mood-classifier`
- **Description**: `AI-powered song mood classification using machine learning`
- **Visibility**: **Public** ✅ (for portfolio)
- **Initialize this repository with**:
  - ❌ **Don't check "Add a README file"** (we already have one)
  - ❌ **Don't check "Add .gitignore"** (we already have one)
  - ❌ **Don't check "Choose a license"** (optional)

4. **Click "Create repository"**

---

## 🔗 **Step 2: Connect Local Repository to GitHub**

### **Copy the Repository URL:**
After creating the repository, GitHub will show you commands. **Copy the HTTPS URL** (looks like):
```
https://github.com/YOUR_USERNAME/song-mood-classifier.git
```

### **Run These Commands in Your Terminal:**
```bash
# Navigate to your project directory
cd /Users/matinmeraj/Documents/sfu/term4/cmpt310/project

# Add GitHub as remote origin
git remote add origin https://github.com/YOUR_USERNAME/song-mood-classifier.git

# Rename main branch (if needed)
git branch -M main

# Push your code to GitHub
git push -u origin main

# Push develop branch
git push -u origin develop
```

---

## ✅ **Step 3: Verify Everything Works**

### **Check Your Repository:**
1. **Go to your GitHub repository**
2. **You should see all your files:**
   - `milestone1_kaggle_classifier.py`
   - `README.md`
   - `requirements.txt`
   - `kaggle_music_dataset.csv`
   - `milestone1_model.pkl`
   - `TEAM_GIT_GUIDE.md`

### **Test Team Member Access:**
1. **Share the repository URL** with your team
2. **Each member should be able to clone it:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/song-mood-classifier.git
   ```

---

## 👥 **Step 4: Team Collaboration Setup**

### **Repository Settings (Optional but Recommended):**
1. **Go to repository Settings**
2. **Click "Manage access"**
3. **Click "Invite a collaborator"**
4. **Add your team members' GitHub usernames**

### **Branch Protection (Optional):**
1. **Go to Settings > Branches**
2. **Add rule for main branch:**
   - ✅ Require pull request reviews
   - ✅ Require status checks
   - ✅ Restrict pushes to main branch

---

## 🎯 **Step 5: Share with Team**

### **Send This to Your Team:**
```
🎵 Song Mood Classification Project

Repository: https://github.com/YOUR_USERNAME/song-mood-classifier

Setup Instructions:
1. Clone the repository
2. Read TEAM_GIT_GUIDE.md
3. Create your feature branch
4. Start working on your assigned feature

Team Responsibilities:
- Matin: Data Collection
- Sarah: Model Training  
- John: Visualization
- Mike: API Integration
```

---

## 🚨 **Troubleshooting**

### **If "remote origin already exists":**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/song-mood-classifier.git
```

### **If "authentication failed":**
1. **Use GitHub Personal Access Token** instead of password
2. **Or use SSH keys** (more secure)

### **If "repository not found":**
1. **Check the URL** is correct
2. **Make sure repository is public**
3. **Verify your GitHub username**

---

## 🎉 **Success!**

Once everything is set up, your team can:
- ✅ Clone the repository
- ✅ Create feature branches
- ✅ Make commits and push changes
- ✅ Create pull requests
- ✅ Collaborate effectively

**Your project is now ready for team collaboration! 🚀**
