# GitHub Repository Setup Guide

## 🎯 **Branching Strategy for 4-Person Team**

### **Main Branches:**
- **`main`** - Production-ready code
- **`develop`** - Integration branch for features

### **Feature Branches (One per person):**
- **`feature/matin-data-collection`** - Data collection and preprocessing
- **`feature/member2-model-training`** - Model training and evaluation  
- **`feature/member3-visualization`** - Data visualization and analysis
- **`feature/member4-api-integration`** - Spotify API integration

## 🚀 **Setup Commands**

### **1. Initialize Repository**
```bash
# In your project directory
git init
git add .
git commit -m "Initial commit: Song Mood Classification System"

# Create GitHub repository (do this on GitHub.com first)
git remote add origin https://github.com/yourusername/song-mood-classifier.git
git branch -M main
git push -u origin main
```

### **2. Create Development Branch**
```bash
git checkout -b develop
git push -u origin develop
```

### **3. Create Feature Branches**
```bash
# Each team member creates their branch
git checkout -b feature/matin-data-collection
git checkout -b feature/member2-model-training  
git checkout -b feature/member3-visualization
git checkout -b feature/member4-api-integration
```

## 📁 **Project Structure**
```
song-mood-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── kaggle_data.csv
│   └── spotify_predictions.csv
├── src/
│   ├── data_collection/
│   │   ├── kaggle_loader.py
│   │   └── spotify_api.py
│   ├── models/
│   │   ├── training.py
│   │   └── prediction.py
│   ├── visualization/
│   │   ├── plots.py
│   │   └── analysis.py
│   └── utils/
│       ├── preprocessing.py
│       └── evaluation.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_development.ipynb
│   └── results_analysis.ipynb
├── tests/
│   ├── test_data_collection.py
│   ├── test_models.py
│   └── test_visualization.py
└── docs/
    ├── project_proposal.md
    └── technical_documentation.md
```

## 🔄 **Workflow for Each Team Member**

### **Daily Workflow:**
```bash
# 1. Start your day
git checkout develop
git pull origin develop

# 2. Switch to your feature branch
git checkout feature/your-name-feature

# 3. Work on your feature
# ... make changes ...

# 4. Commit your changes
git add .
git commit -m "Add: [your feature description]"

# 5. Push to your branch
git push origin feature/your-name-feature

# 6. Create Pull Request on GitHub
```

### **Weekly Integration:**
```bash
# 1. Merge your feature into develop
git checkout develop
git merge feature/your-name-feature
git push origin develop

# 2. Update your feature branch
git checkout feature/your-name-feature
git merge develop
```

## 📋 **Team Responsibilities**

### **Member 1 (Matin) - Data Collection:**
- Kaggle dataset research and loading
- Data preprocessing and cleaning
- Feature engineering
- Data validation

### **Member 2 - Model Training:**
- Model selection and training
- Hyperparameter tuning
- Cross-validation
- Model evaluation

### **Member 3 - Visualization:**
- Data exploration plots
- Model performance visualizations
- Results presentation
- Dashboard creation

### **Member 4 - API Integration:**
- Spotify API setup
- Real-time prediction system
- Playlist creation
- Production deployment

## 🛠️ **Required Files**

### **`.gitignore`**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Jupyter
.ipynb_checkpoints/

# Data files
*.csv
*.json
*.pkl

# API keys
.env
config.ini

# OS
.DS_Store
Thumbs.db
```

### **`requirements.txt`**
```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
spotipy>=2.22.0
jupyter>=1.0.0
requests>=2.28.0
```

## 🎯 **GitHub Repository Settings**

### **Repository Settings:**
- **Visibility**: Public (for portfolio)
- **Issues**: Enabled
- **Projects**: Enabled
- **Wiki**: Enabled
- **Discussions**: Enabled

### **Branch Protection Rules:**
- Require pull request reviews
- Require status checks
- Require branches to be up to date
- Restrict pushes to main branch

## 📊 **Project Timeline**

### **Week 1: Setup & Data Collection**
- Repository setup
- Kaggle data research
- Initial data loading

### **Week 2: Model Development**
- Model training
- Hyperparameter tuning
- Initial evaluation

### **Week 3: Integration & Testing**
- Spotify API integration
- End-to-end testing
- Performance optimization

### **Week 4: Visualization & Documentation**
- Results visualization
- Documentation
- Final presentation

## 🚀 **Getting Started Commands**

```bash
# Clone the repository
git clone https://github.com/yourusername/song-mood-classifier.git
cd song-mood-classifier

# Install dependencies
pip install -r requirements.txt

# Create your feature branch
git checkout -b feature/your-name-feature

# Start working!
```
