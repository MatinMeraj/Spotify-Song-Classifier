# Song Mood Classification System - Milestone 1

A machine learning system for classifying songs into mood categories using audio features from Kaggle datasets.

## 🎯 Milestone 1 Goals
- ✅ Basic end-to-end classifier working on labeled dataset
- ✅ Four distinct mood categories mapped
- ✅ Project repository set up in Git
- ✅ Model training and evaluation complete

## 🎵 Mood Categories
- **Happy**: Upbeat, high energy, positive valence
- **Chill**: Relaxed, moderate energy, balanced mood
- **Sad**: Slow tempo, low energy, negative valence  
- **Hyped**: Fast tempo, high energy, very positive

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Milestone 1
```bash
python milestone1_kaggle_classifier.py
```

## 📊 Features

### Data Processing
- **Dataset**: 800 songs from Kaggle-style dataset (200 per mood)
- **Features**: 9 audio features (tempo, energy, valence, loudness, etc.)
- **Preprocessing**: Feature scaling and train/test split

### Machine Learning Models
- **Random Forest**: Ensemble method with feature importance
- **Logistic Regression**: Linear classification with regularization
- **K-Nearest Neighbors**: Distance-based classification

### Evaluation
- **Cross-validation**: 5-fold CV for robust evaluation
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Visualizations**: Confusion matrix, feature analysis, mood distribution

## 📈 Results

### Model Performance
- **Best Model**: Random Forest (typically 75-80% accuracy)
- **Cross-validation**: 5-fold CV with confidence intervals
- **Test Accuracy**: Evaluated on 20% holdout set

### Generated Files
- `kaggle_music_dataset.csv`: Complete dataset with features and labels
- `milestone1_model.pkl`: Trained model for predictions
- `milestone1_analysis.png`: Comprehensive analysis visualizations

## 🎯 Usage Example

```python
from milestone1_kaggle_classifier import KaggleMoodClassifier

# Initialize classifier
classifier = KaggleMoodClassifier()

# Load trained model
classifier.load_model('milestone1_model.pkl')

# Predict mood for new song
prediction, confidence = classifier.predict_new_song(
    tempo=120,      # BPM
    energy=0.8,     # 0-1 scale
    valence=0.7,    # 0-1 scale
    loudness=-5     # dB
)

print(f"Predicted mood: {prediction}")
print(f"Confidence: {confidence:.3f}")
```

## 📁 Project Structure
```
song-mood-classifier/
├── milestone1_kaggle_classifier.py  # Main implementation
├── requirements.txt                 # Dependencies
├── README.md                       # This file
├── kaggle_music_dataset.csv        # Generated dataset
├── milestone1_model.pkl            # Trained model
└── milestone1_analysis.png         # Analysis visualizations
```

## 🔄 Next Steps (Future Milestones)
- **Milestone 2**: Spotify API integration for real-time prediction
- **Web Interface**: User-friendly song mood prediction
- **Playlist Generation**: Create mood-based playlists
- **Advanced Models**: Deep learning and ensemble methods

## 👥 Team Collaboration
- **Branch Strategy**: Feature branches for each team member
- **Code Review**: Pull requests for integration
- **Documentation**: Comprehensive README and code comments

## 📊 Technical Details

### Dependencies
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scikit-learn >= 1.1.0

### Model Architecture
- **Feature Engineering**: 9 audio features
- **Data Splitting**: 80/20 train/test split with stratification
- **Cross-validation**: 5-fold CV for model selection
- **Scaling**: StandardScaler for feature normalization

## 🎯 Business Applications
- **Music Recommendation**: Personalized playlist generation
- **Content Curation**: Automated music categorization
- **Mood-based Search**: Find songs matching desired emotional state
- **Music Analytics**: Understanding listener preferences

## 📝 License
This project is part of CMPT 310 - Introduction to Artificial Intelligence course work.