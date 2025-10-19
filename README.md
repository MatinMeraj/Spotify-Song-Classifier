# Song Mood Classification System

A machine learning system for classifying songs into mood categories using audio features and machine learning algorithms.

## Project Overview
This project implements a machine learning system that automatically classifies songs into different mood categories (happy, chill, sad, hyped) based on their audio features. The system uses various machine learning algorithms to achieve this classification task.

## Milestone 1 Goals
- Basic end-to-end classifier working on labeled dataset
- Four distinct mood categories mapped
- Project repository set up in Git
- Model training and evaluation complete

## Mood Categories
- **Happy**: Upbeat songs with high energy and positive valence
- **Chill**: Relaxed songs with moderate energy and balanced mood
- **Sad**: Slow tempo songs with low energy and negative valence  
- **Hyped**: Fast tempo songs with high energy and very positive mood

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run the Classifier
```bash
python milestone1_mood_classifier.py
```

## Features

### Data Processing
- Dataset with 900 songs across multiple genres
- 9 audio features including tempo, energy, valence, and loudness
- Feature scaling and train/test split for model evaluation

### Machine Learning Models
- Random Forest: Ensemble method with feature importance analysis
- Logistic Regression: Linear classification with regularization
- K-Nearest Neighbors: Distance-based classification algorithm

### Evaluation
- 5-fold cross-validation for robust model evaluation
- Performance metrics including accuracy, precision, recall, and F1-score
- Comprehensive visualizations including confusion matrix and feature analysis

## Results

### Model Performance
- Best performing model: Random Forest with 81% accuracy
- 5-fold cross-validation with confidence intervals
- Test accuracy evaluated on 20% holdout set

### Generated Files
- `music_dataset.csv`: Complete dataset with features and labels
- `milestone1_model.pkl`: Trained model for predictions
- `milestone1_analysis.png`: Comprehensive analysis visualizations

## Usage Example

```python
from milestone1_mood_classifier import MoodClassifier

# Initialize classifier
classifier = MoodClassifier()

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

## Project Structure
```
song-mood-classifier/
├── milestone1_mood_classifier.py  # Main implementation
├── requirements.txt               # Dependencies
├── README.md                     # This file
├── music_dataset.csv             # Generated dataset
├── milestone1_model.pkl          # Trained model
└── milestone1_analysis.png      # Analysis visualizations
```

## Next Steps (Future Milestones)
- Milestone 2: Spotify API integration for real-time prediction
- Web Interface: User-friendly song mood prediction
- Playlist Generation: Create mood-based playlists
- Advanced Models: Deep learning and ensemble methods

## Team Collaboration
- Branch Strategy: Feature branches for each team member
- Code Review: Pull requests for integration
- Documentation: Comprehensive README and code comments

## Technical Details

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

## Business Applications
- Music Recommendation: Personalized playlist generation
- Content Curation: Automated music categorization
- Mood-based Search: Find songs matching desired emotional state
- Music Analytics: Understanding listener preferences

## License
This project is part of CMPT 310 - Introduction to Artificial Intelligence course work.