# Song Mood Classification System

A comprehensive AI system for classifying songs based on mood using audio features and lyrics analysis. This project demonstrates advanced machine learning techniques including ensemble methods, uncertainty handling, and confidence-based playlist generation.

## 🎯 Project Overview

This system classifies songs into four mood categories:
- **Happy**: High energy, high valence, fast tempo
- **Chill**: Low energy, medium valence, slow tempo  
- **Sad**: Low energy, low valence, slow tempo
- **Hyped**: High energy, high valence, very fast tempo

## 🚀 Features

### Core Functionality
- **Multi-modal Analysis**: Combines audio features and lyrics analysis
- **Multiple ML Algorithms**: KNN, Logistic Regression, Random Forest, SVM
- **Ensemble Methods**: Voting classifier combining multiple models
- **Uncertainty Handling**: Confidence-based prediction filtering
- **Hyperparameter Tuning**: Automated model optimization

### Advanced Features
- **Confidence Visualization**: Detailed analysis of prediction confidence
- **Mood Mapping**: Visual representation of feature relationships
- **Playlist Generation**: Confidence-ranked song recommendations
- **Performance Metrics**: Comprehensive evaluation with confusion matrices

## 📊 Technical Implementation

### Data Processing
- **Audio Features**: Tempo, energy, valence, loudness, danceability, speechiness, acousticness, instrumentalness, liveness
- **Lyrics Features**: Word count, sentiment analysis, emotional intensity, keyword density
- **Data Scaling**: StandardScaler for feature normalization

### Machine Learning Models
1. **Baseline Models**:
   - Rule-based classifier (BPM and energy thresholds)
   - K-Nearest Neighbors
   - Logistic Regression

2. **Advanced Models**:
   - Random Forest with hyperparameter tuning
   - Ensemble voting classifier
   - Support Vector Machine

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class performance metrics
- **Confusion Matrices**: Visual performance analysis
- **Cross-validation**: 5-fold CV for robust evaluation

## 🎵 Results

### Model Performance
- **Audio Model**: 79.0% accuracy
- **Lyrics Model**: 74.0% accuracy  
- **Rule-based Baseline**: 57.5% accuracy
- **Ensemble Model**: 78.5% accuracy
- **Tuned Random Forest**: 77.0% accuracy

### Key Insights
- Audio features outperform lyrics for mood classification
- Ensemble methods provide robust predictions
- Confidence analysis reveals prediction reliability
- High-confidence predictions show improved accuracy

## 🛠️ Installation & Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
# Run basic classifier
python song_mood_classifier.py

# Run advanced classifier with ensemble methods
python advanced_mood_classifier.py
```

### Generated Files
- `song_mood_dataset.csv`: Complete dataset with features and labels
- `confusion_matrix_*.png`: Model performance visualizations
- `mood_analysis.png`: Feature relationship analysis
- `confidence_analysis.png`: Confidence score distributions
- `mood_map.png`: Mood prediction visualization

## 📈 Visualizations

The system generates comprehensive visualizations:
- **Confusion Matrices**: Model performance by mood category
- **Feature Importance**: Most influential audio features
- **Confidence Analysis**: Prediction reliability assessment
- **Mood Maps**: Visual feature relationships
- **Playlist Rankings**: Confidence-sorted song recommendations

## 🔬 Technical Specifications

### Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **spotipy**: Spotify API integration (optional)

### Model Architecture
- **Feature Engineering**: 9 audio + 5 lyrics features
- **Data Splitting**: 80/20 train/test split with stratification
- **Cross-validation**: 5-fold CV for model selection
- **Hyperparameter Tuning**: GridSearchCV for optimization

## 🎯 Business Applications

This system can be applied to:
- **Music Recommendation**: Personalized playlist generation
- **Content Curation**: Automated music categorization
- **Mood-based Search**: Find songs matching desired emotional state
- **Music Analytics**: Understanding listener preferences

## 📝 Future Enhancements

- **Deep Learning**: Neural network implementation with PyTorch
- **Real-time Processing**: Live audio analysis capabilities
- **Multi-language Support**: Lyrics analysis in multiple languages
- **User Feedback**: Learning from user preferences
- **API Integration**: Real-time Spotify playlist analysis

## 🏆 Project Achievements

This project demonstrates proficiency in:
- **Machine Learning**: Multiple algorithms and ensemble methods
- **Data Science**: Feature engineering and model evaluation
- **Python Programming**: Object-oriented design and API integration
- **Data Visualization**: Comprehensive analysis and reporting
- **Statistical Analysis**: Confidence intervals and uncertainty handling
