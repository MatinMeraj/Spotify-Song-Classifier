"""
Advanced Song Mood Classification System with Ensemble Methods and Uncertainty Handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AdvancedSongMoodClassifier:
    def __init__(self):
        self.audio_models = {}
        self.lyrics_models = {}
        self.ensemble_model = None
        self.scalers = {}
        self.mood_labels = ['happy', 'chill', 'sad', 'hyped']
        
    def load_dataset(self, filename='song_mood_dataset.csv'):
        """Load the generated dataset"""
        return pd.read_csv(filename)
    
    def create_ensemble_model(self, X_train, y_train):
        """Create ensemble model with multiple algorithms"""
        # Individual models
        knn = KNeighborsClassifier(n_neighbors=5)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        svm = SVC(probability=True, random_state=42)
        
        # Ensemble with voting
        ensemble = VotingClassifier([
            ('knn', knn),
            ('lr', lr),
            ('rf', rf),
            ('svm', svm)
        ], voting='soft')
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train ensemble
        ensemble.fit(X_train_scaled, y_train)
        
        # Cross-validation
        scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5)
        print(f"Ensemble Model CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        return ensemble, scaler
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for best model"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Random Forest hyperparameter tuning
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
        rf_grid.fit(X_train_scaled, y_train)
        
        print(f"Best Random Forest params: {rf_grid.best_params_}")
        print(f"Best Random Forest score: {rf_grid.best_score_:.3f}")
        
        return rf_grid.best_estimator_, scaler
    
    def uncertainty_handling(self, model, scaler, X_test, threshold=0.6):
        """Handle uncertainty in predictions based on confidence scores"""
        X_test_scaled = scaler.transform(X_test)
        probabilities = model.predict_proba(X_test_scaled)
        
        # Get confidence scores
        max_probs = np.max(probabilities, axis=1)
        predictions = model.predict(X_test_scaled)
        
        # Identify uncertain predictions
        uncertain_mask = max_probs < threshold
        uncertain_predictions = predictions[uncertain_mask]
        uncertain_confidences = max_probs[uncertain_mask]
        
        print(f"\nUncertainty Analysis:")
        print(f"Total predictions: {len(predictions)}")
        print(f"Uncertain predictions (<{threshold}): {np.sum(uncertain_mask)}")
        print(f"Average confidence: {np.mean(max_probs):.3f}")
        print(f"Uncertain predictions confidence: {np.mean(uncertain_confidences):.3f}")
        
        return predictions, max_probs, uncertain_mask
    
    def create_confidence_visualization(self, y_true, predictions, confidences):
        """Create confidence visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confidence distribution
        axes[0,0].hist(confidences, bins=20, alpha=0.7, color='skyblue')
        axes[0,0].axvline(np.mean(confidences), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(confidences):.3f}')
        axes[0,0].set_xlabel('Confidence Score')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Confidence Score Distribution')
        axes[0,0].legend()
        
        # Confidence vs Accuracy
        correct_mask = (y_true == predictions)
        axes[0,1].scatter(confidences[correct_mask], np.ones(np.sum(correct_mask)), 
                          alpha=0.6, color='green', label='Correct')
        axes[0,1].scatter(confidences[~correct_mask], np.ones(np.sum(~correct_mask)), 
                          alpha=0.6, color='red', label='Incorrect')
        axes[0,1].set_xlabel('Confidence Score')
        axes[0,1].set_ylabel('Prediction')
        axes[0,1].set_title('Confidence vs Prediction Accuracy')
        axes[0,1].legend()
        
        # Confidence by mood
        for i, mood in enumerate(self.mood_labels):
            mood_mask = (y_true == mood)
            if np.any(mood_mask):
                axes[1,0].hist(confidences[mood_mask], alpha=0.6, label=mood, bins=10)
        axes[1,0].set_xlabel('Confidence Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Confidence Distribution by Mood')
        axes[1,0].legend()
        
        # High confidence predictions
        high_conf_mask = confidences > 0.8
        high_conf_accuracy = np.mean(correct_mask[high_conf_mask]) if np.any(high_conf_mask) else 0
        axes[1,1].bar(['All Predictions', 'High Confidence'], 
                     [np.mean(correct_mask), high_conf_accuracy], 
                     color=['lightblue', 'darkblue'])
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_title('Accuracy: All vs High Confidence')
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_mood_map(self, df, predictions, confidences):
        """Create mood map visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Tempo vs Energy with confidence
        scatter = axes[0,0].scatter(df['tempo'], df['energy'], c=confidences, 
                                   cmap='viridis', alpha=0.6, s=30)
        axes[0,0].set_xlabel('Tempo (BPM)')
        axes[0,0].set_ylabel('Energy')
        axes[0,0].set_title('Tempo vs Energy (colored by confidence)')
        plt.colorbar(scatter, ax=axes[0,0])
        
        # Valence vs Loudness with predictions
        colors = ['red', 'blue', 'green', 'orange']
        for i, mood in enumerate(self.mood_labels):
            mood_mask = (predictions == mood)
            if np.any(mood_mask):
                axes[0,1].scatter(df.loc[mood_mask, 'valence'], 
                                 df.loc[mood_mask, 'loudness'], 
                                 c=colors[i], label=mood, alpha=0.6, s=30)
        axes[0,1].set_xlabel('Valence')
        axes[0,1].set_ylabel('Loudness (dB)')
        axes[0,1].set_title('Valence vs Loudness (colored by prediction)')
        axes[0,1].legend()
        
        # Confidence by actual mood
        for mood in self.mood_labels:
            mood_mask = (df['mood'] == mood)
            if np.any(mood_mask):
                axes[1,0].hist(confidences[mood_mask], alpha=0.6, label=mood, bins=10)
        axes[1,0].set_xlabel('Confidence Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Confidence by Actual Mood')
        axes[1,0].legend()
        
        # Model performance by mood
        mood_accuracies = []
        for mood in self.mood_labels:
            mood_mask = (df['mood'] == mood)
            if np.any(mood_mask):
                mood_accuracy = np.mean(predictions[mood_mask] == mood)
                mood_accuracies.append(mood_accuracy)
            else:
                mood_accuracies.append(0)
        
        axes[1,1].bar(self.mood_labels, mood_accuracies, color=['red', 'blue', 'green', 'orange'])
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_title('Model Performance by Mood')
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('mood_map.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_playlist_by_confidence(self, df, predictions, confidences, mood='happy'):
        """Create playlist sorted by confidence for a specific mood"""
        mood_mask = (predictions == mood)
        mood_songs = df[mood_mask].copy()
        mood_songs['confidence'] = confidences[mood_mask]
        
        # Sort by confidence (highest first)
        mood_songs = mood_songs.sort_values('confidence', ascending=False)
        
        print(f"\n🎵 Top 10 {mood.title()} Songs (by confidence):")
        print("=" * 50)
        for i, (_, song) in enumerate(mood_songs.head(10).iterrows()):
            print(f"{i+1:2d}. {song['track_name']} - {song['artists']} "
                  f"(Confidence: {song['confidence']:.3f})")
        
        return mood_songs
    
    def advanced_evaluation(self, y_true, y_pred, model_name):
        """Advanced model evaluation with detailed metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n{model_name} Advanced Evaluation:")
        print("=" * 40)
        print(f"Overall Accuracy: {accuracy:.3f}")
        
        # Per-class metrics
        report = classification_report(y_true, y_pred, target_names=self.mood_labels, output_dict=True)
        
        print(f"\nPer-Class Performance:")
        for mood in self.mood_labels:
            if mood in report:
                precision = report[mood]['precision']
                recall = report[mood]['recall']
                f1 = report[mood]['f1-score']
                support = report[mood]['support']
                print(f"{mood:8s}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={support}")
        
        return accuracy, report

def main():
    """Main execution function for advanced classifier"""
    print("🚀 Advanced Song Mood Classification System")
    print("=" * 60)
    
    # Initialize classifier
    classifier = AdvancedSongMoodClassifier()
    
    # Load dataset
    print("\n📊 Loading dataset...")
    df = classifier.load_dataset()
    print(f"Dataset shape: {df.shape}")
    
    # Prepare features
    audio_features = ['tempo', 'energy', 'valence', 'loudness', 'danceability', 
                     'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    lyrics_features = ['word_count', 'positive_sentiment', 'negative_sentiment', 
                      'emotional_intensity', 'keyword_density']
    
    X_audio = df[audio_features]
    X_lyrics = df[lyrics_features]
    y = df['mood']
    
    # Split data
    X_audio_train, X_audio_test, X_lyrics_train, X_lyrics_test, y_train, y_test = train_test_split(
        X_audio, X_lyrics, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train ensemble model
    print("\n🤖 Training ensemble model...")
    ensemble_model, audio_scaler = classifier.create_ensemble_model(X_audio_train, y_train)
    
    # Hyperparameter tuning
    print("\n🔧 Performing hyperparameter tuning...")
    best_model, tuned_scaler = classifier.hyperparameter_tuning(X_audio_train, y_train)
    
    # Make predictions with uncertainty handling
    print("\n🔮 Making predictions with uncertainty analysis...")
    predictions, confidences, uncertain_mask = classifier.uncertainty_handling(
        best_model, tuned_scaler, X_audio_test, threshold=0.6
    )
    
    # Advanced evaluation
    accuracy, report = classifier.advanced_evaluation(y_test, predictions, "Tuned Random Forest")
    
    # Create visualizations
    print("\n📈 Creating advanced visualizations...")
    classifier.create_confidence_visualization(y_test, predictions, confidences)
    classifier.create_mood_map(df.iloc[y_test.index], predictions, confidences)
    
    # Create playlists by confidence
    print("\n🎵 Creating playlists by confidence...")
    for mood in classifier.mood_labels:
        classifier.create_playlist_by_confidence(df.iloc[y_test.index], predictions, confidences, mood)
    
    # Model comparison
    print("\n🏆 Advanced Model Comparison:")
    print("=" * 40)
    
    # Compare different models
    models = {
        'Ensemble': ensemble_model,
        'Tuned Random Forest': best_model
    }
    
    for name, model in models.items():
        if name == 'Ensemble':
            pred = model.predict(audio_scaler.transform(X_audio_test))
        else:
            pred = model.predict(tuned_scaler.transform(X_audio_test))
        
        acc = accuracy_score(y_test, pred)
        print(f"{name:20s}: {acc:.3f}")
    
    print("\n✅ Advanced Song Mood Classification System Complete!")
    return classifier, df

if __name__ == "__main__":
    classifier, dataset = main()
