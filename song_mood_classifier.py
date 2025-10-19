"""
Song Mood Classification System
A comprehensive AI system for classifying songs based on mood using audio features and lyrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SongMoodClassifier:
    def __init__(self):
        self.audio_model = None
        self.lyrics_model = None
        self.scaler = StandardScaler()
        self.mood_labels = ['happy', 'chill', 'sad', 'hyped']
        
    def generate_synthetic_dataset(self, n_samples=1000):
        """Generate synthetic song dataset with realistic audio features"""
        np.random.seed(42)
        
        data = []
        for mood in self.mood_labels:
            n_mood_samples = n_samples // 4
            
            if mood == 'happy':
                # High energy, high valence, fast tempo
                tempo = np.random.normal(130, 20, n_mood_samples)
                energy = np.random.beta(6, 2, n_mood_samples)  # Skewed high
                valence = np.random.beta(6, 2, n_mood_samples)  # Skewed high
                loudness = np.random.normal(-5, 3, n_mood_samples)
                
            elif mood == 'chill':
                # Low energy, medium valence, slow tempo
                tempo = np.random.normal(80, 15, n_mood_samples)
                energy = np.random.beta(2, 6, n_mood_samples)  # Skewed low
                valence = np.random.beta(3, 3, n_mood_samples)  # Balanced
                loudness = np.random.normal(-12, 4, n_mood_samples)
                
            elif mood == 'sad':
                # Low energy, low valence, slow tempo
                tempo = np.random.normal(70, 15, n_mood_samples)
                energy = np.random.beta(2, 6, n_mood_samples)  # Skewed low
                valence = np.random.beta(2, 6, n_mood_samples)  # Skewed low
                loudness = np.random.normal(-15, 3, n_mood_samples)
                
            else:  # hyped
                # High energy, high valence, very fast tempo
                tempo = np.random.normal(150, 25, n_mood_samples)
                energy = np.random.beta(6, 2, n_mood_samples)  # Skewed high
                valence = np.random.beta(6, 2, n_mood_samples)  # Skewed high
                loudness = np.random.normal(-3, 2, n_mood_samples)
            
            # Additional features
            danceability = np.random.beta(3, 3, n_mood_samples)
            speechiness = np.random.beta(2, 8, n_mood_samples)  # Most songs have low speechiness
            acousticness = np.random.beta(2, 8, n_mood_samples)  # Most songs have low acousticness
            instrumentalness = np.random.beta(1, 9, n_mood_samples)  # Most songs have vocals
            liveness = np.random.beta(2, 8, n_mood_samples)  # Most songs are studio recordings
            
            for i in range(n_mood_samples):
                data.append({
                    'track_id': f"{mood}_{i}",
                    'track_name': f"Sample {mood.title()} Song {i}",
                    'artists': f"Artist {i}",
                    'tempo': max(50, min(200, tempo[i])),  # Clamp to realistic range
                    'energy': max(0, min(1, energy[i])),
                    'valence': max(0, min(1, valence[i])),
                    'loudness': max(-60, min(0, loudness[i])),
                    'danceability': max(0, min(1, danceability[i])),
                    'speechiness': max(0, min(1, speechiness[i])),
                    'acousticness': max(0, min(1, acousticness[i])),
                    'instrumentalness': max(0, min(1, instrumentalness[i])),
                    'liveness': max(0, min(1, liveness[i])),
                    'mood': mood
                })
        
        return pd.DataFrame(data)
    
    def generate_lyrics_features(self, df):
        """Generate synthetic lyrics features based on mood"""
        lyrics_features = []
        
        # Sample lyrics patterns for each mood
        mood_keywords = {
            'happy': ['happy', 'joy', 'smile', 'love', 'fun', 'dance', 'party', 'sunshine', 'bright'],
            'chill': ['calm', 'peace', 'quiet', 'relax', 'soft', 'gentle', 'smooth', 'easy', 'cool'],
            'sad': ['cry', 'tears', 'pain', 'hurt', 'lonely', 'dark', 'sad', 'broken', 'lost'],
            'hyped': ['energy', 'pump', 'fire', 'power', 'strong', 'fast', 'intense', 'wild', 'crazy']
        }
        
        for _, row in df.iterrows():
            mood = row['mood']
            keywords = mood_keywords[mood]
            
            # Generate synthetic text features
            word_count = np.random.randint(50, 300)
            positive_words = np.random.randint(5, 20) if mood in ['happy', 'hyped'] else np.random.randint(0, 5)
            negative_words = np.random.randint(5, 20) if mood == 'sad' else np.random.randint(0, 5)
            emotional_intensity = np.random.beta(6, 2) if mood in ['happy', 'hyped'] else np.random.beta(2, 6)
            
            lyrics_features.append({
                'word_count': word_count,
                'positive_sentiment': positive_words / word_count,
                'negative_sentiment': negative_words / word_count,
                'emotional_intensity': emotional_intensity,
                'keyword_density': np.random.beta(3, 7)  # How many mood keywords appear
            })
        
        return pd.DataFrame(lyrics_features)
    
    def create_baseline_rule_based(self, df):
        """Create rule-based baseline model as specified in project proposal"""
        predictions = []
        
        for _, row in df.iterrows():
            tempo = row['tempo']
            energy = row['energy']
            
            # Rule-based classification as per project proposal
            if tempo < 90 and energy < 0.5:
                pred = 'sad' if row['valence'] < 0.3 else 'chill'
            elif tempo > 120 and energy > 0.7:
                pred = 'hyped'
            else:
                pred = 'happy'  # Default fallback
            
            predictions.append(pred)
        
        return predictions
    
    def train_audio_model(self, X_train, y_train):
        """Train audio-based classification models"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        
        # Train Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        
        # Cross-validation to choose best model
        knn_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
        lr_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5)
        
        print(f"KNN CV Score: {knn_scores.mean():.3f} (+/- {knn_scores.std() * 2:.3f})")
        print(f"Logistic Regression CV Score: {lr_scores.mean():.3f} (+/- {lr_scores.std() * 2:.3f})")
        
        # Choose best model
        if knn_scores.mean() > lr_scores.mean():
            self.audio_model = knn
            print("Selected KNN as audio model")
        else:
            self.audio_model = lr
            print("Selected Logistic Regression as audio model")
    
    def train_lyrics_model(self, X_train, y_train):
        """Train lyrics-based classification model"""
        # Create separate scaler for lyrics
        lyrics_scaler = StandardScaler()
        X_train_scaled = lyrics_scaler.fit_transform(X_train)
        
        # Train Logistic Regression for lyrics
        self.lyrics_model = LogisticRegression(random_state=42, max_iter=1000)
        self.lyrics_model.fit(X_train_scaled, y_train)
        
        # Store lyrics scaler
        self.lyrics_scaler = lyrics_scaler
        
        # Cross-validation
        scores = cross_val_score(self.lyrics_model, X_train_scaled, y_train, cv=5)
        print(f"Lyrics Model CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    def predict_audio(self, X):
        """Predict mood using audio features"""
        X_scaled = self.scaler.transform(X)
        return self.audio_model.predict(X_scaled)
    
    def predict_lyrics(self, X):
        """Predict mood using lyrics features"""
        X_scaled = self.lyrics_scaler.transform(X)
        return self.lyrics_model.predict(X_scaled)
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model performance"""
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n{model_name} Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.mood_labels))
        
        return accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=self.mood_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.mood_labels, yticklabels=self.mood_labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names):
        """Plot feature importance for Logistic Regression"""
        if hasattr(self.audio_model, 'coef_'):
            importance = np.abs(self.audio_model.coef_[0])
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.title('Feature Importance - Audio Model')
            plt.xlabel('Importance (Absolute Coefficient)')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_mood_distribution(self, df):
        """Plot mood distribution and feature relationships"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Mood distribution
        df['mood'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Mood Distribution')
        axes[0,0].set_xlabel('Mood')
        axes[0,0].set_ylabel('Count')
        
        # Tempo vs Energy colored by mood
        for mood in self.mood_labels:
            mood_data = df[df['mood'] == mood]
            axes[0,1].scatter(mood_data['tempo'], mood_data['energy'], 
                             label=mood, alpha=0.6, s=30)
        axes[0,1].set_xlabel('Tempo (BPM)')
        axes[0,1].set_ylabel('Energy')
        axes[0,1].set_title('Tempo vs Energy by Mood')
        axes[0,1].legend()
        
        # Valence vs Loudness colored by mood
        for mood in self.mood_labels:
            mood_data = df[df['mood'] == mood]
            axes[1,0].scatter(mood_data['valence'], mood_data['loudness'], 
                             label=mood, alpha=0.6, s=30)
        axes[1,0].set_xlabel('Valence')
        axes[1,0].set_ylabel('Loudness (dB)')
        axes[1,0].set_title('Valence vs Loudness by Mood')
        axes[1,0].legend()
        
        # Energy distribution by mood
        df.boxplot(column='energy', by='mood', ax=axes[1,1])
        axes[1,1].set_title('Energy Distribution by Mood')
        axes[1,1].set_xlabel('Mood')
        axes[1,1].set_ylabel('Energy')
        
        plt.tight_layout()
        plt.savefig('mood_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main execution function"""
    print("🎵 Song Mood Classification System")
    print("=" * 50)
    
    # Initialize classifier
    classifier = SongMoodClassifier()
    
    # Generate synthetic dataset
    print("\n📊 Generating synthetic dataset...")
    df = classifier.generate_synthetic_dataset(1000)
    print(f"Dataset shape: {df.shape}")
    print(f"Mood distribution:\n{df['mood'].value_counts()}")
    
    # Generate lyrics features
    print("\n📝 Generating lyrics features...")
    lyrics_df = classifier.generate_lyrics_features(df)
    df_with_lyrics = pd.concat([df, lyrics_df], axis=1)
    
    # Save dataset
    df_with_lyrics.to_csv('song_mood_dataset.csv', index=False)
    print("Dataset saved to 'song_mood_dataset.csv'")
    
    # Visualize data
    print("\n📈 Creating visualizations...")
    classifier.plot_mood_distribution(df)
    
    # Prepare features
    audio_features = ['tempo', 'energy', 'valence', 'loudness', 'danceability', 
                     'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    lyrics_features = ['word_count', 'positive_sentiment', 'negative_sentiment', 
                      'emotional_intensity', 'keyword_density']
    
    X_audio = df[audio_features]
    X_lyrics = df_with_lyrics[lyrics_features]
    y = df['mood']
    
    # Split data
    X_audio_train, X_audio_test, X_lyrics_train, X_lyrics_test, y_train, y_test = train_test_split(
        X_audio, X_lyrics, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    print("\n🤖 Training audio model...")
    classifier.train_audio_model(X_audio_train, y_train)
    
    print("\n📝 Training lyrics model...")
    classifier.train_lyrics_model(X_lyrics_train, y_train)
    
    # Make predictions
    print("\n🔮 Making predictions...")
    audio_pred = classifier.predict_audio(X_audio_test)
    lyrics_pred = classifier.predict_lyrics(X_lyrics_test)
    
    # Rule-based baseline
    rule_pred = classifier.create_baseline_rule_based(df.iloc[y_test.index])
    
    # Evaluate models
    print("\n📊 Model Evaluation:")
    print("=" * 30)
    
    audio_accuracy = classifier.evaluate_model(y_test, audio_pred, "Audio Model")
    lyrics_accuracy = classifier.evaluate_model(y_test, lyrics_pred, "Lyrics Model")
    rule_accuracy = classifier.evaluate_model(y_test, rule_pred, "Rule-Based Baseline")
    
    # Plot confusion matrices
    classifier.plot_confusion_matrix(y_test, audio_pred, "Audio Model")
    classifier.plot_confusion_matrix(y_test, lyrics_pred, "Lyrics Model")
    classifier.plot_confusion_matrix(y_test, rule_pred, "Rule-Based Baseline")
    
    # Feature importance
    classifier.plot_feature_importance(audio_features)
    
    # Model comparison
    print("\n🏆 Model Comparison:")
    print(f"Rule-Based Baseline: {rule_accuracy:.3f}")
    print(f"Audio Model: {audio_accuracy:.3f}")
    print(f"Lyrics Model: {lyrics_accuracy:.3f}")
    
    # Confidence analysis
    print("\n🎯 Confidence Analysis:")
    audio_proba = classifier.audio_model.predict_proba(classifier.scaler.transform(X_audio_test))
    max_confidence = np.max(audio_proba, axis=1)
    
    print(f"Average confidence: {np.mean(max_confidence):.3f}")
    print(f"High confidence predictions (>0.8): {np.sum(max_confidence > 0.8)}/{len(max_confidence)}")
    
    print("\n✅ Song Mood Classification System Complete!")
    return classifier, df_with_lyrics

if __name__ == "__main__":
    classifier, dataset = main()
