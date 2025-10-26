"""
Milestone 1: Song Mood Classification System
Using Kaggle Dataset for Training and Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MoodClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.mood_labels = ['happy', 'chill', 'sad', 'hyped']
        
    def load_music_dataset(self):
        """Load music dataset"""
        print("Loading music dataset...")
        
        try:
            # Try to load preprocessed dataset
            df = pd.read_csv('music_dataset.csv')
            print(f"Loaded {len(df)} songs from preprocessed dataset")
            print(f"Mood distribution:")
            print(df['mood'].value_counts())
            return df
            
        except FileNotFoundError:
            try:
                # Try loading the songs dataset
                print("Preprocessed dataset not found. Loading songs.csv...")
                df = pd.read_csv('songs.csv')
                print(f"Loaded {len(df)} songs from songs.csv")
                
                # Preprocess (moods already labeled!)
                df = self.preprocess_songs_dataset(df)
                
                # Save preprocessed dataset
                df.to_csv('music_dataset.csv', index=False)
                print(f"Preprocessed and saved {len(df)} songs to 'music_dataset.csv'")
                print(f"Mood distribution:")
                print(df['mood'].value_counts())
                return df
                
            except FileNotFoundError:
                print("songs.csv not found. Creating fallback dataset...")
                return self.create_fallback_dataset()
            except Exception as e:
                print(f"Error loading songs.csv: {e}")
                print("Creating fallback dataset...")
                return self.create_fallback_dataset()
    
    def create_fallback_dataset(self):
        """Create fallback dataset if real data not available"""
        print("Creating fallback dataset...")
        np.random.seed(42)
        
        data = []
        for mood in self.mood_labels:
            n_mood_samples = 200
            
            if mood == 'happy':
                tempo = np.random.normal(125, 30, n_mood_samples)
                energy = np.random.beta(4, 3, n_mood_samples)
                valence = np.random.beta(5, 3, n_mood_samples)
                loudness = np.random.normal(-8, 5, n_mood_samples)
                
            elif mood == 'chill':
                tempo = np.random.normal(95, 25, n_mood_samples)
                energy = np.random.beta(3, 4, n_mood_samples)
                valence = np.random.beta(4, 4, n_mood_samples)
                loudness = np.random.normal(-11, 4, n_mood_samples)
                
            elif mood == 'sad':
                tempo = np.random.normal(85, 20, n_mood_samples)
                energy = np.random.beta(3, 5, n_mood_samples)
                valence = np.random.beta(2, 5, n_mood_samples)
                loudness = np.random.normal(-13, 3, n_mood_samples)
                
            else:  # hyped
                tempo = np.random.normal(145, 35, n_mood_samples)
                energy = np.random.beta(5, 3, n_mood_samples)
                valence = np.random.beta(5, 3, n_mood_samples)
                loudness = np.random.normal(-6, 3, n_mood_samples)
            
            for i in range(n_mood_samples):
                data.append({
                    'track_id': f"fallback_{mood}_{i}",
                    'track_name': f"Fallback {mood.title()} Song {i}",
                    'artists': f"Fallback Artist {i}",
                    'tempo': max(50, min(200, tempo[i])),
                    'energy': max(0, min(1, energy[i])),
                    'valence': max(0, min(1, valence[i])),
                    'loudness': max(-60, min(0, loudness[i])),
                    'danceability': np.random.beta(3, 3),
                    'speechiness': np.random.beta(2, 8),
                    'acousticness': np.random.beta(2, 8),
                    'instrumentalness': np.random.beta(1, 9),
                    'liveness': np.random.beta(2, 8),
                    'mood': mood
                })
        
        df = pd.DataFrame(data)
        print(f"Created fallback dataset with {len(df)} songs")
        return df
    
    def preprocess_songs_dataset(self, df):
        """Preprocess songs.csv dataset (moods already labeled!)"""
        print("Preprocessing songs.csv dataset...")
        print(f"Original dataset size: {len(df)} songs")
        
        # Expected columns in songs.csv
        # artist, track_name, text, length, mood, genre, tempo, loudness, energy, valence, release_year
        
        # Audio features that exist in this dataset
        required_features = ['tempo', 'loudness', 'energy', 'valence']
        
        # Check for mood column
        if 'mood' not in df.columns:
            print("ERROR: 'mood' column not found! Using fallback dataset instead.")
            return self.create_fallback_dataset()
        
        # Show original mood distribution
        print(f"\nOriginal mood distribution:")
        print(df['mood'].value_counts())
        
        # Filter to only include our 4 mood categories (case-insensitive)
        df['mood'] = df['mood'].str.lower().str.strip()
        df = df[df['mood'].isin(self.mood_labels)]
        print(f"\nAfter filtering to 4 moods: {len(df)} songs")
        
        # Check for required features
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f"ERROR: Missing required features: {missing_features}")
            print("Using fallback dataset instead.")
            return self.create_fallback_dataset()
        
        # Remove rows with missing values in required features
        original_len = len(df)
        df = df.dropna(subset=required_features + ['mood'])
        if len(df) < original_len:
            print(f"Removed {original_len - len(df)} rows with missing values")
        
        # FIX: Normalize features that might be on wrong scale
        print("\n=== Checking feature scales ===")
        print(f"Raw feature ranges BEFORE normalization:")
        print(f"  Tempo: {df['tempo'].min():.1f} - {df['tempo'].max():.1f}")
        print(f"  Loudness: {df['loudness'].min():.1f} - {df['loudness'].max():.1f}")
        print(f"  Energy: {df['energy'].min():.1f} - {df['energy'].max():.1f}")
        print(f"  Valence: {df['valence'].min():.1f} - {df['valence'].max():.1f}")
        
        # Convert to numeric (in case they're strings)
        df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
        df['valence'] = pd.to_numeric(df['valence'], errors='coerce')
        
        # Check for all-zero energy (data quality issue)
        if df['energy'].max() == 0 and df['energy'].min() == 0:
            print("ERROR: All energy values are 0! This indicates bad data.")
            print("This will cause perfect classification scores (overfitting).")
            print("Please check your source CSV file for correct energy values.")
        
        # Normalize if needed
        if df['energy'].max() > 1.5:
            print(f"Normalizing energy from 0-100 to 0-1 scale...")
            df['energy'] = df['energy'] / 100.0
        
        if df['valence'].max() > 1.5:
            print(f"Normalizing valence from 0-100 to 0-1 scale...")
            df['valence'] = df['valence'] / 100.0
        
        print(f"\nFinal feature ranges AFTER normalization:")
        print(f"  Tempo: {df['tempo'].min():.1f} - {df['tempo'].max():.1f}")
        print(f"  Loudness: {df['loudness'].min():.1f} - {df['loudness'].max():.1f}")
        print(f"  Energy: {df['energy'].min():.3f} - {df['energy'].max():.3f}")
        print(f"  Valence: {df['valence'].min():.3f} - {df['valence'].max():.3f}")
        
        # Show mood distribution before balancing
        print(f"\nMood distribution (before balancing):")
        print(df['mood'].value_counts())
        
        # Balance dataset - take equal samples from each mood
        min_samples = df['mood'].value_counts().min()
        samples_per_mood = min(min_samples, 500)  # Cap at 500 per mood
        
        print(f"\nBalancing dataset: {samples_per_mood} songs per mood...")
        df = df.groupby('mood', group_keys=False).apply(
            lambda x: x.sample(n=samples_per_mood, random_state=42)
        )
        
        # Keep relevant columns (including text for future use!)
        keep_columns = ['track_name', 'artist'] + required_features + ['mood']
        
        # Add text column if available (for future lyrics analysis)
        if 'text' in df.columns:
            keep_columns.insert(2, 'text')  # Add after track_name and artist
            print("✓ Keeping 'text' column for future lyrics analysis")
        
        # Keep genre if available (useful metadata)
        if 'genre' in df.columns:
            keep_columns.append('genre')
        
        available_columns = [col for col in keep_columns if col in df.columns]
        df = df[available_columns]
        
        print(f"\nPreprocessing complete: {len(df)} songs ready for training")
        print(f"Final mood distribution:")
        print(df['mood'].value_counts())
        
        return df
    
    def preprocess_kaggle_data(self, df):
        """Preprocess Kaggle dataset and assign moods"""
        print("Preprocessing Kaggle data...")
        
        # Required features for our model
        required_features = ['tempo', 'energy', 'valence', 'loudness', 'danceability', 
                           'speechiness', 'acousticness', 'instrumentalness', 'liveness']
        
        # Check if all features exist
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for feature in missing_features:
                df[feature] = 0.5  # Default value
        
        # Remove rows with missing values in key features
        df = df.dropna(subset=required_features)
        
        # Assign moods based on audio features
        print("Assigning moods based on audio features...")
        df['mood'] = df.apply(self.assign_mood_from_features, axis=1)
        
        # Show mood distribution before balancing
        print(f"Mood distribution (before balancing):")
        print(df['mood'].value_counts())
        
        # Balance dataset - take equal samples from each mood
        min_samples = df['mood'].value_counts().min()
        samples_per_mood = min(min_samples, 400)  # Cap at 400 per mood
        
        print(f"Balancing dataset: {samples_per_mood} songs per mood...")
        df = df.groupby('mood', group_keys=False).apply(
            lambda x: x.sample(n=samples_per_mood, random_state=42)
        )
        
        # Keep relevant columns
        keep_columns = ['track_name', 'track_artist'] + required_features + ['mood']
        available_columns = [col for col in keep_columns if col in df.columns]
        df = df[available_columns]
        
        print(f"Preprocessing complete: {len(df)} songs ready for training")
        return df
    
    def assign_mood_from_features(self, row):
        """Assign mood category based on audio features"""
        tempo = row['tempo']
        energy = row['energy']
        valence = row['valence']
        
        # Rule-based mood assignment
        # Hyped: Fast tempo + high energy
        if tempo > 120 and energy > 0.7:
            return 'hyped'
        
        # Happy: High valence + moderate/high energy
        elif valence > 0.6 and energy > 0.5:
            return 'happy'
        
        # Sad: Low valence OR (low energy + low valence)
        elif valence < 0.4 or (energy < 0.5 and valence < 0.5):
            return 'sad'
        
        # Chill: Everything else (moderate values)
        else:
            return 'chill'
    
    def train_models(self, df):
        """Train and compare multiple models"""
        print("\nTraining models on dataset...")
        
        # Prepare features - use only what's available in the dataset
        all_possible_features = ['tempo', 'energy', 'valence', 'loudness', 'danceability', 
                                'speechiness', 'acousticness', 'instrumentalness', 'liveness']
        
        # Detect which features are actually in the dataset
        features = [f for f in all_possible_features if f in df.columns]
        print(f"Using {len(features)} features: {features}")
        
        X = df[features]
        y = df['mood']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        results = {}
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            # Cross-validation
            scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            mean_score = scores.mean()
            std_score = scores.std()
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            
            # Test performance
            y_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'cv_score': mean_score,
                'cv_std': std_score,
                'test_accuracy': test_accuracy,
                'model': model
            }
            
            print(f"{name:20s}: CV={mean_score:.3f}±{std_score:.3f}, Test={test_accuracy:.3f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
        
        # Select best model
        self.model = best_model
        self.feature_names = features  # Store feature names for later use
        best_name = max(results.keys(), key=lambda k: results[k]['cv_score'])
        
        print(f"\nBest Model: {best_name}")
        print(f"CV Score: {results[best_name]['cv_score']:.3f} ± {results[best_name]['cv_std']:.3f}")
        print(f"Test Accuracy: {results[best_name]['test_accuracy']:.3f}")
        
        # Detailed evaluation
        y_pred = best_model.predict(X_test_scaled)
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.mood_labels))
        
        return results, X_test, y_test, y_pred
    
    def create_visualizations(self, df, X_test, y_test, y_pred):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Song Mood Classification Analysis', fontsize=16, fontweight='bold')
        
        # 1. Mood distribution
        mood_counts = df['mood'].value_counts()
        axes[0,0].bar(mood_counts.index, mood_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0,0].set_title('Mood Distribution in Dataset')
        axes[0,0].set_xlabel('Mood')
        axes[0,0].set_ylabel('Number of Songs')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Tempo vs Energy by mood
        for mood in self.mood_labels:
            mood_data = df[df['mood'] == mood]
            axes[0,1].scatter(mood_data['tempo'], mood_data['energy'], 
                             label=mood, alpha=0.6, s=30)
        axes[0,1].set_xlabel('Tempo (BPM)')
        axes[0,1].set_ylabel('Energy')
        axes[0,1].set_title('Tempo vs Energy by Mood')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Valence vs Loudness by mood
        for mood in self.mood_labels:
            mood_data = df[df['mood'] == mood]
            axes[0,2].scatter(mood_data['valence'], mood_data['loudness'], 
                             label=mood, alpha=0.6, s=30)
        axes[0,2].set_xlabel('Valence')
        axes[0,2].set_ylabel('Loudness (dB)')
        axes[0,2].set_title('Valence vs Loudness by Mood')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.mood_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.mood_labels, yticklabels=self.mood_labels, ax=axes[1,0])
        axes[1,0].set_title('Confusion Matrix')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('Actual')
        
        # 5. Feature importance (if Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            axes[1,1].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1,1].set_title('Feature Importance')
            axes[1,1].set_xlabel('Importance')
        else:
            axes[1,1].text(0.5, 0.5, 'Feature importance\nnot available for\nthis model type', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Feature Importance')
        
        # 6. Model performance by mood
        mood_accuracies = []
        for mood in self.mood_labels:
            mood_mask = (y_test == mood)
            if np.any(mood_mask):
                mood_accuracy = np.mean(y_pred[mood_mask] == mood)
                mood_accuracies.append(mood_accuracy)
            else:
                mood_accuracies.append(0)
        
        axes[1,2].bar(self.mood_labels, mood_accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1,2].set_title('Model Performance by Mood')
        axes[1,2].set_xlabel('Mood')
        axes[1,2].set_ylabel('Accuracy')
        axes[1,2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('milestone1_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'milestone1_analysis.png'")
    
    def predict_new_song(self, tempo, energy, valence, loudness, danceability=None, 
                        speechiness=None, acousticness=None, instrumentalness=None, liveness=None):
        """Predict mood for a new song (only required features: tempo, energy, valence, loudness)"""
        if self.model is None:
            return "Model not trained yet!"
        
        # Create feature dictionary with all values
        all_features = {
            'tempo': tempo,
            'energy': energy,
            'valence': valence,
            'loudness': loudness,
            'danceability': danceability if danceability is not None else 0.5,
            'speechiness': speechiness if speechiness is not None else 0.1,
            'acousticness': acousticness if acousticness is not None else 0.1,
            'instrumentalness': instrumentalness if instrumentalness is not None else 0.1,
            'liveness': liveness if liveness is not None else 0.1
        }
        
        # Create feature vector with only the features used in training
        feature_values = [all_features[f] for f in self.feature_names]
        features = np.array([feature_values])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        confidence = np.max(self.model.predict_proba(features_scaled))
        
        return prediction, confidence
    
    def save_model(self, filename='milestone1_model.pkl'):
        """Save the trained model"""
        import joblib
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'mood_labels': self.mood_labels,
            'feature_names': self.feature_names
        }, filename)
        print(f"Model saved as '{filename}'")
    
    def load_model(self, filename='milestone1_model.pkl'):
        """Load a trained model"""
        import joblib
        data = joblib.load(filename)
        self.model = data['model']
        self.scaler = data['scaler']
        self.mood_labels = data['mood_labels']
        self.feature_names = data.get('feature_names', ['tempo', 'energy', 'valence', 'loudness'])  # Default to 4 features
        print(f"Model loaded from '{filename}'")

def main():
    """Main execution for Milestone 1"""
    print("Milestone 1: Song Mood Classification System")
    print("=" * 60)
    print("Using Dataset for Training and Evaluation")
    print("Goal: Basic end-to-end classifier working on labeled dataset")
    print()
    
    # Initialize classifier
    classifier = MoodClassifier()
    
    # Load dataset (will automatically preprocess and save if needed)
    df = classifier.load_music_dataset()
    
    # Train models
    results, X_test, y_test, y_pred = classifier.train_models(df)
    
    # Create visualizations
    classifier.create_visualizations(df, X_test, y_test, y_pred)
    
    # Save model
    classifier.save_model()
    
    # Demonstrate prediction
    print("\nTesting prediction on new song:")
    print("Example: Tempo=120, Energy=0.8, Valence=0.7, Loudness=-5")
    prediction, confidence = classifier.predict_new_song(120, 0.8, 0.7, -5)
    print(f"Predicted mood: {prediction} (confidence: {confidence:.3f})")
    
    print("\nMilestone 1 Complete!")
    print("Ready for team collaboration and GitHub upload")
    
    return classifier, df

if __name__ == "__main__":
    classifier, dataset = main()
