"""
Comparison: Real Spotify Data vs Synthetic Data
Demonstrates the accuracy differences and trade-offs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class DataComparison:
    def __init__(self):
        self.mood_labels = ['happy', 'chill', 'sad', 'hyped']
    
    def create_realistic_spotify_data(self, n_samples=1000):
        """Simulate what real Spotify data would look like"""
        np.random.seed(42)
        data = []
        
        # Real Spotify data has more noise and overlap between categories
        for mood in self.mood_labels:
            n_mood_samples = n_samples // 4
            
            if mood == 'happy':
                # Real data: More variation, some happy songs are slow
                tempo = np.random.normal(120, 30, n_mood_samples)  # More spread
                energy = np.random.beta(4, 3, n_mood_samples)  # Less skewed
                valence = np.random.beta(5, 3, n_mood_samples)  # Less skewed
                loudness = np.random.normal(-8, 5, n_mood_samples)  # More variation
                
            elif mood == 'chill':
                # Real data: Some chill songs can be energetic
                tempo = np.random.normal(90, 25, n_mood_samples)  # More overlap
                energy = np.random.beta(3, 4, n_mood_samples)  # Less clear separation
                valence = np.random.beta(4, 4, n_mood_samples)  # More balanced
                loudness = np.random.normal(-10, 6, n_mood_samples)
                
            elif mood == 'sad':
                # Real data: Some sad songs are fast (sad but energetic)
                tempo = np.random.normal(85, 20, n_mood_samples)  # Less clear separation
                energy = np.random.beta(3, 5, n_mood_samples)  # Less clear separation
                valence = np.random.beta(2, 5, n_mood_samples)  # Less clear separation
                loudness = np.random.normal(-12, 4, n_mood_samples)
                
            else:  # hyped
                # Real data: Some hyped songs are slower
                tempo = np.random.normal(140, 35, n_mood_samples)  # More overlap
                energy = np.random.beta(5, 3, n_mood_samples)  # Less clear separation
                valence = np.random.beta(5, 3, n_mood_samples)  # Less clear separation
                loudness = np.random.normal(-6, 4, n_mood_samples)
            
            # Add realistic noise and outliers
            for i in range(n_mood_samples):
                # 10% chance of outlier (song that doesn't fit typical pattern)
                if np.random.random() < 0.1:
                    tempo[i] = np.random.normal(100, 20)  # Random tempo
                    energy[i] = np.random.beta(3, 3)  # Random energy
                    valence[i] = np.random.beta(3, 3)  # Random valence
                
                data.append({
                    'track_id': f"spotify_{mood}_{i}",
                    'track_name': f"Real {mood.title()} Song {i}",
                    'artists': f"Real Artist {i}",
                    'tempo': max(50, min(200, tempo[i])),
                    'energy': max(0, min(1, energy[i])),
                    'valence': max(0, min(1, valence[i])),
                    'loudness': max(-60, min(0, loudness[i])),
                    'danceability': np.random.beta(3, 3),
                    'speechiness': np.random.beta(2, 8),
                    'acousticness': np.random.beta(2, 8),
                    'instrumentalness': np.random.beta(1, 9),
                    'liveness': np.random.beta(2, 8),
                    'mood': mood,
                    'data_type': 'realistic_spotify'
                })
        
        return pd.DataFrame(data)
    
    def create_clean_synthetic_data(self, n_samples=1000):
        """Our current synthetic data (very clean patterns)"""
        np.random.seed(42)
        data = []
        
        for mood in self.mood_labels:
            n_mood_samples = n_samples // 4
            
            if mood == 'happy':
                # Very clean patterns
                tempo = np.random.normal(130, 20, n_mood_samples)
                energy = np.random.beta(6, 2, n_mood_samples)  # Very skewed high
                valence = np.random.beta(6, 2, n_mood_samples)  # Very skewed high
                loudness = np.random.normal(-5, 3, n_mood_samples)
                
            elif mood == 'chill':
                tempo = np.random.normal(80, 15, n_mood_samples)
                energy = np.random.beta(2, 6, n_mood_samples)  # Very skewed low
                valence = np.random.beta(3, 3, n_mood_samples)
                loudness = np.random.normal(-12, 4, n_mood_samples)
                
            elif mood == 'sad':
                tempo = np.random.normal(70, 15, n_mood_samples)
                energy = np.random.beta(2, 6, n_mood_samples)  # Very skewed low
                valence = np.random.beta(2, 6, n_mood_samples)  # Very skewed low
                loudness = np.random.normal(-15, 3, n_mood_samples)
                
            else:  # hyped
                tempo = np.random.normal(150, 25, n_mood_samples)
                energy = np.random.beta(6, 2, n_mood_samples)  # Very skewed high
                valence = np.random.beta(6, 2, n_mood_samples)  # Very skewed high
                loudness = np.random.normal(-3, 2, n_mood_samples)
            
            for i in range(n_mood_samples):
                data.append({
                    'track_id': f"synthetic_{mood}_{i}",
                    'track_name': f"Synthetic {mood.title()} Song {i}",
                    'artists': f"Synthetic Artist {i}",
                    'tempo': max(50, min(200, tempo[i])),
                    'energy': max(0, min(1, energy[i])),
                    'valence': max(0, min(1, valence[i])),
                    'loudness': max(-60, min(0, loudness[i])),
                    'danceability': np.random.beta(3, 3),
                    'speechiness': np.random.beta(2, 8),
                    'acousticness': np.random.beta(2, 8),
                    'instrumentalness': np.random.beta(1, 9),
                    'liveness': np.random.beta(2, 8),
                    'mood': mood,
                    'data_type': 'clean_synthetic'
                })
        
        return pd.DataFrame(data)
    
    def compare_datasets(self):
        """Compare the two datasets"""
        print("🔄 Creating realistic Spotify data...")
        spotify_data = self.create_realistic_spotify_data(1000)
        
        print("🔄 Creating clean synthetic data...")
        synthetic_data = self.create_clean_synthetic_data(1000)
        
        # Combine for comparison
        combined_data = pd.concat([spotify_data, synthetic_data], ignore_index=True)
        
        # Prepare features
        features = ['tempo', 'energy', 'valence', 'loudness', 'danceability', 
                   'speechiness', 'acousticness', 'instrumentalness', 'liveness']
        
        X = combined_data[features]
        y = combined_data['mood']
        data_type = combined_data['data_type']
        
        # Split data
        X_train, X_test, y_train, y_test, type_train, type_test = train_test_split(
            X, y, data_type, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train separate models
        print("\n🤖 Training models...")
        
        # Model on realistic data
        realistic_mask = (type_train == 'realistic_spotify')
        if np.any(realistic_mask):
            X_realistic = X_train[realistic_mask]
            y_realistic = y_train[realistic_mask]
            
            realistic_model = RandomForestClassifier(n_estimators=100, random_state=42)
            realistic_model.fit(X_realistic, y_realistic)
            
            # Test on realistic data
            realistic_test_mask = (type_test == 'realistic_spotify')
            if np.any(realistic_test_mask):
                X_realistic_test = X_test[realistic_test_mask]
                y_realistic_test = y_test[realistic_test_mask]
                
                realistic_pred = realistic_model.predict(X_realistic_test)
                realistic_accuracy = accuracy_score(y_realistic_test, realistic_pred)
            else:
                realistic_accuracy = 0
        else:
            realistic_accuracy = 0
        
        # Model on clean data
        clean_mask = (type_train == 'clean_synthetic')
        if np.any(clean_mask):
            X_clean = X_train[clean_mask]
            y_clean = y_train[clean_mask]
            
            clean_model = RandomForestClassifier(n_estimators=100, random_state=42)
            clean_model.fit(X_clean, y_clean)
            
            # Test on clean data
            clean_test_mask = (type_test == 'clean_synthetic')
            if np.any(clean_test_mask):
                X_clean_test = X_test[clean_test_mask]
                y_clean_test = y_test[clean_test_mask]
                
                clean_pred = clean_model.predict(X_clean_test)
                clean_accuracy = accuracy_score(y_clean_test, clean_pred)
            else:
                clean_accuracy = 0
        else:
            clean_accuracy = 0
        
        # Cross-contamination test
        print("\n🧪 Cross-contamination test...")
        if np.any(realistic_mask) and np.any(clean_test_mask):
            realistic_on_clean = realistic_model.predict(X_clean_test)
            realistic_on_clean_accuracy = accuracy_score(y_clean_test, realistic_on_clean)
        else:
            realistic_on_clean_accuracy = 0
            
        if np.any(clean_mask) and np.any(realistic_test_mask):
            clean_on_realistic = clean_model.predict(X_realistic_test)
            clean_on_realistic_accuracy = accuracy_score(y_realistic_test, clean_on_realistic)
        else:
            clean_on_realistic_accuracy = 0
        
        # Results
        print("\n📊 ACCURACY COMPARISON:")
        print("=" * 50)
        print(f"Realistic Spotify Data:     {realistic_accuracy:.3f}")
        print(f"Clean Synthetic Data:      {clean_accuracy:.3f}")
        print(f"Realistic → Clean:         {realistic_on_clean_accuracy:.3f}")
        print(f"Clean → Realistic:         {clean_on_realistic_accuracy:.3f}")
        
        # Create visualization
        self.plot_comparison(spotify_data, synthetic_data)
        
        return {
            'realistic_accuracy': realistic_accuracy,
            'clean_accuracy': clean_accuracy,
            'cross_contamination': {
                'realistic_on_clean': realistic_on_clean_accuracy,
                'clean_on_realistic': clean_on_realistic_accuracy
            }
        }
    
    def plot_comparison(self, spotify_data, synthetic_data):
        """Plot comparison between datasets"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Tempo vs Energy comparison
        for i, (data, title) in enumerate([(spotify_data, 'Realistic Spotify Data'), 
                                         (synthetic_data, 'Clean Synthetic Data')]):
            for mood in self.mood_labels:
                mood_data = data[data['mood'] == mood]
                if len(mood_data) > 0:
                    axes[0, i].scatter(mood_data['tempo'], mood_data['energy'], 
                                     label=mood, alpha=0.6, s=20)
            axes[0, i].set_xlabel('Tempo (BPM)')
            axes[0, i].set_ylabel('Energy')
            axes[0, i].set_title(f'{title}\nTempo vs Energy')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
        
        # Valence vs Loudness comparison
        for i, (data, title) in enumerate([(spotify_data, 'Realistic Spotify Data'), 
                                         (synthetic_data, 'Clean Synthetic Data')]):
            for mood in self.mood_labels:
                mood_data = data[data['mood'] == mood]
                if len(mood_data) > 0:
                    axes[1, i].scatter(mood_data['valence'], mood_data['loudness'], 
                                     label=mood, alpha=0.6, s=20)
            axes[1, i].set_xlabel('Valence')
            axes[1, i].set_ylabel('Loudness (dB)')
            axes[1, i].set_title(f'{title}\nValence vs Loudness')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        # Feature distribution comparison
        feature = 'energy'
        axes[0, 2].hist(spotify_data[feature], alpha=0.7, bins=20, label='Realistic Spotify')
        axes[0, 2].hist(synthetic_data[feature], alpha=0.7, bins=20, label='Clean Synthetic')
        axes[0, 2].set_xlabel(feature.title())
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title(f'{feature.title()} Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Hide unused subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('data_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run the comparison"""
    print("🔍 Data Quality Comparison: Realistic vs Clean Synthetic")
    print("=" * 60)
    
    comparison = DataComparison()
    results = comparison.compare_datasets()
    
    print("\n🎯 KEY INSIGHTS:")
    print("=" * 30)
    print("✅ Clean synthetic data has higher accuracy (but less realistic)")
    print("❌ Realistic data has lower accuracy (but more representative)")
    print("🔄 Cross-contamination shows generalization issues")
    print("📊 Trade-off: Accuracy vs Realism")

if __name__ == "__main__":
    main()
