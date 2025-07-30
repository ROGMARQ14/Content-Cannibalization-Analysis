# modules/analyzers/ml_scoring.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MLScoringEngine:
    """Machine learning-based dynamic scoring for cannibalization risk"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_type = 'gradient_boosting'  # or 'random_forest'
        
        # Enhanced default weights based on SEO best practices
        self.default_weights = {
            'title_similarity': 0.25,      # Reduced from 0.30
            'h1_similarity': 0.15,         # Reduced from 0.20
            'semantic_similarity': 0.20,   # Reduced from 0.35
            'keyword_overlap': 0.15,       # Same
            'intent_match': 0.10,          # New: AI-powered intent matching
            'serp_overlap': 0.15,          # New: Actual SERP competition
            # New performance-based features
            'traffic_difference': 0.05,    # Traffic disparity between pages
            'position_variance': 0.05,     # Position stability
            'ctr_ratio': 0.05             # CTR comparison
        }
        
        if model_path:
            self.load_model(model_path)
        else:
            self.initialize_default_model()
    
    def initialize_default_model(self):
        """Initialize with a pre-trained model or default weights"""
        logger.info("Initializing ML scoring engine with default configuration")
        
        # Create a synthetic training dataset based on expert knowledge
        synthetic_data = self._create_synthetic_training_data()
        
        # Train initial model
        self._train_model(synthetic_data['features'], synthetic_data['labels'])
    
    def _create_synthetic_training_data(self) -> Dict[str, np.ndarray]:
        """Create synthetic training data based on SEO expertise"""
        n_samples = 1000
        
        # Generate feature combinations
        features = []
        labels = []
        
        for _ in range(n_samples):
            # Generate random feature values
            title_sim = np.random.beta(2, 5)  # Skewed towards lower values
            h1_sim = np.random.beta(2, 5)
            semantic_sim = np.random.beta(3, 3)  # More balanced
            keyword_overlap = np.random.beta(2, 4)
            intent_match = np.random.choice([0, 1], p=[0.7, 0.3])
            serp_overlap = np.random.beta(1, 5)  # Rare but impactful
            traffic_diff = np.random.exponential(0.3)
            position_var = np.random.exponential(0.2)
            ctr_ratio = np.random.lognormal(0, 0.5)
            
            features.append([
                title_sim, h1_sim, semantic_sim, keyword_overlap,
                intent_match, serp_overlap, traffic_diff, 
                position_var, ctr_ratio
            ])
            
            # Calculate label based on expert rules
            risk_score = self._calculate_synthetic_risk(
                title_sim, h1_sim, semantic_sim, keyword_overlap,
                intent_match, serp_overlap, traffic_diff
            )
            labels.append(risk_score)
        
        return {
            'features': np.array(features),
            'labels': np.array(labels)
        }
    
    def _calculate_synthetic_risk(self, title_sim, h1_sim, semantic_sim, 
                                 keyword_overlap, intent_match, serp_overlap, 
                                 traffic_diff):
        """Calculate risk score based on expert rules"""
        # High risk scenarios
        if serp_overlap > 0.7 and intent_match == 1:
            return 0.9 + np.random.normal(0, 0.05)
        
        if title_sim > 0.8 and semantic_sim > 0.7 and intent_match == 1:
            return 0.85 + np.random.normal(0, 0.05)
        
        # Medium risk
        if (title_sim > 0.6 or semantic_sim > 0.6) and keyword_overlap > 0.4:
            return 0.5 + np.random.normal(0, 0.1)
        
        # Low risk
        base_risk = (
            title_sim * 0.25 + 
            h1_sim * 0.15 + 
            semantic_sim * 0.20 + 
            keyword_overlap * 0.15 +
            intent_match * 0.10 +
            serp_overlap * 0.15
        )
        
        # Adjust for traffic difference
        if traffic_diff > 1:  # Large traffic difference
            base_risk *= 0.8  # Lower risk if one page dominates
        
        return np.clip(base_risk + np.random.normal(0, 0.05), 0, 1)
    
    def _train_model(self, features: np.ndarray, labels: np.ndarray):
        """Train the ML model"""
        logger.info(f"Training {self.model_type} model with {len(features)} samples")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Initialize model based on type
        if self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        
        # Train model
        self.model.fit(features_scaled, labels)
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        # Evaluate model
        scores = cross_val_score(self.model, features_scaled, labels, cv=5)
        logger.info(f"Model cross-validation score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance"""
        feature_names = list(self.default_weights.keys())
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.feature_importance = dict(zip(feature_names, importances))
            
            # Log feature importance
            sorted_importance = sorted(self.feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True)
            logger.info("Feature importance:")
            for feature, importance in sorted_importance:
                logger.info(f"  {feature}: {importance:.3f}")
    
    def calculate_adaptive_score(self, features: Dict[str, float]) -> Tuple[float, Dict]:
        """Calculate cannibalization risk score with ML-optimized weights"""
        try:
            if self.model:
                # Prepare feature vector in correct order
                feature_vector = []
                for feature_name in self.default_weights.keys():
                    value = features.get(feature_name, 0)
                    feature_vector.append(value)
                
                # Scale features
                feature_array = np.array(feature_vector).reshape(1, -1)
                feature_scaled = self.scaler.transform(feature_array)
                
                # Predict risk score
                risk_score = self.model.predict(feature_scaled)[0]
                risk_score = np.clip(risk_score, 0, 1)
                
                # Get feature contributions
                contributions = self._calculate_feature_contributions(features)
                
                return risk_score, contributions
                
            else:
                # Fallback to weighted scoring
                return self._calculate_weighted_score(features)
                
        except Exception as e:
            logger.error(f"Error in adaptive scoring: {e}")
            return self._calculate_weighted_score(features)
    
    def _calculate_weighted_score(self, features: Dict[str, float]) -> Tuple[float, Dict]:
        """Fallback weighted scoring method"""
        score = 0
        contributions = {}
        
        for feature, value in features.items():
            weight = self.default_weights.get(feature, 0.1)
            contribution = weight * value
            score += contribution
            contributions[feature] = contribution
        
        return min(score, 1.0), contributions
    
    def _calculate_feature_contributions(self, features: Dict[str, float]) -> Dict:
        """Calculate how much each feature contributes to the final score"""
        contributions = {}
        
        if self.feature_importance:
            total_importance = sum(self.feature_importance.values())
            
            for feature, value in features.items():
                importance = self.feature_importance.get(feature, 0)
                contributions[feature] = (importance / total_importance) * value
        else:
            # Use default weights
            for feature, value in features.items():
                weight = self.default_weights.get(feature, 0.1)
                contributions[feature] = weight * value
        
        return contributions
    
    def update_with_feedback(self, 
                           cannibalization_pairs: List[Dict],
                           user_validations: List[bool]):
        """Update model based on user feedback"""
        logger.info(f"Updating model with {len(user_validations)} user validations")
        
        # Prepare training data from user feedback
        features = []
        labels = []
        
        for pair, is_valid in zip(cannibalization_pairs, user_validations):
            feature_vector = []
            for feature_name in self.default_weights.keys():
                value = pair.get(feature_name, 0)
                feature_vector.append(value)
            
            features.append(feature_vector)
            # Convert boolean validation to risk score
            labels.append(0.9 if is_valid else 0.1)
        
        if len(features) >= 10:  # Minimum samples for update
            # Combine with existing synthetic data
            synthetic_data = self._create_synthetic_training_data()
            
            all_features = np.vstack([synthetic_data['features'], np.array(features)])
            all_labels = np.hstack([synthetic_data['labels'], np.array(labels)])
            
            # Retrain model
            self._train_model(all_features, all_labels)
            
            logger.info("Model updated successfully with user feedback")
    
    def get_risk_category(self, risk_score: float) -> str:
        """Convert risk score to category"""
        if risk_score >= 0.7:
            return "High"
        elif risk_score >= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def save_model(self, path: str):
        """Save trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model and scaler"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data['feature_importance']
            self.model_type = model_data.get('model_type', 'gradient_boosting')
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.initialize_default_model()