#!/usr/bin/env python3
"""
Auto-Learning Module for Document Classification
Automatically collects new classification results and retrains the ML model
"""

import os
import json
import joblib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import threading
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import fitz  # PyMuPDF for text extraction

logger = logging.getLogger(__name__)

class AutoLearningSystem:
    """
    Automatic learning system that collects classification results and retrains the model
    """
    
    def __init__(self, model_dir: str = "model", learning_data_dir: str = "learning_data"):
        self.model_dir = model_dir
        self.learning_data_dir = learning_data_dir
        self.learning_data_file = os.path.join(learning_data_dir, "new_classifications.json")
        self.learning_threshold = 50  # Retrain after collecting 50 new samples
        self.min_confidence_threshold = 0.7  # Only learn from high-confidence predictions
        self.last_training_time = None
        self.training_interval_hours = 24  # Retrain at most once per day
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.learning_data_dir, exist_ok=True)
        
        # Initialize learning data file
        if not os.path.exists(self.learning_data_file):
            self._save_learning_data([])
        
        logger.info(f"Auto-learning system initialized. Learning threshold: {self.learning_threshold} samples")
    
    def add_classification_result(self, pdf_path: str, category: str, confidence: str, 
                                session_id: str, page_results: List[Dict] = None):
        """
        Add a new classification result to the learning dataset
        
        Args:
            pdf_path: Path to the classified PDF
            category: Predicted category
            confidence: Confidence score (e.g., "3/4", "4/4")
            session_id: Classification session ID
            page_results: Detailed page-level results
        """
        try:
            # Parse confidence score
            confidence_numeric = self._parse_confidence(confidence)
            
            # Only learn from high-confidence predictions
            if confidence_numeric < self.min_confidence_threshold:
                logger.debug(f"Skipping low-confidence result: {confidence} ({confidence_numeric:.2f})")
                return
            
            # Extract text from PDF
            text_content = self._extract_text_from_pdf(pdf_path)
            if not text_content or len(text_content.strip()) < 10:
                logger.debug(f"Skipping PDF with insufficient text content: {pdf_path}")
                return
            
            # Create learning sample
            learning_sample = {
                "text": text_content,
                "category": category,
                "confidence": confidence,
                "confidence_numeric": confidence_numeric,
                "session_id": session_id,
                "pdf_path": pdf_path,
                "timestamp": datetime.now().isoformat(),
                "page_results": page_results or []
            }
            
            # Add to learning data
            learning_data = self._load_learning_data()
            learning_data.append(learning_sample)
            self._save_learning_data(learning_data)
            
            logger.info(f"Added learning sample: {category} ({confidence}) - Total samples: {len(learning_data)}")
            
            # Check if we should retrain
            if len(learning_data) >= self.learning_threshold:
                self._trigger_retraining()
                
        except Exception as e:
            logger.error(f"Error adding classification result to learning data: {e}")
    
    def _parse_confidence(self, confidence_str: str) -> float:
        """Parse confidence string like '3/4' to numeric value"""
        try:
            if "/" in str(confidence_str):
                votes, total = str(confidence_str).split("/")
                return float(votes) / float(total)
            else:
                return float(confidence_str)
        except:
            return 0.0
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF for training"""
        try:
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found: {pdf_path}")
                return ""
            
            doc = fitz.open(pdf_path)
            text_content = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_content += page.get_text() + "\n"
            
            doc.close()
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    def _load_learning_data(self) -> List[Dict]:
        """Load learning data from file"""
        try:
            with open(self.learning_data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading learning data: {e}")
            return []
    
    def _save_learning_data(self, data: List[Dict]):
        """Save learning data to file"""
        try:
            with open(self.learning_data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")
    
    def _trigger_retraining(self):
        """Trigger model retraining in background thread"""
        # Check if we've trained recently
        if (self.last_training_time and 
            datetime.now() - self.last_training_time < timedelta(hours=self.training_interval_hours)):
            logger.info("Skipping retraining - trained recently")
            return
        
        # Start retraining in background thread
        training_thread = threading.Thread(target=self._retrain_model, daemon=True)
        training_thread.start()
        logger.info("Started background model retraining")
    
    def _retrain_model(self):
        """Retrain the ML model with new data"""
        try:
            logger.info("🤖 Starting automatic model retraining...")
            
            # Load original training data
            from training_dataset import datasets
            original_texts, original_labels = [], []
            for category, texts_list in datasets.items():
                original_texts.extend(texts_list)
                original_labels.extend([category] * len(texts_list))
            
            # Load new learning data
            learning_data = self._load_learning_data()
            new_texts = [sample["text"] for sample in learning_data]
            new_labels = [sample["category"] for sample in learning_data]
            
            # Combine datasets
            all_texts = original_texts + new_texts
            all_labels = original_labels + new_labels
            
            logger.info(f"Combined training data: {len(original_texts)} original + {len(new_texts)} new = {len(all_texts)} total")
            
            # Create and train new model
            vectorizer = TfidfVectorizer(
                max_features=15000,
                ngram_range=(1, 4),
                min_df=1,
                max_df=0.95,
                lowercase=True,
                stop_words=None,
                sublinear_tf=True,
                strip_accents='unicode'
            )
            
            X = vectorizer.fit_transform(all_texts)
            
            model = LogisticRegression(
                max_iter=5000,
                C=1.0,
                class_weight='balanced',
                random_state=42,
                solver='lbfgs'
            )
            
            model.fit(X, all_labels)
            
            # Evaluate new model
            cv_scores = cross_val_score(model, X, all_labels, cv=5, scoring='accuracy')
            training_accuracy = model.score(X, all_labels)
            
            logger.info(f"New model performance:")
            logger.info(f"  Training accuracy: {training_accuracy:.3f}")
            logger.info(f"  Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Save new model (backup old one first)
            self._backup_current_model()
            
            joblib.dump(model, os.path.join(self.model_dir, "classifier.pkl"))
            joblib.dump(vectorizer, os.path.join(self.model_dir, "vectorizer.pkl"))
            
            # Update model info
            model_info = {
                "training_samples": len(all_texts),
                "original_samples": len(original_texts),
                "learned_samples": len(new_texts),
                "categories": list(set(all_labels)),
                "feature_count": X.shape[1],
                "cv_accuracy": cv_scores.mean(),
                "training_accuracy": training_accuracy,
                "last_retrain_time": datetime.now().isoformat(),
                "vectorizer_params": vectorizer.get_params(),
                "model_params": model.get_params()
            }
            
            with open(os.path.join(self.model_dir, "model_info.json"), "w") as f:
                json.dump(model_info, f, indent=2, default=str)
            
            # Clear learning data after successful training
            self._save_learning_data([])
            self.last_training_time = datetime.now()
            
            logger.info(f"✅ Automatic model retraining completed successfully!")
            logger.info(f"📊 Model improved with {len(new_texts)} new samples")
            
        except Exception as e:
            logger.error(f"❌ Automatic model retraining failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _backup_current_model(self):
        """Backup current model before retraining"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(self.model_dir, "backups", timestamp)
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup model files
            for filename in ["classifier.pkl", "vectorizer.pkl", "model_info.json"]:
                src_path = os.path.join(self.model_dir, filename)
                if os.path.exists(src_path):
                    backup_path = os.path.join(backup_dir, filename)
                    import shutil
                    shutil.copy2(src_path, backup_path)
            
            logger.info(f"Model backed up to: {backup_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to backup model: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning system"""
        try:
            learning_data = self._load_learning_data()
            
            category_counts = {}
            for sample in learning_data:
                category = sample.get("category", "Unknown")
                category_counts[category] = category_counts.get(category, 0) + 1
            
            return {
                "total_samples": len(learning_data),
                "threshold": self.learning_threshold,
                "samples_until_retrain": max(0, self.learning_threshold - len(learning_data)),
                "category_distribution": category_counts,
                "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
                "min_confidence_threshold": self.min_confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error getting learning stats: {e}")
            return {"error": str(e)}

# Global auto-learning instance
auto_learner = None

def initialize_auto_learning(model_dir: str = "model", learning_data_dir: str = "learning_data"):
    """Initialize the global auto-learning system"""
    global auto_learner
    auto_learner = AutoLearningSystem(model_dir, learning_data_dir)
    return auto_learner

def add_classification_for_learning(pdf_path: str, category: str, confidence: str, 
                                  session_id: str, page_results: List[Dict] = None):
    """Add classification result to learning system"""
    if auto_learner:
        auto_learner.add_classification_result(pdf_path, category, confidence, session_id, page_results)
    else:
        logger.warning("Auto-learning system not initialized")

def get_auto_learning_stats() -> Dict[str, Any]:
    """Get auto-learning statistics"""
    if auto_learner:
        return auto_learner.get_learning_stats()
    else:
        return {"error": "Auto-learning system not initialized"}