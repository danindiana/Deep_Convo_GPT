"""
Neural Network Arousal Classifier

Machine learning model for classifying arousal states based on
physiological and hormonal features.

Uses:
- Feedforward neural network for classification
- Multi-layer perceptron (MLP)
- Feature engineering from physiological signals

Classes:
- Baseline (0): Resting state
- Desire (1): Early arousal
- Arousal (2): Active arousal
- Orgasm (3): Climax
- Resolution (4): Post-orgasmic

References:
- Chivers, M. L., et al. (2010). Agreement of self-reported and genital measures.
- Pattern recognition in psychophysiology

Author: Deep Convo GPT Research Team
Date: 2025-11-19
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import seaborn as sns
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class ArousalStateClassifier:
    """
    Neural network classifier for arousal states.

    Features:
    - Heart rate
    - Blood pressure
    - Skin conductance
    - Genital blood flow
    - Pupil dilation
    - Respiration rate
    - Hormone levels (testosterone, oxytocin, prolactin, cortisol)

    States:
    0 - Baseline
    1 - Desire
    2 - Arousal
    3 - Orgasm
    4 - Resolution
    """

    def __init__(
        self,
        hidden_layers: tuple = (64, 32, 16),
        activation: str = 'relu',
        learning_rate: float = 0.001,
        max_iterations: int = 500,
        random_state: int = 42
    ):
        """
        Initialize neural network classifier.

        Parameters
        ----------
        hidden_layers : tuple
            Size of each hidden layer
        activation : str
            Activation function ('relu', 'tanh', 'logistic')
        learning_rate : float
            Learning rate for gradient descent
        max_iterations : int
            Maximum training iterations
        random_state : int
            Random seed for reproducibility
        """
        self.classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver='adam',
            learning_rate_init=learning_rate,
            max_iter=max_iterations,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )

        self.scaler = StandardScaler()
        self.is_trained = False

        self.state_names = [
            'Baseline',
            'Desire',
            'Arousal',
            'Orgasm',
            'Resolution'
        ]

        self.feature_names = [
            'Heart Rate (bpm)',
            'Systolic BP (mmHg)',
            'Diastolic BP (mmHg)',
            'Skin Conductance (μS)',
            'Genital Blood Flow (index)',
            'Pupil Dilation (mm)',
            'Respiration Rate (br/min)',
            'Testosterone (nmol/L)',
            'Oxytocin (pg/mL)',
            'Prolactin (ng/mL)',
            'Cortisol (μg/dL)'
        ]

    def generate_synthetic_data(
        self,
        n_samples: int = 1000,
        noise_level: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data.

        Creates realistic physiological patterns for each arousal state.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        noise_level : float
            Amount of noise to add (0-1)

        Returns
        -------
        X : ndarray (n_samples, n_features)
            Feature matrix
        y : ndarray (n_samples,)
            State labels
        """
        samples_per_state = n_samples // 5
        X_list = []
        y_list = []

        # State 0: Baseline
        X_baseline = np.array([
            np.random.normal(70, 5, samples_per_state),   # HR
            np.random.normal(120, 10, samples_per_state), # Sys BP
            np.random.normal(80, 5, samples_per_state),   # Dia BP
            np.random.normal(5, 1, samples_per_state),    # Skin cond
            np.random.normal(0.3, 0.1, samples_per_state),# Genital
            np.random.normal(3, 0.3, samples_per_state),  # Pupil
            np.random.normal(14, 2, samples_per_state),   # Resp
            np.random.normal(15, 3, samples_per_state),   # Testosterone
            np.random.normal(1.0, 0.2, samples_per_state),# Oxytocin
            np.random.normal(5, 1, samples_per_state),    # Prolactin
            np.random.normal(10, 2, samples_per_state)    # Cortisol
        ]).T
        X_list.append(X_baseline)
        y_list.append(np.zeros(samples_per_state))

        # State 1: Desire
        X_desire = np.array([
            np.random.normal(80, 5, samples_per_state),
            np.random.normal(125, 10, samples_per_state),
            np.random.normal(82, 5, samples_per_state),
            np.random.normal(7, 1.5, samples_per_state),
            np.random.normal(0.5, 0.1, samples_per_state),
            np.random.normal(4, 0.4, samples_per_state),
            np.random.normal(16, 2, samples_per_state),
            np.random.normal(16, 3, samples_per_state),
            np.random.normal(1.2, 0.2, samples_per_state),
            np.random.normal(5, 1, samples_per_state),
            np.random.normal(9, 2, samples_per_state)
        ]).T
        X_list.append(X_desire)
        y_list.append(np.ones(samples_per_state))

        # State 2: Arousal
        X_arousal = np.array([
            np.random.normal(110, 10, samples_per_state),
            np.random.normal(140, 12, samples_per_state),
            np.random.normal(90, 6, samples_per_state),
            np.random.normal(15, 3, samples_per_state),
            np.random.normal(0.9, 0.15, samples_per_state),
            np.random.normal(5.5, 0.5, samples_per_state),
            np.random.normal(22, 3, samples_per_state),
            np.random.normal(17, 3, samples_per_state),
            np.random.normal(1.5, 0.3, samples_per_state),
            np.random.normal(5.5, 1, samples_per_state),
            np.random.normal(8, 2, samples_per_state)
        ]).T
        X_list.append(X_arousal)
        y_list.append(np.full(samples_per_state, 2))

        # State 3: Orgasm
        X_orgasm = np.array([
            np.random.normal(160, 15, samples_per_state),
            np.random.normal(170, 15, samples_per_state),
            np.random.normal(100, 8, samples_per_state),
            np.random.normal(25, 4, samples_per_state),
            np.random.normal(1.0, 0.1, samples_per_state),
            np.random.normal(7, 0.6, samples_per_state),
            np.random.normal(35, 5, samples_per_state),
            np.random.normal(18, 3, samples_per_state),
            np.random.normal(4.0, 0.5, samples_per_state),  # Oxytocin surge
            np.random.normal(6, 1.5, samples_per_state),
            np.random.normal(7, 2, samples_per_state)
        ]).T
        X_list.append(X_orgasm)
        y_list.append(np.full(samples_per_state, 3))

        # State 4: Resolution
        X_resolution = np.array([
            np.random.normal(85, 8, samples_per_state),
            np.random.normal(130, 12, samples_per_state),
            np.random.normal(85, 6, samples_per_state),
            np.random.normal(8, 2, samples_per_state),
            np.random.normal(0.4, 0.1, samples_per_state),
            np.random.normal(4.5, 0.4, samples_per_state),
            np.random.normal(18, 3, samples_per_state),
            np.random.normal(15, 3, samples_per_state),
            np.random.normal(2.0, 0.3, samples_per_state),
            np.random.normal(15, 3, samples_per_state),     # Prolactin high
            np.random.normal(9, 2, samples_per_state)
        ]).T
        X_list.append(X_resolution)
        y_list.append(np.full(samples_per_state, 4))

        # Concatenate all states
        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        # Add noise
        noise = np.random.normal(0, noise_level, X.shape)
        X += noise

        # Ensure non-negative values
        X = np.maximum(X, 0)

        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        return X, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the neural network classifier.

        Parameters
        ----------
        X : ndarray
            Feature matrix (n_samples, n_features)
        y : ndarray
            Labels (n_samples,)
        test_size : float
            Fraction of data for testing

        Returns
        -------
        dict
            Training metrics (accuracy, f1_score, etc.)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train classifier
        print("Training neural network...")
        self.classifier.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred_train = self.classifier.predict(X_train_scaled)
        y_pred_test = self.classifier.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')

        self.is_trained = True
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.y_pred = y_pred_test

        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'n_iterations': self.classifier.n_iter_
        }

        print(f"✓ Training complete")
        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Test accuracy:  {test_acc:.3f}")
        print(f"  Test F1 score:  {test_f1:.3f}")
        print(f"  Iterations:     {self.classifier.n_iter_}")

        return metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict arousal states.

        Parameters
        ----------
        X : ndarray
            Feature matrix

        Returns
        -------
        states : ndarray
            Predicted state labels
        probabilities : ndarray
            Prediction probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)
        states = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)

        return states, probabilities

    def plot_confusion_matrix(self) -> plt.Figure:
        """Plot confusion matrix of test predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        cm = confusion_matrix(self.y_test, self.y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.state_names,
            yticklabels=self.state_names,
            ax=ax
        )
        ax.set_xlabel('Predicted State', fontsize=12)
        ax.set_ylabel('True State', fontsize=12)
        ax.set_title('Confusion Matrix - Arousal State Classification',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_feature_importance(self) -> plt.Figure:
        """Plot feature importance (weight magnitudes)."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        # Get weights from first layer
        weights = np.abs(self.classifier.coefs_[0]).mean(axis=1)

        # Sort by importance
        indices = np.argsort(weights)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(weights)), weights[indices], color='steelblue')
        ax.set_yticks(range(len(weights)))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.set_xlabel('Average Absolute Weight', fontsize=12)
        ax.set_title('Feature Importance in Arousal Classification',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_learning_curve(self) -> plt.Figure:
        """Plot learning curves (loss over iterations)."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.classifier.loss_curve_, linewidth=2, color='steelblue')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_prediction_confidence(self) -> plt.Figure:
        """Plot prediction confidence distribution."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        probs = self.classifier.predict_proba(self.X_test)
        max_probs = probs.max(axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(max_probs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(max_probs.mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {max_probs.mean():.3f}')
        ax.set_xlabel('Maximum Probability', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Prediction Confidence Distribution',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig


def example_usage():
    """Demonstrate classifier usage."""
    print("=" * 70)
    print("Neural Network Arousal State Classifier")
    print("=" * 70)
    print()

    # Create classifier
    print("Initializing classifier...")
    classifier = ArousalStateClassifier(
        hidden_layers=(64, 32, 16),
        learning_rate=0.001,
        max_iterations=500
    )
    print("✓ Classifier initialized")
    print()

    # Generate synthetic data
    print("Generating synthetic training data (n=2000)...")
    X, y = classifier.generate_synthetic_data(n_samples=2000, noise_level=0.15)
    print(f"✓ Data generated: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  State distribution: {np.bincount(y.astype(int))}")
    print()

    # Train
    metrics = classifier.train(X, y, test_size=0.2)
    print()

    # Generate plots
    print("Generating visualizations...")
    fig1 = classifier.plot_confusion_matrix()
    fig1.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: confusion_matrix.png")

    fig2 = classifier.plot_feature_importance()
    fig2.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: feature_importance.png")

    fig3 = classifier.plot_learning_curve()
    fig3.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: learning_curve.png")

    fig4 = classifier.plot_prediction_confidence()
    fig4.savefig('prediction_confidence.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: prediction_confidence.png")

    print()

    # Make sample prediction
    print("Example Predictions:")
    print("-" * 70)

    # Sample from each state
    sample_indices = [0, 200, 400, 600, 800]  # One from each state
    X_samples = X[sample_indices]
    y_true = y[sample_indices]

    states_pred, probs_pred = classifier.predict(X_samples)

    for i, (true_state, pred_state, prob) in enumerate(zip(y_true, states_pred, probs_pred)):
        true_name = classifier.state_names[int(true_state)]
        pred_name = classifier.state_names[int(pred_state)]
        max_prob = prob.max()
        print(f"Sample {i+1}:")
        print(f"  True:      {true_name}")
        print(f"  Predicted: {pred_name} (confidence: {max_prob:.3f})")
        print()

    print("=" * 70)
    print("Example complete!")
    print("=" * 70)

    return classifier


if __name__ == '__main__':
    classifier = example_usage()

    # Show plots (comment out if running headless)
    # plt.show()
