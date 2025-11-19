"""
Test and Validate All Models

Comprehensive testing suite for arousal neuroscience models.

Tests:
- Arousal dynamics model
- Hormonal dynamics model
- Neural network classifier
- Integrated model

Author: Deep Convo GPT Research Team
Date: 2025-11-19
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Import all models
from arousal_dynamics import ArousalModel, example_stimuli
from hormonal_dynamics import HormonalDynamicsModel
from neural_network_classifier import ArousalStateClassifier
from integrated_arousal_model import IntegratedArousalModel


def test_arousal_dynamics():
    """Test basic arousal dynamics model."""
    print("\n" + "="*70)
    print("TEST 1: Arousal Dynamics Model")
    print("="*70)

    try:
        # Create model
        model = ArousalModel()
        print("✓ Model instantiation successful")

        # Run simulation
        t, E, I, arousal = model.simulate(duration=50)
        print(f"✓ Simulation successful ({len(t)} timesteps)")

        # Validate outputs
        assert len(t) == len(E) == len(I) == len(arousal), "Output length mismatch"
        assert arousal.max() > 0, "No arousal detected"
        assert E.max() > 0, "No excitation detected"
        print("✓ Output validation passed")

        # Test different stimuli
        stimuli = example_stimuli()
        for name, stim_func in stimuli.items():
            t, E, I, arousal = model.simulate(duration=60, stimulus=stim_func)
            assert len(t) > 0, f"Simulation failed for {name} stimulus"
        print(f"✓ All {len(stimuli)} stimulus patterns tested successfully")

        print("\n✅ Arousal Dynamics Model: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Arousal Dynamics Model: TEST FAILED")
        print(f"   Error: {e}")
        return False


def test_hormonal_dynamics():
    """Test hormonal dynamics model."""
    print("\n" + "="*70)
    print("TEST 2: Hormonal Dynamics Model")
    print("="*70)

    try:
        # Test male model
        model_male = HormonalDynamicsModel(sex='male')
        print("✓ Male model instantiation successful")

        results_male = model_male.simulate(duration=6.0, dt=0.01)
        print(f"✓ Male simulation successful")

        # Validate hormone levels are physiologically reasonable
        assert 10 < results_male['Testosterone'].mean() < 25, \
            f"Testosterone out of range: {results_male['Testosterone'].mean()}"
        assert results_male['Prolactin'].min() >= 0, "Negative prolactin"
        assert results_male['Oxytocin'].min() >= 0, "Negative oxytocin"
        print("✓ Male hormone levels within physiological range")

        # Test female model
        model_female = HormonalDynamicsModel(sex='female')
        results_female = model_female.simulate(duration=6.0, dt=0.01)
        print("✓ Female simulation successful")

        # Female should have higher estrogen
        assert results_female['Estrogen'].mean() > results_male['Estrogen'].mean(), \
            "Female estrogen should be higher than male"
        print("✓ Sex differences validated")

        # Test arousal response
        def arousal_func(t):
            return 1.0 if 1.0 < t < 2.0 else 0.0

        results_arousal = model_male.simulate(
            duration=4.0,
            dt=0.01,
            arousal_signal=arousal_func,
            orgasm_times=[1.5]
        )

        # Oxytocin should peak after orgasm
        orgasm_idx = int(1.5 / 0.01)
        post_orgasm = results_arousal['Oxytocin'][orgasm_idx:orgasm_idx+100]
        assert post_orgasm.max() > results_arousal['Oxytocin'][:orgasm_idx].mean() * 2, \
            "Oxytocin should surge after orgasm"
        print("✓ Orgasm response validated")

        print("\n✅ Hormonal Dynamics Model: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Hormonal Dynamics Model: TEST FAILED")
        print(f"   Error: {e}")
        return False


def test_neural_network_classifier():
    """Test neural network classifier."""
    print("\n" + "="*70)
    print("TEST 3: Neural Network Classifier")
    print("="*70)

    try:
        # Create classifier
        classifier = ArousalStateClassifier(max_iterations=200)
        print("✓ Classifier instantiation successful")

        # Generate training data
        X, y = classifier.generate_synthetic_data(n_samples=500, noise_level=0.1)
        print(f"✓ Data generation successful ({X.shape[0]} samples, {X.shape[1]} features)")

        # Validate data
        assert X.shape[0] == 500, "Incorrect number of samples"
        assert X.shape[1] == 11, "Incorrect number of features"
        assert len(np.unique(y)) == 5, "Should have 5 arousal states"
        print("✓ Data validation passed")

        # Train classifier
        metrics = classifier.train(X, y, test_size=0.3)
        print("✓ Training successful")

        # Validate metrics
        assert metrics['test_accuracy'] > 0.6, \
            f"Accuracy too low: {metrics['test_accuracy']}"
        assert metrics['test_f1'] > 0.6, \
            f"F1 score too low: {metrics['test_f1']}"
        print(f"✓ Performance metrics acceptable (acc={metrics['test_accuracy']:.3f}, f1={metrics['test_f1']:.3f})")

        # Test prediction
        X_test = X[:5]
        states, probs = classifier.predict(X_test)
        assert len(states) == 5, "Wrong number of predictions"
        assert probs.shape == (5, 5), "Wrong probability shape"
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities don't sum to 1"
        print("✓ Prediction functionality validated")

        print("\n✅ Neural Network Classifier: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Neural Network Classifier: TEST FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_model():
    """Test integrated multi-level model."""
    print("\n" + "="*70)
    print("TEST 4: Integrated Multi-Level Model")
    print("="*70)

    try:
        # Create model
        model = IntegratedArousalModel(
            sex='male',
            excitation_sensitivity=1.0,
            stress_level=0.1
        )
        print("✓ Integrated model instantiation successful")

        # Run short simulation
        results = model.simulate(duration=0.5, dt=0.01)  # 30 minutes
        print("✓ Simulation successful")

        # Validate all output keys present
        required_keys = [
            'time', 'stimulus', 'excitation', 'inhibition', 'arousal',
            'Testosterone', 'Prolactin', 'Oxytocin',
            'heart_rate', 'genital_blood_flow'
        ]
        for key in required_keys:
            assert key in results, f"Missing output: {key}"
        print("✓ All output variables present")

        # Validate physiological ranges
        assert 60 <= results['heart_rate'].min() <= 200, "Heart rate out of range"
        assert 0 <= results['genital_blood_flow'].max() <= 1.5, "Genital flow out of range"
        print("✓ Physiological values within reasonable ranges")

        # Test that arousal affects hormones
        arousal_nonzero = results['arousal'] > 0.1
        if arousal_nonzero.any():
            oxy_during = results['Oxytocin'][arousal_nonzero].mean()
            oxy_baseline = results['Oxytocin'][~arousal_nonzero].mean() if (~arousal_nonzero).any() else oxy_during
            assert oxy_during >= oxy_baseline, "Oxytocin should increase with arousal"
        print("✓ Cross-level interactions validated")

        print("\n✅ Integrated Model: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Integrated Model: TEST FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run comprehensive test suite."""
    print("\n" + "🧪"*35)
    print("  COMPREHENSIVE MODEL TEST SUITE  ")
    print("🧪"*35 + "\n")

    results = {
        'Arousal Dynamics': test_arousal_dynamics(),
        'Hormonal Dynamics': test_hormonal_dynamics(),
        'Neural Network': test_neural_network_classifier(),
        'Integrated Model': test_integrated_model()
    }

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    all_passed = True
    for model_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{model_name:.<50} {status}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! All models are functioning correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED. Review errors above.")

    print()
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
