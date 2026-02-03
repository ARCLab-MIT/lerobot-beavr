"""
Comprehensive Train-Inference Correspondence Test for MACT Policy.

This test verifies the correspondence between training and inference paths.

Key findings:
1. n_action_steps is ENFORCED to be 1 in MACTConfig (line 165-169)
2. With n_action_steps=1: History updates at every observation_stride
3. The critical test is whether identical histories produce identical actions
"""

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.mact.configuration_mact import MACTConfig
from lerobot.policies.mact.modeling_mact import MACTPolicy
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

HISTORY_TOKEN = "history_token"


def create_config(observation_stride: int = 1) -> MACTConfig:
    """Create a minimal MACT config for testing."""
    input_features = {
        "observation.images.laptop": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    return MACTConfig(
        n_obs_steps=5,
        observation_stride=observation_stride,
        n_history_tokens=5,
        input_features=input_features,
        output_features=output_features,
        chunk_size=10,
        n_action_steps=1,  # MUST be 1 per config constraint
    )


def test_history_encoder_equivalence():
    """
    TEST 1: Verify HistoryEncoder parallel scan matches recurrent stepping.

    This is THE fundamental test for train-inference correspondence.
    Training uses forward() with parallel Mamba scan.
    Inference uses step() with recurrent Mamba updates.
    """
    print("\n" + "=" * 70)
    print("TEST 1: History Encoder Parallel vs Recurrent Equivalence")
    print("=" * 70)

    config = create_config(observation_stride=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    policy = MACTPolicy(config).to(device)
    policy.eval()

    batch_size = 2
    n_obs = config.n_obs_steps
    images = torch.randn(batch_size, n_obs, 1, 3, 224, 224, device=device)
    states = torch.randn(batch_size, n_obs, 6, device=device)

    # TRAINING PATH: Parallel scan
    train_batch = {
        OBS_IMAGES: images,
        OBS_STATE: states,
    }
    with torch.no_grad():
        h_seq_parallel = policy.history_encoder(train_batch)

    # INFERENCE PATH: Recurrent stepping (direct, not through select_action)
    cache = policy.history_encoder.init_cache(batch_size, dtype=images.dtype)
    h_seq_recurrent = []

    with torch.no_grad():
        for t in range(n_obs):
            obs_t = images[:, t]
            x_t = policy.history_encoder.fuse_one_timestep(obs_t, timestep_idx=t)
            h_t, cache = policy.history_encoder.step(x_t, cache)
            h_seq_recurrent.append(h_t)

    h_seq_recurrent = torch.stack(h_seq_recurrent, dim=1)

    max_diff = (h_seq_parallel - h_seq_recurrent).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        h_seq_parallel.flatten(), h_seq_recurrent.flatten(), dim=0
    ).item()

    print(f"  Parallel shape: {h_seq_parallel.shape}")
    print(f"  Recurrent shape: {h_seq_recurrent.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    passed = max_diff < 1e-3 and cos_sim > 0.999
    print(f"\n  RESULT: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed, max_diff


def test_select_action_history_accumulation():
    """
    TEST 2: Verify select_action accumulates history correctly.

    With n_action_steps=1 and observation_stride=S:
    - predict_action_chunk called at EVERY step
    - History updated at steps 0, S, 2S, 3S, ...
    """
    print("\n" + "=" * 70)
    print("TEST 2: Select Action History Accumulation")
    print("=" * 70)

    observation_stride = 2
    config = create_config(observation_stride=observation_stride)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    policy = MACTPolicy(config).to(device)
    policy.eval()

    n_frames = 10
    batch_size = 1
    images = torch.randn(batch_size, n_frames, 1, 3, 224, 224, device=device)
    states = torch.randn(batch_size, n_frames, 6, device=device)

    policy.reset()

    history_update_steps = []
    for t in range(n_frames):
        old_count = getattr(policy, "_strided_obs_idx", 0) or 0

        batch = {
            "observation.images.laptop": images[:, t, 0],
            "observation.state": states[:, t],
        }
        policy.select_action(batch)

        new_count = getattr(policy, "_strided_obs_idx", 0) or 0
        if new_count > old_count:
            history_update_steps.append(t)

    expected_updates = list(range(0, n_frames, observation_stride))

    print(f"  Observation stride: {observation_stride}")
    print(f"  Total frames: {n_frames}")
    print(f"  History update steps: {history_update_steps}")
    print(f"  Expected: {expected_updates}")
    print(f"  Total history tokens: {len(policy._history_tokens)}")

    passed = history_update_steps == expected_updates
    print(f"\n  RESULT: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


def test_full_pipeline_correspondence():
    """
    TEST 3: Full pipeline correspondence - the critical test.

    Compares:
    - Training: history_encoder.forward() -> model()
    - Inference: select_action loop -> model()

    Using IDENTICAL observations at the SAME positions.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Full Pipeline Correspondence")
    print("=" * 70)

    observation_stride = 1  # Keep stride=1 for exact frame correspondence
    config = create_config(observation_stride=observation_stride)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    policy = MACTPolicy(config).to(device)
    policy.eval()

    batch_size = 1
    n_obs = config.n_obs_steps
    images = torch.randn(batch_size, n_obs, 1, 3, 224, 224, device=device)
    states = torch.randn(batch_size, n_obs, 6, device=device)

    # =========== TRAINING PATH ===========
    train_batch = {
        OBS_IMAGES: images,
        OBS_STATE: states[:, -1],
    }

    with torch.no_grad():
        history_train = policy.history_encoder(train_batch)

        model_batch_train = {
            HISTORY_TOKEN: history_train,
            OBS_IMAGES: images[:, -1],
            OBS_STATE: states[:, -1],
        }
        actions_train, _ = policy.model(model_batch_train)

    # =========== INFERENCE PATH ===========
    policy.reset()

    # Step through all observations
    for t in range(n_obs):
        batch = {
            "observation.images.laptop": images[:, t, 0],
            "observation.state": states[:, t],
        }
        action_inference = policy.select_action(batch)

    # Get accumulated history
    history_inference = torch.stack(list(policy._history_tokens), dim=1)

    # Compare histories
    history_diff = (history_train - history_inference).abs().max().item()
    history_cos = torch.nn.functional.cosine_similarity(
        history_train.flatten(), history_inference.flatten(), dim=0
    ).item()

    print(f"  History comparison:")
    print(f"    Train shape: {history_train.shape}")
    print(f"    Inference shape: {history_inference.shape}")
    print(f"    Max diff: {history_diff:.2e}")
    print(f"    Cosine sim: {history_cos:.6f}")

    # Now compare MODEL outputs given the SAME history
    # This isolates whether the model produces same output for same input
    with torch.no_grad():
        model_batch_inference = {
            HISTORY_TOKEN: history_inference,  # Use inference history
            OBS_IMAGES: images[:, -1],
            OBS_STATE: states[:, -1],
        }
        actions_from_inf_history, _ = policy.model(model_batch_inference)

    action_diff = (actions_train - actions_from_inf_history).abs().max().item()
    action_cos = torch.nn.functional.cosine_similarity(
        actions_train.flatten(), actions_from_inf_history.flatten(), dim=0
    ).item()

    print(f"\n  Action comparison (from model with respective histories):")
    print(f"    Max diff: {action_diff:.2e}")
    print(f"    Cosine sim: {action_cos:.6f}")

    # Also check: using IDENTICAL history
    with torch.no_grad():
        model_batch_same_history = {
            HISTORY_TOKEN: history_train,  # Force same history
            OBS_IMAGES: images[:, -1],
            OBS_STATE: states[:, -1],
        }
        actions_same_history, _ = policy.model(model_batch_same_history)

    same_history_diff = (actions_train - actions_same_history).abs().max().item()
    print(f"\n  Sanity check (model with SAME history):")
    print(f"    Max diff: {same_history_diff:.2e}")

    history_passed = history_diff < 1e-3
    action_passed = action_diff < 1e-2

    print(f"\n  History correspondence: {'PASS ✓' if history_passed else 'FAIL ✗'}")
    print(f"  Action correspondence: {'PASS ✓' if action_passed else 'FAIL ✗'}")

    overall_passed = history_passed and action_passed
    print(f"  OVERALL: {'PASS ✓' if overall_passed else 'FAIL ✗'}")

    if not overall_passed:
        print("\n  DIAGNOSIS:")
        if not history_passed:
            print("    - History mismatch indicates fuse_one_timestep/forward differ")
        if not action_passed and history_passed:
            print("    - Actions differ despite matching history - check model inputs")

    return overall_passed


def test_action_at_each_step():
    """
    TEST 4: Compare actions at each inference step against "training" equivalent.

    At each step t, we compare:
    - Inference: select_action after observing frames 0..t
    - Training: Fresh policy observing same frames 0..t
    """
    print("\n" + "=" * 70)
    print("TEST 4: Per-Step Action Consistency")
    print("=" * 70)

    config = create_config(observation_stride=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    policy = MACTPolicy(config).to(device)
    policy.eval()

    batch_size = 1
    n_obs = config.n_obs_steps
    images = torch.randn(batch_size, n_obs, 1, 3, 224, 224, device=device)
    states = torch.randn(batch_size, n_obs, 6, device=device)

    # Collect inference actions
    policy.reset()
    inference_actions = []
    for t in range(n_obs):
        batch = {
            "observation.images.laptop": images[:, t, 0],
            "observation.state": states[:, t],
        }
        action = policy.select_action(batch)
        inference_actions.append(action.clone())

    # Collect "training" actions (fresh policy at each step)
    training_actions = []
    for t in range(n_obs):
        # Fresh policy for each "training sample"
        test_policy = MACTPolicy(config).to(device)
        test_policy.load_state_dict(policy.state_dict())
        test_policy.eval()
        test_policy.reset()

        for s in range(t + 1):
            batch = {
                "observation.images.laptop": images[:, s, 0],
                "observation.state": states[:, s],
            }
            action = test_policy.select_action(batch)

        training_actions.append(action.clone())

    print(f"  Comparing actions at each step:")
    all_diffs = []
    for t in range(n_obs):
        diff = (inference_actions[t] - training_actions[t]).abs().max().item()
        all_diffs.append(diff)
        status = "✓" if diff < 1e-3 else "✗"
        print(f"    T={t}: diff = {diff:.2e} {status}")

    max_diff = max(all_diffs)
    passed = max_diff < 1e-3
    print(f"\n  Max diff across all steps: {max_diff:.2e}")
    print(f"  RESULT: {'PASS ✓' if passed else 'FAIL ✗'}")

    return passed


def run_all_tests():
    """Run all correspondence tests."""
    print("\n" + "#" * 70)
    print("# MACT Train-Inference Correspondence Tests")
    print("#" * 70)
    print("\nNote: MACTConfig enforces n_action_steps=1")
    print("      All tests use this constraint.\n")

    results = {
        "1: History Encoder Equivalence": test_history_encoder_equivalence()[0],
        "2: History Accumulation": test_select_action_history_accumulation(),
        "3: Full Pipeline": test_full_pipeline_correspondence(),
        "4: Per-Step Consistency": test_action_at_each_step(),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\n  Overall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
