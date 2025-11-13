"""
Test script for checkpoint loading and saving functionality.
Tests that checkpoints can be saved and loaded correctly without data corruption.
"""

import os
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_files.tm_config import Config_tm
from src.agents.rainbow_tm import Rainbow


def test_checkpoint_save_and_load():
    """Test that checkpoints can be saved and loaded without errors."""
    print("\n" + "="*60)
    print("TEST 1: Basic Checkpoint Save and Load")
    print("="*60)
    
    # Create agent with config
    config = Config_tm()
    agent = Rainbow(config)
    
    # Get initial state
    initial_epsilon = agent.epsilon
    initial_policy_state = agent.policy_network.state_dict()
    initial_target_state = agent.target_network.state_dict()
    
    print(f"✓ Agent created")
    print(f"  - Initial epsilon: {initial_epsilon}")
    print(f"  - Policy network parameters: {sum(p.numel() for p in agent.policy_network.parameters())}")
    
    # Save checkpoint
    checkpoint_dir = "test_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "test_checkpoint.pt")
    
    try:
        agent.save_checkpoint(checkpoint_path, episode=1, step=100, additional_info={"test": "data"})
        assert os.path.exists(checkpoint_path), "Checkpoint file was not created"
        print(f"✓ Checkpoint saved successfully to {checkpoint_path}")
        
        # Check file size
        file_size = os.path.getsize(checkpoint_path)
        print(f"  - Checkpoint file size: {file_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"✗ Failed to save checkpoint: {e}")
        return False
    
    # Create new agent and load checkpoint
    agent2 = Rainbow(config)
    
    try:
        loaded_checkpoint = agent2.load_checkpoint(checkpoint_path)
        print(f"✓ Checkpoint loaded successfully")
        
        # Verify loaded state
        assert agent2.epsilon == initial_epsilon, f"Epsilon mismatch: {agent2.epsilon} != {initial_epsilon}"
        print(f"  - Epsilon restored: {agent2.epsilon}")
        
        assert 'episode' in loaded_checkpoint, "Episode not in checkpoint"
        assert loaded_checkpoint['episode'] == 1, f"Episode mismatch: {loaded_checkpoint['episode']}"
        print(f"  - Episode: {loaded_checkpoint['episode']}")
        
        assert 'step' in loaded_checkpoint, "Step not in checkpoint"
        assert loaded_checkpoint['step'] == 100, f"Step mismatch: {loaded_checkpoint['step']}"
        print(f"  - Step: {loaded_checkpoint['step']}")
        
        # Verify network states match
        loaded_policy_state = agent2.policy_network.state_dict()
        for key in initial_policy_state:
            if not torch.equal(initial_policy_state[key], loaded_policy_state[key]):
                print(f"✗ Policy network state mismatch for key: {key}")
                return False
        print(f"  - Policy network state verified")
        
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return False
    
    # Clean up
    os.remove(checkpoint_path)
    print(f"✓ Test passed!\n")
    return True


def test_existing_checkpoint_loading():
    """Test loading of existing checkpoints from the project."""
    print("="*60)
    print("TEST 2: Load Existing Project Checkpoints")
    print("="*60)
    
    checkpoint_dir = Path("checkpoints/Fall_6_cranked_3_Rainbow_TM20FULL_Dueling+PER+Double")
    
    if not checkpoint_dir.exists():
        print(f"⊘ Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    # List available checkpoints
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    print(f"Found {len(checkpoint_files)} checkpoint files:")
    for cp_file in checkpoint_files:
        size_mb = cp_file.stat().st_size / 1024 / 1024
        print(f"  - {cp_file.name} ({size_mb:.2f} MB)")
    
    if not checkpoint_files:
        print("✗ No checkpoint files found")
        return False
    
    # Try loading each checkpoint
    config = Config_tm()
    results = {}
    
    for checkpoint_file in checkpoint_files[:3]:  # Test first 3 checkpoints
        print(f"\nLoading {checkpoint_file.name}...")
        try:
            agent = Rainbow(config)
            checkpoint = agent.load_checkpoint(str(checkpoint_file))
            
            # Verify critical fields
            assert 'epsilon' in checkpoint, "Missing epsilon"
            assert 'episode' in checkpoint, "Missing episode"
            assert 'step' in checkpoint, "Missing step"
            
            print(f"  ✓ Successfully loaded")
            print(f"    - Episode: {checkpoint['episode']}, Step: {checkpoint['step']}")
            print(f"    - Epsilon: {checkpoint['epsilon']:.6f}")
            
            if config.use_prioritized_replay:
                print(f"    - Replay buffer beta: {agent.replay_buffer._sampler.beta:.6f}")
            
            results[checkpoint_file.name] = True
            
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            results[checkpoint_file.name] = False
    
    # Summary
    success_count = sum(1 for v in results.values() if v)
    print(f"\n✓ Successfully loaded {success_count}/{len(results)} checkpoints")
    
    if success_count == len(results):
        print(f"✓ Test passed!\n")
        return True
    else:
        print(f"✗ Some checkpoints failed to load\n")
        return False


def test_checkpoint_contents():
    """Test that checkpoint contains all expected fields."""
    print("="*60)
    print("TEST 3: Verify Checkpoint Structure")
    print("="*60)
    
    config = Config_tm()
    agent = Rainbow(config)
    
    checkpoint_dir = "test_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "structure_test.pt")
    
    try:
        agent.save_checkpoint(checkpoint_path, episode=5, step=250, additional_info={"run_id": "test123"})
        
        # Load raw checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=agent.device)
        
        print("Checkpoint structure:")
        for key, value in checkpoint.items():
            if isinstance(value, dict):
                print(f"  ✓ {key}: dict with {len(value)} entries")
            elif isinstance(value, torch.Tensor):
                print(f"  ✓ {key}: tensor {value.shape}")
            elif isinstance(value, list):
                print(f"  ✓ {key}: list with {len(value)} items")
            else:
                print(f"  ✓ {key}: {type(value).__name__}")
        
        # Verify essential fields
        essential_fields = [
            'episode', 'step', 'epsilon',
            'policy_network_state_dict', 'target_network_state_dict',
            'optimizer_state_dict', 'config',
            'n_step_buffer', 'n_step_buffer_len'
        ]
        
        missing_fields = [f for f in essential_fields if f not in checkpoint]
        if missing_fields:
            print(f"\n✗ Missing essential fields: {missing_fields}")
            return False
        
        print(f"\n✓ All {len(essential_fields)} essential fields present")
        
        # Check for prioritized replay fields if applicable
        if config.use_prioritized_replay:
            if 'beta' in checkpoint:
                print(f"✓ Prioritized replay beta: {checkpoint['beta']}")
            else:
                print(f"⊘ Warning: PER enabled but beta not in checkpoint")
        
        # Clean up
        os.remove(checkpoint_path)
        print(f"✓ Test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CHECKPOINT LOADING TEST SUITE")
    print("="*60)
    
    results = {
        "Basic Save/Load": test_checkpoint_save_and_load(),
        "Existing Checkpoints": test_existing_checkpoint_loading(),
        "Checkpoint Structure": test_checkpoint_contents(),
    }
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60 + "\n")
    
    # Clean up test directory
    import shutil
    if os.path.exists("test_checkpoints"):
        shutil.rmtree("test_checkpoints")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
