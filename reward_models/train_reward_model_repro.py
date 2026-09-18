"""CLI entry point for the restartable reward-model trainer."""

try:
    from .reward_training import main
except ImportError:  # Direct execution: python reward_models/train_reward_model_repro.py
    from reward_training import main


if __name__ == "__main__":
    raise SystemExit(main())
