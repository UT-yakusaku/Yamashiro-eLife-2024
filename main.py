import os
import logging
import datetime
from omegaconf import OmegaConf
from trainer import TrainerMultihead


def setup_logging(log_dir: str) -> None:
    """Configure logging for the training run."""
    log_file = os.path.join(log_dir, "main.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Logging initialized.")


def create_result_dir(base_dir: str) -> str:
    """Create a unique result directory based on current time."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, f"autotrain_multihead_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    logging.debug(f"Created result directory: {result_dir}")
    return result_dir


def main(cfg) -> None:
    """Main training execution with error handling."""
    try:
        trainer = TrainerMultihead(cfg)
        trainer.train()
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user. Cleaning up...")
        trainer.remove_result_dir()


if __name__ == "__main__":
    # Load configuration
    cfg = OmegaConf.load("config.yaml")

    # Create a new result directory
    cfg.path.result_dir = create_result_dir(cfg.path.result_dir)

    # Initialize logging
    setup_logging(cfg.path.result_dir)

    # Save configuration snapshot
    config_path = os.path.join(cfg.path.result_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)
    logging.info(f"Configuration saved to {config_path}")

    # Start training
    main(cfg)
