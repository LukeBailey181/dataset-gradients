from data import MnistManager, RandomMemorySetManager
import wandb

# TODO add proper config files
USE_WANDB = False


def main():
    # Setup wandb
    if USE_WANDB:
        run_name = f"test_run"

        wandb.init(
            project="dataset_gradients",
            entity="lukebailey",
            name=run_name,
            config={
                "dataset_name": "mnist",
            },
        )

    memory_set_manager = RandomMemorySetManager(p=0.5, random_seed=42)
    mnist_manager = MnistManager(
        memory_set_manager=memory_set_manager, use_wandb=USE_WANDB
    )

    # Train on first task
    mnist_manager.train()

    # Train on second task
    mnist_manager.next_task()
    mnist_manager.train()


if __name__ == "__main__":
    main()
