from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Type, Set

import torch
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from jaxtyping import Float
from torch.utils.data.dataset import Dataset as TorchDataset
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import wandb

from data import MemorySetManager
from models import MnistMLP
from utils import MNIST_FEATURE_SIZE


class Task:
    """Class storing all the data for a certain task in a continual learning setting.

    Every task contains some gold standard train_set, all the information you would want for that task,
    then some test_set, which is used to evaluate the model on that task, and a memory set, which is used
    when the task is not the current primary continual learning task, but instead is in the past.
    """

    def __init__(
        self,
        train_x: Float[Tensor, "n f"],
        train_y: Float[Tensor, "n 1"],
        test_x: Float[Tensor, "m f"],
        test_y: Float[Tensor, "m 1"],
        task_labels: Set[int],
        memory_set_manager: MemorySetManager,
    ):
        """
        Args:
            train_x: The training examples for this task.
            train_y: The training labels for this task.
            test_x: The test examples for this task.
            test_y: The test labels for this task.
            task_labels: Set of labels that this task uses.
            memory_set_manager: The memory set manager to use to create the memory set.
        """
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.task_labels = task_labels

        self.memory_x, self.memory_y = memory_set_manager.create_memory_set(
            train_x, train_y
        )
        self.task_labels = task_labels
        self.active = False


class ContinualLearningManager(ABC):
    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        dataset_path: str = "./data",
        use_wandb=True,
    ):
        self.use_wandb = use_wandb

        train_x, train_y, test_x, test_y = self._load_dataset(dataset_path=dataset_path)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.memory_set_manager = memory_set_manager

        self.tasks = self._init_tasks()  # List of all tasks
        self.task_index = (
            0  # Index of the current task, all tasks <= task_index are active
        )

    @abstractmethod
    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them"""
        pass

    @abstractmethod
    def _load_dataset(
        self,
    ) -> Tuple[
        Float[Tensor, "n f"],
        Float[Tensor, "n 1"],
        Float[Tensor, "m f"],
        Float[Tensor, "m 1"],
    ]:
        """Load full dataset for all tasks"""
        pass

    @abstractmethod
    def train(self, batch_size, lr, use_memory_set=True) -> None:
        """Train on all tasks with index <= self.task_index

        Args:
            use_memory_set: True then tasks with index < task_index use memory set,
                otherwise they use the full training set.
        """
        pass

    def next_task(self) -> None:
        """Iterate to next task"""
        self.task_index += 1
        if self.task_index >= len(self.tasks):
            raise IndexError("No more tasks")
        self.tasks[self.task_index].active = True


class MnistManager(ContinualLearningManager):
    def __init__(
        self, memory_set_manager: MemorySetManager, dataset_path: str = "./data"
    ):
        super().__init__(memory_set_manager, dataset_path)
        self.model = MnistMLP()

    def train(self, epochs=5, batch_size=32, lr=0.01, use_memory_set=True) -> None:
        # Get tasks
        running_tasks = self.tasks[: self.task_index + 1]
        for task in running_tasks:
            assert task.active

        terminal_task = running_tasks[-1]
        memory_tasks = running_tasks[:-1]  # This could be empty

        # Create a dataset for all tasks <= task_index
        if use_memory_set:
            combined_train_x = torch.cat(
                [task.memory_x for task in running_tasks] + [terminal_task.train_x]
            )
            combined_train_y = torch.cat(
                [task.memory_y for task in running_tasks] + [terminal_task.train_y]
            )
        else:
            combined_train_x = torch.cat([task.train_x for task in running_tasks]) + [
                terminal_task.train_x
            ]
            combined_train_y = torch.cat([task.train_y for task in running_tasks]) + [
                terminal_task.train_y
            ]

        assert combined_train_x.shape[1] == MNIST_FEATURE_SIZE

        # Identify the labels for the combined dataset
        combined_labels = set.union(*[task.task_labels for task in running_tasks])

        # Randomize the dataset
        n = combined_train_x.shape[0]
        perm = torch.randperm(n)
        combined_train_x = combined_train_x[perm]
        combined_train_x = combined_train_x[perm]

        # Put into batches
        dataset = TorchDataset(combined_train_x, combined_train_y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        breakpoint()
        # Train on batches
        criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification tasks
        optimizer = SGD(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.use_wandb:
                    wandb.log({f"loss_task_idx_{self.task_index}": loss.item()})

        return

    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them. For now hardcoded for MNIST"""

        # TODO Make this task init a function of an input config file

        # Set up tasks
        # Task 1 should just contain examples in the dataset with labels from 0-8
        task_1 = self._create_mnist_task(
            set(range(9)), self.memory_set_manager, active=True
        )
        # Task 2 should contain examples in the dataset with label 9
        task_2 = self._create_mnist_task(
            set([9]), self.memory_set_manager, active=False
        )

        return [task_1, task_2]

    def _create_mnist_task(
        self,
        target_labels: Set[int],
        memory_set_manager: MemorySetManager,
        active: bool = False,
    ) -> Task:
        index = torch.where(self.train_y in target_labels)
        train_x = self.train_x[index]
        train_y = self.train_y[index]
        test_x = self.test_x[index]
        test_y = self.test_y[index]
        task = Task(train_x, train_y, test_x, test_y, target_labels, memory_set_manager)
        task.active = active

        return task

    def _convert_torch_dataset_to_tensor(
        self, dataset: Type[TorchDataset]
    ) -> Tuple[Float[Tensor, "n f"], Float[Tensor, "n 1"]]:
        xs = torch.stack([x.flatten() for x, y in dataset])
        ys = torch.Tensor([y for x, y in dataset])

        return (xs, ys)

    def _load_dataset(
        self, dataset_path: str
    ) -> Tuple[
        Float[Tensor, "n f"],
        Float[Tensor, "n 1"],
        Float[Tensor, "m f"],
        Float[Tensor, "m 1"],
    ]:
        # Define a transform to normalize the data
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # Download and load the training data
        trainset = torchvision.datasets.MNIST(
            root=dataset_path, train=True, download=True, transform=transform
        )
        train_x, train_y = self._convert_torch_dataset_to_tensor(trainset)

        # Download and load the test data
        testset = torchvision.datasets.MNIST(
            root=dataset_path, train=False, download=True, transform=transform
        )
        test_x, test_y = self._convert_torch_dataset_to_tensor(testset)

        return train_x, train_y, test_x, test_y
