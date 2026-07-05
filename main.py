from data.dataloader import load_MNIST
from src.visualization import visualize_base_representation


def main():
    data, targets = load_MNIST()
   # visualize_base_representation(data[0], targets[0])

if __name__ == '__main__':
    main()