import matplotlib.pyplot as plt

def visualize_base_representation(image, label):

    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label.item()}")
    plt.axis('off')
    plt.show()
