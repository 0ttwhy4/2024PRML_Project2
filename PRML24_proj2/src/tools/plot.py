import matplotlib.pyplot as plt

def plot_curve(training_loss, training_accuracy, val_loss, val_accuracy, num_epochs, save_dir):
    epochs = range(1, num_epochs+1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(num_epochs/10, 10), 10))
    
    ax1.plot(epochs, training_loss, label='Training Loss', marker='o')
    ax1.plot(epochs, val_loss, label='Validation Loss', marker='o')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, training_accuracy, label='Training Accuracy', marker='o')
    ax2.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_dir)