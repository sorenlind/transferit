"""Custom callbacks used during training."""

from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


class PlotLosses(Callback):
    """Simple callback for plotting losses to a file."""

    def __init__(self, output_folder, model_name):
        self.output_folder = output_folder
        self.model_name = model_name

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.i += 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.x, self.losses, label="loss")
        ax.plot(self.x, self.val_losses, label="val_loss")
        ax.legend()
        fig.savefig(str(self.output_folder / f"{self.model_name}_loss.png"))
        plt.close(fig)
