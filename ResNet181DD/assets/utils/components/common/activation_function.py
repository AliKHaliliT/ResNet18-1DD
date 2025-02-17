import tensorflow as tf
from ....activations.hard_swish import HardSwish
from ....activations.mish import Mish


def activation_function(activation: str) -> tf.keras.layers.Layer:

    """

    Method to get the activation function.
    
    The asset was sourced from the MobileViViT Repository.
    Link: https://github.com/AliKHaliliT/MobileViViT


    Parameters
    ----------
    activation : str
        Activation function name.
            The options are:
                `"relu"`
                    Rectified Linear Unit activation function.
                `"leaky_relu"`
                    Leaky Rectified Linear Unit activation function.
                `"hard_swish"`
                    Hard Swish activation function.
                `"mish"`
                    Mish activation function.

    
    Returns
    -------
    activation : tf.keras.layers.Layer
        Activation function.

    """

    if activation not in ["relu", "leaky_relu", "hard_swish", "mish"]:
        raise ValueError(f"Unknown activation function. Received: {activation} with type {type(activation)}")
    

    if activation == "relu":
        return tf.keras.layers.ReLU()
    elif activation == "leaky_relu":
        return tf.keras.layers.LeakyReLU()
    elif activation == "hard_swish":
        return HardSwish()
    elif activation == "mish":
        return Mish()