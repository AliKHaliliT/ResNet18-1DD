import tensorflow as tf
from typing import Union


class HardSwish(tf.keras.layers.Layer):

    """

    HardSwish activation function from the paper "Searching for MobileNetV3"
    Link: https://arxiv.org/abs/1905.02244

    The asset was sourced from the MobileViViT Repository.
    Link: https://github.com/AliKHaliliT/MobileViViT

    """

    def __init__(self, **kwargs) -> None:

        """

        Constructor of the HardSwish activation function.
        
        
        Parameters
        ----------
        None.

        
        Returns
        -------
        None.
        
        """
        
        super().__init__(**kwargs)


    def build(self, input_shape: Union[tf.TensorShape, tuple[int, ...]]) -> None:

        """

        Build method of the Embedding2D layer.


        Parameters
        ----------  
        input_shape : tf.TensorShape or tuple
            Shape of the input tensor.

            
        Returns
        -------
        None.

        """

        super().build(input_shape)
    

    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the HardSwish activation function.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.

        
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.
        
        """


        return tf.multiply(X, tf.divide(tf.nn.relu6(tf.add(X, 3.0)), 6.0))