import tensorflow as tf
from typing import Optional, Union, Any
from ..utils.components.common.activation_function import activation_function


@tf.keras.utils.register_keras_serializable()
class Conv1DLayer(tf.keras.layers.Layer):

    """
    
    A simple custom Conv layer containing Conv1D, a Normalizarion layer and an Activation at the end. 

    The asset was sourced from the MobileViViT Repository.
    Link: https://github.com/AliKHaliliT/MobileViViT
    
    """

    def __init__(self, filters: int, kernel_size: tuple[int], 
                 strides: tuple[int], padding: str, 
                 use_bias: bool = True, normalization: Optional[str] = None, 
                 activation: Optional[str] = None, **kwargs) -> None:

        """

        Constructor of the Conv layer.
        
        
        Parameters
        ----------
        filters : int
            Number of filters in the convolutional layer.

        kernel_size : tuple
            Kernel size of the convolutional layer.

        strides : tuple
            Strides of the convolutional layer.

        padding : str
            Padding of the convolutional layer.
                The options are:
                    `"valid"`
                        No padding.
                    `"same"`
                        Padding with zeros.

        use_bias : bool
            Bias term for the layer. The default is `True`.

        normalization : str, optional
            Normalization of the layer. The default value is `None`.
                The options are:
                    `"batch_norm"`
                        Batch normalization.
                    `"layer_norm"`
                        Layer normalization.

        activation : str, optional
            Activation function of the layer. The default value is `None`.
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
        None.
        
        """

        if not isinstance(filters, int) or filters < 0:
            raise ValueError(f"filters must be a non-negative integer. Received: {filters} with type {type(filters)}")
        if len(kernel_size) != 1 or not all(isinstance(k, int) and k > 0 for k in kernel_size):
            raise ValueError(f"kernel_size must be a tuple of a single positive integer. Received: {kernel_size} with type {type(kernel_size)}")
        if len(strides) != 1 or not all(isinstance(s, int) and s > 0 for s in strides):
            raise ValueError(f"strides must be a tuple of a single positive integer. Received: {strides} with type {type(strides)}")
        if not isinstance(padding, str) or padding not in ["valid", "same"]:
            raise ValueError(f"padding must be either 'valid' or 'same'. Received: {padding} with type {type(padding)}")
        if not isinstance(use_bias, bool):
            raise TypeError(f"use_bias must be a boolean. Received: {use_bias} with type {type(use_bias)}")
        if normalization not in [None, "batch_norm", "layer_norm"]:
            raise ValueError(f"normalization must be one of 'batch_norm', 'layer_norm', or None. Received: {normalization} with type {type(normalization)}")
        if activation not in [None, "relu", "leaky_relu", "hard_swish", "mish"]:
            raise ValueError(f"activation must be one of 'relu', 'leaky_relu', 'hard_swish', 'mish', or None. Received: {activation} with type {type(activation)}")
        

        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.normalization = normalization
        self.activation = activation

        self.convolution = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, 
                                                  strides=self.strides, padding=self.padding, 
                                                  use_bias=self.use_bias)
        if self.normalization == "batch_norm":
            self.normalize = tf.keras.layers.BatchNormalization()
        elif self.normalization == "layer_norm":
            self.normalize = tf.keras.layers.LayerNormalization()
        if self.activation is not None:
            self.activate = activation_function(self.activation)


    def build(self, input_shape: Union[tf.TensorShape, tuple[int, int, int]]) -> None:

        """

        Build method of the Conv layer.


        Parameters
        ----------  
        input_shape : tf.TensorShape or tuple
            Shape of the input tensor.


        Returns
        -------
        None.

        """

        super().build(input_shape)

        self.convolution.build(input_shape)
                                

    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the Conv layer.


        Parameters
        ----------
        X : tf.Tensor
            Input tensor.
        
            
        Returns
        -------
        X_transformed : tf.Tensor
            Output tensor.

        """

        X_transformed = self.convolution(X)
        if self.normalization:
            X_transformed = self.normalize(X_transformed)


        return self.activate(X_transformed) if self.activation is not None else X_transformed
    

    def compute_output_shape(self, input_shape: Union[tf.TensorShape, tuple[int, int, int]]) -> Union[tf.TensorShape, tuple[int, int, int]]:

        """
        
        Method to compute the output shape of the Conv layer.


        Parameters
        ----------
        input_shape : tf.TensorShape or tuple
            Shape of the input tensor. 


        Returns
        -------
        output_shape : tuple
            Shape of the output tensor after applying the Conv1D layer. 
        
        """


        return self.convolution.compute_output_shape(input_shape)


    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the Conv layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the Conv layer.

        """
            
        config = super().get_config()

        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "normalization": self.normalization,
            "activation": self.activation,
        })


        return config
    

    def get_build_config(self) -> dict[str, Any]:

        """

        Method to get the build configuration of the Conv layer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the Conv layer.

        """
            
        config = super().get_config()

        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "normalization": self.normalization,
            "activation": self.activation,
        })


        return config