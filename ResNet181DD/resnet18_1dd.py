from tensorflow.keras.saving import register_keras_serializable # type: ignore
import tensorflow as tf
from .assets.layers.conv1d_layer import Conv1DLayer
from .assets.blocks.residual1d_d import Residual1DD
from typing import Union, Any


@register_keras_serializable()
class ResNet181DD(tf.keras.Model):

    """

    Custom ResNet18 Network with Conv1D Convolutions and Residual blocks. 
    
    The stem block is the ResNet-C block from the paper "Bag of Tricks for Image Classification with Convolutional Neural Networks"
    Link: https://arxiv.org/abs/1812.01187

    """

    def __init__(self, units: int = 256, **kwargs) -> None:

        """

        Constructor of the ResNet181DD network.
        

        Parameters
        ----------
        units : int, optional
            Number of units in the head. The default value is `256`.


        Returns
        -------
        None.
        
        """

        if not isinstance(units, int) or units <= 0:
            raise ValueError(f"units must be a positive integer. Received: {units} with type {type(units)}")


        super().__init__(**kwargs)

        self.units = units

        # Stem
        self.stem = Conv1DLayer(filters=32, kernel_size=(3, ), strides=(2, ), 
                                padding="same", use_bias=False, 
                                normalization="batch_norm", activation="relu")
        self.stem1 = Conv1DLayer(filters=32, kernel_size=(3, ), strides=(1, ), 
                                 padding="same", use_bias=False, 
                                 normalization="batch_norm", activation="relu")
        self.stem2 = Conv1DLayer(filters=64, kernel_size=(3, ), strides=(1, ), 
                                 padding="same", use_bias=False, 
                                 normalization="batch_norm", activation="relu")
        self.stem3 = tf.keras.layers.MaxPool1D(pool_size=(2, ), strides=(2, ))

        # Backbone
        ## Stage 1
        self.block = Residual1DD(filters=64, strides=(1, ))
        ## Stage 2
        self.block1 = Residual1DD(filters=128, strides=(1, ))
        self.block2 = Residual1DD(filters=128, strides=(1, ))
        ## Stage 3
        self.block3 = Residual1DD(filters=256, strides=(1, ))
        self.block4 = Residual1DD(filters=256, strides=(1, ))
        ## Stage 4
        self.block5 = Residual1DD(filters=512, strides=(1, ))
        self.block6 = Residual1DD(filters=512, strides=(1, ))

        # Head
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.embedding = tf.keras.layers.Dense(units=self.units)


    def build(self, input_shape: Union[tf.TensorShape, tuple[int, int, int]]) -> None:

        """

        Build method of the ResNet181DD network.


        Parameters
        ----------  
        input_shape : tf.TensorShape or tuple
            Shape of the input tensor.

            
        Returns
        -------
        None.

        """

        super().build(input_shape)

        # Stem
        self.stem.build(input_shape)
        input_shape = self.stem.compute_output_shape(input_shape)
        self.stem1.build(input_shape)
        input_shape = self.stem1.compute_output_shape(input_shape)
        self.stem2.build(input_shape)
        input_shape = self.stem2.compute_output_shape(input_shape)
        self.stem3.build(input_shape)
        input_shape = self.stem3.compute_output_shape(input_shape)

        # Backbone
        ## Stage 1
        self.block.build(input_shape)
        input_shape = self.block.compute_output_shape(input_shape)
        self.block1.build(input_shape)
        input_shape = self.block1.compute_output_shape(input_shape)
        ## Stage 2
        self.block2.build(input_shape)
        input_shape = self.block2.compute_output_shape(input_shape)
        self.block3.build(input_shape)
        input_shape = self.block3.compute_output_shape(input_shape)
        ## Stage 3
        self.block4.build(input_shape)
        input_shape = self.block4.compute_output_shape(input_shape)
        self.block5.build(input_shape)
        input_shape = self.block5.compute_output_shape(input_shape)
        ## Stage 4
        self.block6.build(input_shape)
        input_shape = self.block6.compute_output_shape(input_shape)

        # Head
        self.pooling.build(input_shape)
        input_shape = self.pooling.compute_output_shape(input_shape)
        self.embedding.build(input_shape)


    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        """

        Call method of the ResNet181DD network.
        

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        
        
        Returns
        -------
        outputs : tf.Tensor
            Output tensor.
        
        """

        # Stem
        inputs_transformed = self.stem(inputs)
        inputs_transformed = self.stem1(inputs_transformed)
        inputs_transformed = self.stem2(inputs_transformed)
        inputs_transformed = self.stem3(inputs_transformed)

        # Backbone
        ## Stage 1
        inputs_transformed = self.block(inputs_transformed)
        inputs_transformed = self.block1(inputs_transformed)
        ## Stage 2
        inputs_transformed = self.block2(inputs_transformed)
        inputs_transformed = self.block3(inputs_transformed)
        ## Stage 3
        inputs_transformed = self.block4(inputs_transformed)
        inputs_transformed = self.block5(inputs_transformed)
        ## Stage 4
        inputs_transformed = self.block6(inputs_transformed)

        # Head
        inputs_transformed = self.pooling(inputs_transformed)
        outputs = self.embedding(inputs_transformed)


        return outputs
    

    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the ResNet181DD network.
        
        
        Parameters
        ----------
        None.
        
        
        Returns
        -------
        config : dict
            Configuration of the ResNet181DD network.
        
        """

        config = super().get_config()

        config.update({
            "units": self.units,
        })


        return config
    

    @classmethod
    def from_config(cls, config):

        """

        Method to set the configuration of the ResNet181DD network.
        
        
        Parameters
        ----------
        config : dict
            Configuration of the ResNet181DD network.
        
        
        Returns
        -------
        model : tf.keras.Model
            The Loaded ResNet181DD network.
        
        """


        return cls(**config)
    

    def get_build_config(self) -> dict[str, Any]:
        

        """

        Method to get the configuration of the ResNet181DD network.
        
        
        Parameters
        ----------
        None.
        
        
        Returns
        -------
        config : dict
            Configuration of the ResNet181DD network.
        
        """

        config = super().get_config()

        config.update({
            "units": self.units,
        })


        return config


    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> "ResNet181DD":

        """

        Method to set the build configuration of the ResNet181DD network.
        
        
        Parameters
        ----------
        config : dict
            Configuration of the ResNet181DD network.
        
        
        Returns
        -------
        model : tf.keras.Model
            The Loaded ResNet181DD network.
        
        """


        return cls(**config)