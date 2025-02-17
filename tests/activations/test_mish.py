import unittest
import tensorflow as tf
from ResNet181DD.assets.activations.mish import Mish


class TestMish(unittest.TestCase):

    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1, 1, 1, 3))

        # Act
        output = Mish()(input_tensor)

        # Assert
        self.assertEqual(output.shape, input_tensor.shape)


if __name__ == "__main__":
    unittest.main()