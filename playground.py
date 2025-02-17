import os
# Reduce TensorFlow log level for minimal logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'


from ResNet181DD import ResNet181DD


model = ResNet181DD()
model.build((None, 128, 32))
model.summary()