# custom_lstm.py
import tensorflow as tf
from tensorflow.keras.layers import LSTM

class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        # Remove 'time_major' from kwargs if it's passed to avoid conflict
        kwargs.pop('time_major', None)
        
        # Add any additional custom configurations or initializations if needed
        # You can add other default arguments here if needed
        
        super().__init__(*args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        # Optionally, you can customize the call function further if you need 
        # to add custom logic when the layer is called
        
        # Call the parent class's call method with necessary arguments
        return super().call(inputs, training=training, mask=mask)

    def get_config(self):
        # Ensure the custom class can be serialized/deserialized by saving its config
        config = super().get_config()
        # Add any additional configurations to the config dictionary if necessary
        return config
