class MLP(tf.keras.layers.Layer):
    """
        this class is a implementation of the mlp block described in the swin transformer paper, which contains
        2 fully connected layer with GelU activation.
    """
    def __init__(self, config: ConfigDict, **kwargs):
        """
            Params:
                input_neurons(dtype: int)   : input dimension for the mlp block, it needed only for .summary() method.
                input_dims(dtype: int)  : number of neurons in the hidden
                                              layer(fully connected layer).
                output_neurons(dtype: iny)  ; number of neurons in the last
                                              layer(fully connected layer) of mlp.
                act_type(type: str)         ; type of activation needed. in paper, GeLU is used.
                dropout_rate(dtype: float)  : dropout rate in the dropout layer.
                prefix(type: str)           : used for the naming the layers.
        """
        super(MLP, self).__init__(**kwargs)
        in_features = config.in_features
        hidden_features = config.hidden_features
        mlp_drop = config.mlp_drop

        self.fc1 = Dense(hidden_features, name=f'mlp/fc1', bias_initializer=keras.initializers.RandomNormal(stddev=1e-6))
        self.fc2 = Dense(in_features, name=f'mlp/fc2', bias_initializer=keras.initializers.RandomNormal(stddev=1e-6))
        self.drop = Dropout(mlp_drop)

    def call(self, x):
        x = self.fc1(x)
        x = tf.keras.activations.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
