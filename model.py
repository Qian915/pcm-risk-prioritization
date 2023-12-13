import tensorflow as tf
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm_a = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_b = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_a = layers.Dropout(rate)
        self.dropout_b = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout_a(attn_output)
        out_a = self.layernorm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output)
        return self.layernorm_b(out_a + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, act_x):
        maxlen = tf.shape(act_x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        act_x = self.token_emb(act_x)
        return act_x + positions
    

def get_prediction_model(max_case_length, vocab_size, embed_dim = 36, num_heads = 4, ff_dim = 64):
    # input features: activity & time
    input_act = layers.Input(shape=(max_case_length,))
    input_time = layers.Input(shape=(1,))

    x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(input_act)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x_t = layers.Dense(32, activation="relu")(input_time)
    x = layers.Concatenate()([x, x_t])
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    output_act = layers.Dense(vocab_size, activation="softmax", name='output_act')(x)  # softmax for activity prediction
    output_time = layers.Dense(1, activation="linear", name='output_time')(x)   # linear for time prediction
    transformer = tf.keras.Model(inputs=[input_act, input_time], outputs=[output_act, output_time])
    return transformer