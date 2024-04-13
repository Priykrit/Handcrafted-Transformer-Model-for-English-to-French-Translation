import tensorflow as tf
import pickle
from tensorflow import keras
from keras.layers import TextVectorization
import numpy as np
import gradio as gr

def masked_loss(label, pred):
    mask = label != 0

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    "Custom learning rate for Adam optimizer"
    def __init__(self, key_dim, warmup_steps=4000):
        super().__init__()
        self.key_dim = key_dim
        self.warmup_steps = warmup_steps
        self.d = tf.cast(self.key_dim, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        # to make save and load a model using custom layer possible0
        config = {
            "key_dim": self.key_dim,
            "warmup_steps": self.warmup_steps,
        }
        return config


def pos_enc_matrix(L, d, n=10000):
    """Create positional encoding matrix

    Args:
        L: Input dimension (length)
        d: Output dimension (depth), even only
        n: Constant for the sinusoidal functions

    Returns:
        numpy matrix of floats of dimension L-by-d. At element (k,2i) the value
        is sin(k/n^(2i/d)) while at element (k,2i+1) the value is cos(k/n^(2i/d))
    """
    assert d % 2 == 0, "Output dimension needs to be an even integer"
    d2 = d//2
    P = np.zeros((L, d))
    k = np.arange(L).reshape(-1, 1)     # L-column vector
    i = np.arange(d2).reshape(1, -1)    # d-row vector
    denom = np.power(n, -i/d2)          # n**(-2*i/d)
    args = k * denom                    # (L,d) matrix
    P[:, ::2] = np.sin(args)
    P[:, 1::2] = np.cos(args)
    return P


class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self,sequence_length,vocab_size,embed_dim,**kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # embedding
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size,output_dim=embed_dim,mask_zero=True
        )
        matrix = pos_enc_matrix(sequence_length,embed_dim)
        self.position_embeddings = tf.constant(matrix,dtype='float32')

    def call(self,inputs):
        embedded_tokens = self.token_embeddings(inputs)
        return embedded_tokens+self.position_embeddings

    def compute_mask(self, *args, **kwargs):
        return self.token_embeddings.compute_mask(*args,**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length':self.sequence_length,
            'vocab_size': self.vocab_size,
            'embed_dim':self.embed_dim,
        })
        return config


with open('vectorize.pickle','rb') as fp:
    data = pickle.load(fp)

eng_vectorizer = TextVectorization.from_config(data['engvec_config'])
eng_vectorizer.set_weights(data['engvec_weights'])
fra_vectorizer = TextVectorization.from_config(data['fravec_config'])
fra_vectorizer.set_weights(data['fravec_weights'])


# Load the trained model
custom_objects = {"PositionalEmbedding": PositionalEmbedding,
                  "CustomSchedule": CustomSchedule,
                  "masked_loss": masked_loss,
                  "masked_accuracy": masked_accuracy}
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model("eng-fra-transformer.h5")

# training parameters used
seq_len = 20
vocab_size_en = 10000
vocab_size_fr = 20000

def translate(sentence):
    """Create the translated sentence"""
    enc_tokens = eng_vectorizer([sentence])
    lookup = list(fra_vectorizer.get_vocabulary())
    start_sentinel, end_sentinel = "[start]", "[end]"
    output_sentence = [start_sentinel]
    # generate the translated sentence word by word
    for i in range(seq_len):
        vector = fra_vectorizer([" ".join(output_sentence)])
        assert vector.shape == (1, seq_len+1)
        dec_tokens = vector[:, :-1]
        assert dec_tokens.shape == (1, seq_len)
        pred = model([enc_tokens, dec_tokens])
        assert pred.shape == (1, seq_len, vocab_size_fr)
        word = lookup[np.argmax(pred[0, i, :])]
        output_sentence.append(word)
        if word == end_sentinel:
            break
    return ' '.join(output_sentence[1:-1])

with gr.Blocks() as demo:
    name = gr.Textbox(label="English text")
    output = gr.Textbox(label="Translated text")
    translate_btn = gr.Button("To French")
    translate_btn.click(fn=translate, inputs=name, outputs=output, api_name="translate")



if __name__ == "__main__":
    demo.launch()