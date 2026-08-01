"""
Exact mirror of the model architecture defined in the Flickr8K notebook.

Cell 20 of the notebook:

    input1 = Input(shape=(1920,))
    input2 = Input(shape=(max_length,))

    img_features = Dense(256, activation='relu')(input1)
    img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)

    sentence_features = Embedding(vocab_size, 256, mask_zero=False)(input2)
    merged = concatenate([img_features_reshaped, sentence_features], axis=1)
    sentence_features = LSTM(256)(merged)
    x = Dropout(0.5)(sentence_features)
    x = add([x, img_features])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(vocab_size, activation='softmax')(x)

    caption_model = Model(inputs=[input1, input2], outputs=output)
    caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

The encoder is DenseNet201 with the final classification layer removed
(`fe = Model(inputs=model.input, outputs=model.layers[-2].output)`),
producing a 1920-dim feature vector.
"""

from __future__ import annotations

from typing import Tuple

# Heavy import — done lazily inside functions to keep module import light.
def _tf():
    import tensorflow as tf  # noqa: WPS433
    return tf


# Default training hyper-params taken from the notebook.
DEFAULT_IMG_SIZE: int = 224
DEFAULT_EMBED_DIM: int = 256
DEFAULT_LSTM_UNITS: int = 256
DEFAULT_DENSE_UNITS: int = 128
DEFAULT_DROPOUT: float = 0.5
ENCODER_FEATURE_DIM: int = 1920  # DenseNet201 penultimate layer


def build_encoder(img_size: int = DEFAULT_IMG_SIZE) -> "object":
    """
    DenseNet201 encoder with the classification head removed.

    Mirrors notebook cell 15:
        model = DenseNet201()
        fe = Model(inputs=model.input, outputs=model.layers[-2].output)
    """
    tf = _tf()
    from tensorflow.keras.applications import DenseNet201
    from tensorflow.keras.models import Model

    base = DenseNet201(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        pooling="avg",
    )
    # base.output -> (None, 1920)
    return Model(inputs=base.input, outputs=base.output, name="densenet201_encoder")


def build_decoder(
    vocab_size: int,
    max_length: int,
    embed_dim: int = DEFAULT_EMBED_DIM,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    dense_units: int = DEFAULT_DENSE_UNITS,
    dropout: float = DEFAULT_DROPOUT,
) -> "object":
    """
    LSTM decoder — exact replica of notebook cell 20.

    Inputs:
        - input1: image feature vector (None, 1920)
        - input2: partial caption token sequence (None, max_length)
    Output:
        - softmax over vocab (None, vocab_size)
    """
    tf = _tf()
    from tensorflow.keras.layers import (
        Add,
        Concatenate,
        Dense,
        Dropout,
        Embedding,
        Input,
        LSTM,
        Reshape,
    )
    from tensorflow.keras.models import Model

    input1 = Input(shape=(ENCODER_FEATURE_DIM,), name="image_features")
    input2 = Input(shape=(max_length,), name="caption_tokens")

    img_features = Dense(embed_dim, activation="relu", name="img_fc")(input1)
    img_features_reshaped = Reshape((1, embed_dim), name="img_reshape")(img_features)

    sentence_features = Embedding(
        vocab_size, embed_dim, mask_zero=False, name="caption_embedding"
    )(input2)
    merged = Concatenate(axis=1, name="img_caption_concat")(
        [img_features_reshaped, sentence_features]
    )
    sentence_features = LSTM(lstm_units, name="decoder_lstm")(merged)

    x = Dropout(dropout, name="dropout_1")(sentence_features)
    x = Add(name="residual_add")([x, img_features])
    x = Dense(dense_units, activation="relu", name="fc_128")(x)
    x = Dropout(dropout, name="dropout_2")(x)
    output = Dense(vocab_size, activation="softmax", name="vocab_softmax")(x)

    model = Model(inputs=[input1, input2], outputs=output, name="caption_decoder")
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def build_full_pipeline(
    vocab_size: int,
    max_length: int,
    img_size: int = DEFAULT_IMG_SIZE,
) -> Tuple[object, object]:
    """Return (encoder, decoder)."""
    return build_encoder(img_size), build_decoder(vocab_size, max_length)
