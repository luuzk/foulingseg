from math import ceil

from tensorflow.keras import applications as ka
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, regularizers


### Semantic Segmentation models

def efficientnet_unet(b=0, input_shape=None, **kwargs) -> models.Model:
    '''Input image must be in range [0, 255]
    '''
    b2net: dict[int, models.Model] = {
        0: ka.EfficientNetB0,
        1: ka.EfficientNetB1,
        2: ka.EfficientNetB2,
        3: ka.EfficientNetB3,
        4: ka.EfficientNetB4,
        5: ka.EfficientNetB5,
        6: ka.EfficientNetB6,
        7: ka.EfficientNetB7,
    }
    assert b in b2net.keys()

    # EfficientNetBx with pre-trained weights
    backbone = b2net[b](input_shape=input_shape, include_top=False, 
        weights='imagenet')
    
    inputs = backbone.input              # 224x224x   3
    skip_layer_names = ('rescaling',     # 224x224x   3
                        'block1',        # 112x112x  16
                        'block2',        #  56x 56x  24
                        'block3',        #  28x 28x  40
                        'block5',        #  14x 14x 112
                        'block7')        #   7x  7x 320

    # Extract skip connections and freeze batch norm layers
    skips = extract_skips(backbone, skip_layer_names)
    backbone = models.Model(inputs, skips, name=f"EfficientNetB{b}")
    freeze_bn_layers(backbone)

    return append_decoder(backbone, **kwargs)


def baseline_unet(encoder_filters=64, stages=5, input_shape=None, dropout=0, **kwargs):
    assert encoder_filters % 8 == 0 # (tensor core acceleration)
    assert int(stages) > 0

    conv_kwargs = {
        "padding":"same",
        "use_bias":False,
        "kernel_initializer":"he_normal"
    }

    ### Model input
    if input_shape is None:
        inputs = layers.Input((None, None, 3))
    else:
        assert len(input_shape) == 3
        inputs = layers.Input(input_shape)
    x = inputs
    x = layers.Rescaling(1./255.)(x) # to range [0, 1] # pylint: disable=no-member

    # Build contraction path
    skips = []
    for stage in range(int(stages)):
        prefix = f"encoder_{stage}_"
        
        ### Convolution
        # 2x (Conv -> BN -> ReLU6)
        x = layers.Conv2D(
            encoder_filters, 
            kernel_size=3, 
            **conv_kwargs,
            name=prefix + "conv_0")(x)
        x = layers.BatchNormalization(
            momentum=0.99,
            name=prefix + "conv_0_BN")(x)
        x = layers.ReLU(name=prefix + "conv_0_relu")(x)

        x = layers.Conv2D(
            encoder_filters, 
            kernel_size=3, 
            **conv_kwargs,
            name=prefix + "conv_1")(x)
        x = layers.BatchNormalization(
            momentum=0.99,
            name=prefix + "conv_1_BN")(x)
        x = layers.ReLU(name=prefix + "conv_1_relu")(x)

        skips.append(x)
        encoder_filters *= 2

        ### Downsampling
        if stage < stages - 1:
            x = layers.MaxPooling2D(
                pool_size=2,
                padding="same",
                name=prefix + "max_pool")(x)
    
    if dropout:
        x = layers.Dropout(dropout)(x)

    # Skip connections to model
    backbone = models.Model(inputs, skips, name="plain_encoder")

    return append_decoder(backbone, **kwargs)

### Building blocks

def append_decoder(backbone: models.Model,
                   num_classes: int,
                   preprocessing=None,
                   encoder_freeze=True,
                   decoder_filters=320,
                   aux_classifier=False,
                   output_bias=None,
                   with_softmax=True,
                   decoder_filter_decay=2,
                   **kwargs) -> models.Model:
    assert decoder_filters % 8 == 0
    assert decoder_filter_decay > 1 

    # Call encapsulated backbone to get skip outputs (ascending)
    if preprocessing:
        x = preprocessing(backbone.input)
    else:
        x = backbone.input
    *skips, x = backbone(x)#, training=False) # deactive dropout & BN
    x = layers.ReLU(6., name="enc_out_relu")(x)

    # Build decoder with skip connections
    for i, skip in reversed(list(enumerate(skips))):
        x = DecoderUpsamplingBlock(decoder_filters, i, **kwargs)(x, skip)

        # Reduce filters to multiple of 8 (tensor core acceleration)
        decoder_filters = int(8 * ceil(decoder_filters / decoder_filter_decay / 8))

    # Decoder head
    x = layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        padding='same',
        use_bias=True,
        kernel_initializer='he_normal',
        name='out_conv')(x)
    x = layers.Softmax(name="mask", dtype='float32')(x) if with_softmax \
        else layers.Activation("linear", name="mask", dtype='float32')(x)

    # Terminal classification head
    if aux_classifier == "pool_dist":
        y = layers.GlobalAveragePooling2D(name="pool")(x)

    # Freeze encoder
    if encoder_freeze:
        backbone.trainable = False

    # Build model
    model = models.Model(backbone.input, [x, y] if aux_classifier else x)

    # Set initial output bias
    if output_bias is not None:
        K.set_value(model.get_layer('out_conv').bias, output_bias)

    return model


# pylint: disable=unused-argument
def DecoderUpsamplingBlock(filters,
                           block_id,
                           upsampling="bilinear",
                           decoder_skips=False,
                           squeeze_excitate=False,
                           squeeze_excitate_ratio=8,
                           l2=0.,
                           **kwargs) -> layers.Layer:
    '''Decoder block with 1x upsampling and 2x convolutional layers'''
    prefix = f"decoder_{block_id}_"
    conv_kwargs = {
        "padding":"same",
        "use_bias":False,
        "kernel_initializer":"he_normal"
    }

    def _decoder_layer(inputs, skip):
        x = inputs

        ### Upsampling
        assert upsampling in {"nearest", "bilinear", "bilinear_additive", "transposed"}
        if upsampling in {"nearest", "bilinear"}:
            x = layers.UpSampling2D(
                size=2,
                interpolation=upsampling,
                name=prefix + upsampling + "_up")(x)
        elif upsampling == "bilinear_additive":
            new_shape = K.concatenate([K.shape(x)[:-1], [K.int_shape(x)[-1] // 4], [4]])
            x = K.reshape(x, new_shape)
            x = K.sum(x, axis=-1)
        else:
            x = layers.Conv2DTranspose(
                filters,
                kernel_size=2,
                strides=2,
                **conv_kwargs,
                name=prefix + "transposed_up")(x)
        
        ### Residual connections
        # Residual U-Net connection
        x = layers.Concatenate(name=prefix + "concat")([x, skip])

        # Residual decoder connection
        if decoder_skips:
            y = layers.Conv2D(
                filters,
                kernel_size=1,
                **conv_kwargs,
                name=prefix + "res_conv")(x)

        ### Convolution
        # 2x (Conv -> BN -> ReLU6)
        x = layers.Conv2D(
            filters,
            kernel_size=3,
            kernel_regularizer=regularizers.l2(l2),
            **conv_kwargs,
            name=prefix + "conv_0")(x)
        x = layers.BatchNormalization(
            momentum=0.99,
            name=prefix + "conv_0_BN")(x)
        x = layers.ReLU(6., name=prefix + "conv_0_relu")(x)

        x = layers.Conv2D(
            filters,
            kernel_size=3,
            kernel_regularizer=regularizers.l2(l2),
            **conv_kwargs,
            name=prefix + "conv_1")(x)
        x = layers.BatchNormalization(
            momentum=0.99,
            name=prefix + "conv_1_BN")(x)
        x = layers.ReLU(6., name=prefix + "conv_1_relu")(x)

        ### Channelwise attention
        # Squeeze Excitation Block
        if squeeze_excitate:
            # Squeeze
            z = layers.GlobalAveragePooling2D(
                name=prefix + "se_avg_pool")(x)
            # Excitate: FC -> ReLU -> FC -> Sigmoid
            z = layers.Dense(
                max(int(filters // squeeze_excitate_ratio), 1),
                name=prefix + "se_dense_0")(z)
            z = layers.ReLU(6., name=prefix + "se_dense_0_relu")(z)

            z = layers.Dense(
                filters, 
                name=prefix + "se_dense_1")(z)
            z = layers.Activation("sigmoid", name=prefix + "se_dense_1_sigmoid")(z)
            # Scale input
            x = layers.Multiply(name=prefix + "se_mult")([x, z])

        # Dropout (see EfficientNet, stochastic depth)
        # drop_connect_rate * b / blocks,
        if decoder_skips:
            x = layers.Add(name=prefix + "res_add")([x, y])

        return x            
    return _decoder_layer

### Model-related methods

def extract_skips(backbone: models.Model, skip_layer_names, print_skips=True):
    # Find last layer that matches name condition
    layer_names = [layer.name for layer in backbone.layers][::-1]
    skip_layer_names = [next(n for n in layer_names if name in n) 
        for name in skip_layer_names]
    
    if print_skips:
        print("\n  ".join(["Skip layers:"] + skip_layer_names))

    # Extract skip connections
    return [backbone.get_layer(name=name).output for name in skip_layer_names]


def extract_encoder(model: models.Model) -> models.Model:
    return next(l for l in model.layers if isinstance(l, models.Model))


def layer_index_from_encoder(model: models.Model, layer_name):
    encoder = extract_encoder(model)

    # Index of first unfreezed layer
    index = 0
    if layer_name:
        index = encoder.get_layer(name=layer_name)
        index = encoder.layers.index(index)
    return index


def unfreeze_encoder(model: models.Model, fine_tune_after=0):
    encoder = extract_encoder(model)

    # Unfreeze layers but keep BN freezed
    encoder.trainable = True
    for i, layer in enumerate(encoder.layers):
        if i <= fine_tune_after or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    layer_name = encoder.get_layer(index=fine_tune_after).name
    print("Unfreezed layers after", layer_name, "of", encoder.name)


def freeze_bn_layers(model: models.Model):
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def make_feature_extractor(model: models.Model) -> models.Model:
    # Extract encoder nodes
    encoder_inp = model.input
    encoder_out = model.get_layer("enc_out_relu").input

    # Feacture extraction head on top
    encoder_out = layers.GlobalAveragePooling2D(name="embedding")(encoder_out)
    return models.Model(encoder_inp, encoder_out)


def model_from_args(func, load_weights=None, **model_kwargs) -> models.Model:
    model = func(**model_kwargs)
    assert isinstance(model, models.Model)

    if load_weights is not None:
        model.load_weights(load_weights)

    return model
