ACTIVATIONS = [
    "ReLU",
    "RReLU",
    "PReLU",
    "ELU",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "Sigmoid",
    "GELU",
]

LOSSES = [
    "MAPELoss",
    "SMAPELoss",
    "MASELoss",
    "NormalizedDeviationLoss",
    "MSELoss",
    "L1Loss",
    "SmoothL1Loss",
    "BCEWithLogitsLoss",
    "BCELoss",
    "CrossEntropyLoss",
    "NLLLoss",
    "PoissonNLLLoss",
    "KLDivLoss",
    "MarginRankingLoss",
    "HingeEmbeddingLoss",
    "MultiLabelMarginLoss",
    "CosineEmbeddingLoss",
    "MultiMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
]

OPTIMIZERS = ["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta", "AdamW"]

_DEPRECATED_VAE2_BLOCKS = (
    "GenericVAE2",
    "TrendVAE2",
    "SeasonalityVAE2",
    "VAE2",
    "WaveletV3VAE2",
    "HaarWaveletV3VAE2",
    "DB2WaveletV3VAE2",
    "DB3WaveletV3VAE2",
    "DB4WaveletV3VAE2",
    "DB10WaveletV3VAE2",
    "DB20WaveletV3VAE2",
    "Coif1WaveletV3VAE2",
    "Coif2WaveletV3VAE2",
    "Coif3WaveletV3VAE2",
    "Coif10WaveletV3VAE2",
    "Symlet2WaveletV3VAE2",
    "Symlet3WaveletV3VAE2",
    "Symlet10WaveletV3VAE2",
    "Symlet20WaveletV3VAE2",
)

DEPRECATED_BLOCKS = {
    block_name: block_name.replace("VAE2", "VAE")
    for block_name in _DEPRECATED_VAE2_BLOCKS
}


def get_deprecated_block_message(block_name: str) -> str:
    replacement = DEPRECATED_BLOCKS[block_name]
    return (
        f"Block type '{block_name}' has been deprecated and removed. "
        f"Use '{replacement}' instead."
    )


BLOCKS = [
    "Generic",
    "NHiTSGenericRoot",
    "BottleneckGeneric",
    "GenericAE",
    "BottleneckGenericAE",
    "GenericAEBackcast",
    "GenericAEBackcastAE",
    "Trend",
    "TrendAE",
    "Seasonality",
    "SeasonalityAE",
    "AutoEncoder",
    "AutoEncoderAE",
    # VAE block (RootBlock backbone with reparameterization trick)
    "VAE",
    # Learned-Gate (LG) AE blocks
    "GenericAELG",
    "BottleneckGenericAELG",
    "TrendAELG",
    "SeasonalityAELG",
    "AutoEncoderAELG",
    "GenericAEBackcastAELG",
    # Variational AE (VAE) blocks
    "GenericVAE",
    "BottleneckGenericVAE",
    "TrendVAE",
    "SeasonalityVAE",
    "AutoEncoderVAE",
    "GenericAEBackcastVAE",
    # V3 Wavelet blocks (orthonormal DWT basis) — generic base class + family wrappers
    "WaveletV3",
    "HaarWaveletV3",
    "DB2WaveletV3",
    "DB3WaveletV3",
    "DB4WaveletV3",
    "DB10WaveletV3",
    "DB20WaveletV3",
    "Coif1WaveletV3",
    "Coif2WaveletV3",
    "Coif3WaveletV3",
    "Coif10WaveletV3",
    "Symlet2WaveletV3",
    "Symlet3WaveletV3",
    "Symlet10WaveletV3",
    "Symlet20WaveletV3",
    # TrendWavelet family — merged polynomial + DWT basis block
    "TrendWavelet",
    # TrendWaveletAE family — merged polynomial + DWT basis block with AE backbone (Option A)
    "TrendWaveletAE",
    "TrendWaveletAELG",
    # TrendWaveletGeneric family — three-way additive: trend + wavelet + learned generic basis
    "TrendWaveletGeneric",
    "TrendWaveletGenericAE",
    "TrendWaveletGenericAELG",
    "TrendWaveletGenericVAE",
    # V3AE Wavelet blocks (orthonormal DWT basis, AE bottleneck backbone) (Option B)
    "WaveletV3AE",
    "HaarWaveletV3AE",
    "DB2WaveletV3AE",
    "DB3WaveletV3AE",
    "DB4WaveletV3AE",
    "DB10WaveletV3AE",
    "DB20WaveletV3AE",
    "Coif1WaveletV3AE",
    "Coif2WaveletV3AE",
    "Coif3WaveletV3AE",
    "Coif10WaveletV3AE",
    "Symlet2WaveletV3AE",
    "Symlet3WaveletV3AE",
    "Symlet10WaveletV3AE",
    "Symlet20WaveletV3AE",
    # V3AELG Wavelet blocks (orthonormal DWT basis, Learned-Gate AE backbone)
    "WaveletV3AELG",
    "HaarWaveletV3AELG",
    "DB2WaveletV3AELG",
    "DB3WaveletV3AELG",
    "DB4WaveletV3AELG",
    "DB10WaveletV3AELG",
    "DB20WaveletV3AELG",
    "Coif1WaveletV3AELG",
    "Coif2WaveletV3AELG",
    "Coif3WaveletV3AELG",
    "Coif10WaveletV3AELG",
    "Symlet2WaveletV3AELG",
    "Symlet3WaveletV3AELG",
    "Symlet10WaveletV3AELG",
    "Symlet20WaveletV3AELG",
    # V3VAE Wavelet blocks (orthonormal DWT basis, variational AE backbone)
    "WaveletV3VAE",
    "HaarWaveletV3VAE",
    "DB2WaveletV3VAE",
    "DB3WaveletV3VAE",
    "DB4WaveletV3VAE",
    "DB10WaveletV3VAE",
    "DB20WaveletV3VAE",
    "Coif1WaveletV3VAE",
    "Coif2WaveletV3VAE",
    "Coif3WaveletV3VAE",
    "Coif10WaveletV3VAE",
    "Symlet2WaveletV3VAE",
    "Symlet3WaveletV3VAE",
    "Symlet10WaveletV3VAE",
    "Symlet20WaveletV3VAE",
]
