"""Deterministic plate-image preprocessing variants for PARSeq ANPR.

The variants map directly to topics in the IMP302m course (gray-level
processing, linear/non-linear filtering, morphology, restoration/wavelets,
and binary processing).  All functions return RGB PIL images because PARSeq
expects three input channels even when the useful signal is grayscale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


@dataclass(frozen=True)
class PreprocessingConfig:
    name: str
    course_topic: str
    description: str
    grayscale: bool = True
    gray_channel: str = "luma"  # luma, red, green, blue, max, hsv_v, lab_l, best_contrast
    autocontrast: bool = False
    histogram_equalization: bool = False
    percentile_low: float | None = None
    percentile_high: float | None = None
    clahe_clip_limit: float | None = None
    clahe_tile_size: int = 8
    gamma: float = 1.0
    illumination: str = "none"  # none, retinex, multiscale_retinex, homomorphic, local_norm
    illumination_sigma: float = 15.0
    deblur: str = "none"  # none, wiener_deconv, richardson_lucy
    deblur_kernel_size: int = 3
    deblur_sigma: float = 0.8
    deblur_balance: float = 0.02
    deblur_iterations: int = 3
    denoise: str = "none"  # none, gaussian, median, bilateral, wavelet_haar, nlm, wiener
    gaussian_sigma: float = 0.8
    median_ksize: int = 3
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    nlm_h: float = 3.0
    wiener_ksize: int = 3
    sharpen_alpha: float = 0.0
    sharpen_sigma: float = 1.0
    sharpen_method: str = "unsharp"  # unsharp, laplacian, dog
    sharpen_sigma_large: float = 1.6
    morphology: str = "none"  # none, close, blackhat, gradient
    morphology_ksize: int = 3
    morphology_kernel_width: int | None = None
    morphology_kernel_height: int | None = None
    morphology_strength: float = 0.6
    edge_enhancement: str = "none"  # none, sobel, dual_stroke
    edge_strength: float = 0.15
    character_isolation: str = "none"  # none, content_crop, component_mask, component_fusion
    character_margin: float = 0.08
    background_blur_sigma: float = 1.2
    pre_upscale_factor: float = 1.0
    pre_upscale_min_height: int = 0
    pre_upscale_min_width: int = 0
    pre_upscale_interpolation: str = "lanczos"  # nearest, cubic, lanczos
    adaptive_policy: str = "none"  # dataset-specific deterministic routing policy
    threshold: str = "none"  # none, otsu, adaptive
    adaptive_block_size: int = 25
    adaptive_c: int = 7
    resize_interpolation: str = "bicubic"  # applied by the benchmark transform
    resize_mode: str = "stretch"  # stretch or letterbox

    def to_dict(self) -> dict:
        return asdict(self)


RAW_CONFIG = PreprocessingConfig(
    name="raw_rgb",
    course_topic="Reference",
    description="Original RGB crop; resize and normalize only.",
    grayscale=False,
)

# This exactly reproduces preprocess_plate_image() used to train the supplied
# refinement checkpoint: grayscale -> CLAHE -> bilateral -> unsharp (1.5/-0.5).
DEFAULT_CONFIG = PreprocessingConfig(
    name="train_baseline",
    course_topic="2.1-2.2 Linear and nonlinear enhancement",
    description="Training-time baseline: gray + CLAHE + bilateral + mild unsharp mask.",
    clahe_clip_limit=2.0,
    denoise="bilateral",
    sharpen_alpha=0.5,
)


SWEEP_CONFIGS = [
    DEFAULT_CONFIG,
    RAW_CONFIG,
    PreprocessingConfig(
        name="grayscale",
        course_topic="1.1 Basic gray-level processing",
        description="Grayscale only.",
    ),
    PreprocessingConfig(
        name="autocontrast",
        course_topic="1.1 Basic gray-level processing",
        description="Global percentile-free contrast stretching.",
        autocontrast=True,
    ),
    PreprocessingConfig(
        name="hist_equalization",
        course_topic="1.1 Basic gray-level processing",
        description="Global histogram equalization.",
        histogram_equalization=True,
    ),
    PreprocessingConfig(
        name="clahe_gray",
        course_topic="2 Image enhancement",
        description="Local contrast enhancement on grayscale luminance.",
        clahe_clip_limit=2.0,
    ),
    PreprocessingConfig(
        name="clahe_lab",
        course_topic="3.10 Multichannel image recovery",
        description="Color-preserving CLAHE on the LAB luminance channel.",
        grayscale=False,
        clahe_clip_limit=2.0,
    ),
    PreprocessingConfig(
        name="clahe_gaussian",
        course_topic="2.1 Linear filtering",
        description="CLAHE followed by Gaussian noise suppression.",
        clahe_clip_limit=2.0,
        denoise="gaussian",
    ),
    PreprocessingConfig(
        name="clahe_median",
        course_topic="2.2 Nonlinear filtering",
        description="CLAHE followed by median filtering for impulse noise.",
        clahe_clip_limit=2.0,
        denoise="median",
    ),
    PreprocessingConfig(
        name="clahe_bilateral",
        course_topic="2.2 Nonlinear filtering",
        description="Training baseline without sharpening (edge-preserving denoise ablation).",
        clahe_clip_limit=2.0,
        denoise="bilateral",
    ),
    PreprocessingConfig(
        name="clahe_unsharp",
        course_topic="2.1 Spatial enhancement",
        description="Training baseline without bilateral filtering (sharpening ablation).",
        clahe_clip_limit=2.0,
        sharpen_alpha=0.5,
    ),
    PreprocessingConfig(
        name="baseline_strong_unsharp",
        course_topic="2.1 Spatial enhancement",
        description="Training baseline with stronger high-frequency emphasis.",
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=1.0,
    ),
    PreprocessingConfig(
        name="baseline_morph_close",
        course_topic="2.3 Morphological filtering",
        description="Training baseline plus a small closing operation to reconnect strokes.",
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=0.5,
        morphology="close",
    ),
    PreprocessingConfig(
        name="baseline_blackhat",
        course_topic="2.3 Morphological filtering",
        description="Training baseline plus black-hat dark-stroke enhancement.",
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=0.5,
        morphology="blackhat",
    ),
    PreprocessingConfig(
        name="baseline_gamma_0_8",
        course_topic="1.1 Gray-level transforms",
        description="Gamma brightening before the training-time pipeline.",
        gamma=0.8,
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=0.5,
    ),
    PreprocessingConfig(
        name="baseline_gamma_1_2",
        course_topic="1.1 Gray-level transforms",
        description="Gamma darkening before the training-time pipeline.",
        gamma=1.2,
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=0.5,
    ),
    PreprocessingConfig(
        name="clahe_wavelet_haar",
        course_topic="3.5 Wavelet denoising",
        description="CLAHE plus single-level Haar soft-threshold denoising.",
        clahe_clip_limit=2.0,
        denoise="wavelet_haar",
        sharpen_alpha=0.5,
    ),
    PreprocessingConfig(
        name="wavelet_haar",
        course_topic="3.5 Wavelet denoising",
        description="Pure single-level Haar soft-threshold denoising without CLAHE or sharpening.",
        denoise="wavelet_haar",
    ),
    PreprocessingConfig(
        name="otsu_threshold",
        course_topic="1.2 Basic binary processing",
        description="CLAHE followed by global Otsu binarization.",
        clahe_clip_limit=2.0,
        threshold="otsu",
    ),
    PreprocessingConfig(
        name="adaptive_threshold",
        course_topic="1.2 Basic binary processing",
        description="CLAHE followed by local adaptive binarization.",
        clahe_clip_limit=2.0,
        threshold="adaptive",
    ),
    PreprocessingConfig(
        name="baseline_resize_bilinear",
        course_topic="7.1 Image sampling and interpolation",
        description="Training-time enhancement with bilinear model-input resizing.",
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=0.5,
        resize_interpolation="bilinear",
    ),
    PreprocessingConfig(
        name="baseline_resize_lanczos",
        course_topic="7.1 Image sampling and interpolation",
        description="Training-time enhancement with Lanczos model-input resizing.",
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=0.5,
        resize_interpolation="lanczos",
    ),
    # Extended classical image-processing sweep.
    PreprocessingConfig(
        name="percentile_stretch_1_99",
        course_topic="1.1 Gray-level transforms",
        description="Robust 1st-99th percentile contrast stretching.",
        percentile_low=1.0,
        percentile_high=99.0,
    ),
    PreprocessingConfig(
        name="percentile_stretch_2_98",
        course_topic="1.1 Gray-level transforms",
        description="Robust 2nd-98th percentile contrast stretching.",
        percentile_low=2.0,
        percentile_high=98.0,
    ),
    PreprocessingConfig(
        name="gamma_0_9",
        course_topic="1.1 Gray-level transforms",
        description="Mild gamma brightening without local enhancement.",
        gamma=0.9,
    ),
    PreprocessingConfig(
        name="gamma_1_1",
        course_topic="1.1 Gray-level transforms",
        description="Mild gamma darkening without local enhancement.",
        gamma=1.1,
    ),
    PreprocessingConfig(
        name="retinex_single",
        course_topic="3 Image restoration",
        description="Single-scale Retinex illumination normalization.",
        illumination="retinex",
        illumination_sigma=15.0,
    ),
    PreprocessingConfig(
        name="retinex_multiscale",
        course_topic="3 Image restoration",
        description="Multi-scale Retinex illumination normalization.",
        illumination="multiscale_retinex",
    ),
    PreprocessingConfig(
        name="homomorphic_filter",
        course_topic="2.5 Frequency-domain filtering",
        description="Homomorphic high-pass illumination correction in the log-frequency domain.",
        illumination="homomorphic",
    ),
    PreprocessingConfig(
        name="local_contrast_norm",
        course_topic="2.4 Spatial filtering",
        description="Local mean and variance normalization.",
        illumination="local_norm",
    ),
    PreprocessingConfig(
        name="nlm_denoise",
        course_topic="3.3 Noise reduction",
        description="Mild non-local means denoising.",
        denoise="nlm",
        nlm_h=3.0,
    ),
    PreprocessingConfig(
        name="wiener_3x3",
        course_topic="3.6 Minimum mean square error filtering",
        description="Adaptive local Wiener denoising with a 3x3 window.",
        denoise="wiener",
    ),
    PreprocessingConfig(
        name="unsharp_mild",
        course_topic="2.1 Spatial enhancement",
        description="Mild unsharp masking on grayscale.",
        sharpen_alpha=0.25,
    ),
    PreprocessingConfig(
        name="laplacian_mild",
        course_topic="2.4 Spatial filtering",
        description="Mild Laplacian high-frequency sharpening.",
        sharpen_alpha=0.20,
        sharpen_method="laplacian",
    ),
    PreprocessingConfig(
        name="dog_sharpen",
        course_topic="2.4 Spatial filtering",
        description="Difference-of-Gaussians band-pass sharpening.",
        sharpen_alpha=0.35,
        sharpen_method="dog",
        sharpen_sigma=0.6,
        sharpen_sigma_large=1.4,
    ),
    PreprocessingConfig(
        name="channel_red",
        course_topic="3.10 Multichannel image recovery",
        description="Use the red channel as a replicated grayscale input.",
        gray_channel="red",
    ),
    PreprocessingConfig(
        name="channel_green",
        course_topic="3.10 Multichannel image recovery",
        description="Use the green channel as a replicated grayscale input.",
        gray_channel="green",
    ),
    PreprocessingConfig(
        name="channel_blue",
        course_topic="3.10 Multichannel image recovery",
        description="Use the blue channel; potentially useful for yellow plates.",
        gray_channel="blue",
    ),
    PreprocessingConfig(
        name="channel_max_rgb",
        course_topic="3.10 Multichannel image recovery",
        description="Use the maximum RGB channel per pixel.",
        gray_channel="max",
    ),
    PreprocessingConfig(
        name="channel_hsv_value",
        course_topic="3.10 Multichannel image recovery",
        description="Use the HSV value channel.",
        gray_channel="hsv_v",
    ),
    PreprocessingConfig(
        name="channel_lab_l",
        course_topic="3.10 Multichannel image recovery",
        description="Use perceptual LAB luminance.",
        gray_channel="lab_l",
    ),
    PreprocessingConfig(
        name="channel_best_contrast",
        course_topic="3.10 Multichannel image recovery",
        description="Select the RGB/luma channel with the largest robust intensity range per image.",
        gray_channel="best_contrast",
    ),
    PreprocessingConfig(
        name="morph_close_horizontal",
        course_topic="2.3 Morphological filtering",
        description="Light horizontal 3x1 closing to reconnect broken horizontal strokes.",
        morphology="close",
        morphology_kernel_width=3,
        morphology_kernel_height=1,
    ),
    PreprocessingConfig(
        name="morph_close_vertical",
        course_topic="2.3 Morphological filtering",
        description="Light vertical 1x3 closing to reconnect broken vertical strokes.",
        morphology="close",
        morphology_kernel_width=1,
        morphology_kernel_height=3,
    ),
    PreprocessingConfig(
        name="morph_gradient_mild",
        course_topic="2.3 Morphological filtering",
        description="Blend a mild morphological edge gradient into grayscale.",
        morphology="gradient",
        morphology_strength=0.20,
    ),
    PreprocessingConfig(
        name="clahe_clip1_tile4",
        course_topic="2 Image enhancement",
        description="Gentler CLAHE with smaller 4x4 tiles.",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
    ),
    PreprocessingConfig(
        name="clahe_clip1_tile4_unsharp",
        course_topic="2 Image enhancement",
        description="Gentle 4x4 CLAHE plus mild unsharp masking.",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
        sharpen_alpha=0.25,
    ),
    # Multi-stage restoration and enhancement combinations. Deblurring is
    # deliberately followed by a mild low-pass/edge-preserving filter to
    # suppress the ringing and amplified noise produced by inverse filtering.
    PreprocessingConfig(
        name="wiener_deconv",
        course_topic="3 Image restoration",
        description="Pure mild Wiener deconvolution without a following denoiser.",
        deblur="wiener_deconv",
        deblur_sigma=0.7,
        deblur_balance=0.03,
    ),
    PreprocessingConfig(
        name="richardson_lucy",
        course_topic="3 Image restoration",
        description="Pure three-iteration Richardson-Lucy restoration without CLAHE or denoising.",
        deblur="richardson_lucy",
        deblur_sigma=0.7,
        deblur_iterations=3,
    ),
    PreprocessingConfig(
        name="wiener_deconv_gaussian_lowpass",
        course_topic="3 Image restoration",
        description="Mild Wiener deconvolution followed by Gaussian low-pass filtering.",
        deblur="wiener_deconv",
        deblur_sigma=0.7,
        deblur_balance=0.03,
        denoise="gaussian",
        gaussian_sigma=0.45,
    ),
    PreprocessingConfig(
        name="wiener_deconv_bilateral_lowpass",
        course_topic="3 Image restoration",
        description="Mild Wiener deconvolution followed by edge-preserving bilateral filtering.",
        deblur="wiener_deconv",
        deblur_sigma=0.7,
        deblur_balance=0.03,
        denoise="bilateral",
        bilateral_d=3,
        bilateral_sigma_color=25.0,
        bilateral_sigma_space=25.0,
    ),
    PreprocessingConfig(
        name="clahe_wiener_deconv_gaussian",
        course_topic="2-3 Enhancement and restoration",
        description="Gentle CLAHE, Wiener deconvolution, then Gaussian low-pass filtering.",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
        deblur="wiener_deconv",
        deblur_sigma=0.7,
        deblur_balance=0.03,
        denoise="gaussian",
        gaussian_sigma=0.45,
    ),
    PreprocessingConfig(
        name="clahe_wiener_deconv_bilateral",
        course_topic="2-3 Enhancement and restoration",
        description="Gentle CLAHE, Wiener deconvolution, then bilateral low-pass filtering.",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
        deblur="wiener_deconv",
        deblur_sigma=0.7,
        deblur_balance=0.03,
        denoise="bilateral",
        bilateral_d=3,
        bilateral_sigma_color=25.0,
        bilateral_sigma_space=25.0,
    ),
    PreprocessingConfig(
        name="homomorphic_wiener_bilateral",
        course_topic="3 Illumination and restoration",
        description="Homomorphic illumination correction, Wiener deconvolution, then bilateral filtering.",
        illumination="homomorphic",
        deblur="wiener_deconv",
        deblur_sigma=0.7,
        deblur_balance=0.03,
        denoise="bilateral",
        bilateral_d=3,
        bilateral_sigma_color=25.0,
        bilateral_sigma_space=25.0,
    ),
    PreprocessingConfig(
        name="rl_deblur_gaussian_lowpass",
        course_topic="3 Image restoration",
        description="Three Richardson-Lucy iterations followed by Gaussian low-pass filtering.",
        deblur="richardson_lucy",
        deblur_sigma=0.7,
        deblur_iterations=3,
        denoise="gaussian",
        gaussian_sigma=0.45,
    ),
    PreprocessingConfig(
        name="rl_deblur_bilateral_lowpass",
        course_topic="3 Image restoration",
        description="Three Richardson-Lucy iterations followed by bilateral filtering.",
        deblur="richardson_lucy",
        deblur_sigma=0.7,
        deblur_iterations=3,
        denoise="bilateral",
        bilateral_d=3,
        bilateral_sigma_color=25.0,
        bilateral_sigma_space=25.0,
    ),
    PreprocessingConfig(
        name="clahe_rl_deblur_bilateral",
        course_topic="2-3 Enhancement and restoration",
        description="Gentle CLAHE, Richardson-Lucy deblurring, then bilateral filtering.",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
        deblur="richardson_lucy",
        deblur_sigma=0.7,
        deblur_iterations=3,
        denoise="bilateral",
        bilateral_d=3,
        bilateral_sigma_color=25.0,
        bilateral_sigma_space=25.0,
    ),
    PreprocessingConfig(
        name="homomorphic_rl_deblur_bilateral",
        course_topic="3 Illumination and restoration",
        description="Homomorphic correction, Richardson-Lucy deblurring, then bilateral filtering.",
        illumination="homomorphic",
        deblur="richardson_lucy",
        deblur_sigma=0.7,
        deblur_iterations=3,
        denoise="bilateral",
        bilateral_d=3,
        bilateral_sigma_color=25.0,
        bilateral_sigma_space=25.0,
    ),
    PreprocessingConfig(
        name="homomorphic_unsharp_025",
        course_topic="2-3 Illumination and edge enhancement",
        description="Homomorphic illumination correction followed by mild unsharp masking.",
        illumination="homomorphic",
        sharpen_alpha=0.25,
    ),
    PreprocessingConfig(
        name="homomorphic_bilateral_mild",
        course_topic="2-3 Illumination and denoising",
        description="Homomorphic correction followed by a mild bilateral filter.",
        illumination="homomorphic",
        denoise="bilateral",
        bilateral_d=3,
        bilateral_sigma_color=25.0,
        bilateral_sigma_space=25.0,
    ),
    PreprocessingConfig(
        name="percentile_2_98_unsharp_025",
        course_topic="1-2 Contrast and edge enhancement",
        description="Robust 2-98 percentile stretch followed by mild unsharp masking.",
        percentile_low=2.0,
        percentile_high=98.0,
        sharpen_alpha=0.25,
    ),
    PreprocessingConfig(
        name="percentile_2_98_clahe1",
        course_topic="1-2 Global and local contrast enhancement",
        description="Robust percentile stretch followed by gentle local CLAHE.",
        percentile_low=2.0,
        percentile_high=98.0,
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
    ),
    PreprocessingConfig(
        name="gamma11_clahe1",
        course_topic="1-2 Intensity and local contrast enhancement",
        description="Mild gamma darkening followed by gentle local CLAHE.",
        gamma=1.1,
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
    ),
    PreprocessingConfig(
        name="clahe_sobel_fusion_010",
        course_topic="4 Edge processing",
        description="Gentle CLAHE with a 10% Sobel edge-magnitude fusion.",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
        edge_enhancement="sobel",
        edge_strength=0.10,
    ),
    PreprocessingConfig(
        name="homomorphic_sobel_fusion_010",
        course_topic="3-4 Illumination and edge processing",
        description="Homomorphic correction with a 10% Sobel edge-magnitude fusion.",
        illumination="homomorphic",
        edge_enhancement="sobel",
        edge_strength=0.10,
    ),
    PreprocessingConfig(
        name="dual_stroke_mild",
        course_topic="2.3 Morphological stroke enhancement",
        description="Enhance both dark and bright character strokes with top-hat/black-hat responses.",
        edge_enhancement="dual_stroke",
        edge_strength=0.15,
    ),
    PreprocessingConfig(
        name="clahe_dual_stroke_mild",
        course_topic="2-4 Contrast and stroke enhancement",
        description="Gentle CLAHE followed by dual-polarity character-stroke enhancement.",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
        edge_enhancement="dual_stroke",
        edge_strength=0.15,
    ),
    PreprocessingConfig(
        name="homomorphic_dual_stroke_mild",
        course_topic="3-4 Illumination and stroke enhancement",
        description="Homomorphic correction followed by dual-polarity stroke enhancement.",
        illumination="homomorphic",
        edge_enhancement="dual_stroke",
        edge_strength=0.15,
    ),
    PreprocessingConfig(
        name="content_crop_gray",
        course_topic="6 Character-region isolation",
        description="Detect the union of plausible character components, crop it with a margin, then OCR.",
        character_isolation="content_crop",
    ),
    PreprocessingConfig(
        name="content_crop_clahe",
        course_topic="2-6 Contrast and character-region isolation",
        description="Character-region crop followed by gentle CLAHE.",
        character_isolation="content_crop",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
    ),
    PreprocessingConfig(
        name="content_crop_homomorphic",
        course_topic="3-6 Illumination and character-region isolation",
        description="Character-region crop followed by homomorphic correction.",
        character_isolation="content_crop",
        illumination="homomorphic",
    ),
    PreprocessingConfig(
        name="component_mask_gray",
        course_topic="6 Character segmentation",
        description="Keep detected character components and replace plate background by its median intensity.",
        character_isolation="component_mask",
    ),
    PreprocessingConfig(
        name="component_fusion_gray",
        course_topic="6 Character segmentation",
        description="Keep character components sharp while softly suppressing the plate background.",
        character_isolation="component_fusion",
    ),
    PreprocessingConfig(
        name="component_fusion_clahe",
        course_topic="2-6 Contrast and character segmentation",
        description="Gentle CLAHE plus component-guided background suppression.",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
        character_isolation="component_fusion",
    ),
    PreprocessingConfig(
        name="component_fusion_homomorphic",
        course_topic="3-6 Illumination and character segmentation",
        description="Homomorphic correction plus component-guided background suppression.",
        illumination="homomorphic",
        character_isolation="component_fusion",
    ),
    # Dataset-adaptive policies. They route each image to one of three already
    # benchmarked pipelines using only label-free image statistics.
    PreprocessingConfig(
        name="adaptive_quality_cv",
        course_topic="Adaptive preprocessing",
        description="Cross-validated quality router: homomorphic for weak images, RL for high detail, CLAHE otherwise.",
        adaptive_policy="quality_cv",
    ),
    PreprocessingConfig(
        name="adaptive_quality_conservative",
        course_topic="Adaptive preprocessing",
        description="Conservative quality router with narrower homomorphic and RL activation regions.",
        adaptive_policy="quality_conservative",
    ),
    PreprocessingConfig(
        name="adaptive_brightness_3way",
        course_topic="Adaptive preprocessing",
        description="Route dark, medium, and bright plates to homomorphic, CLAHE, and RL pipelines.",
        adaptive_policy="brightness_3way",
    ),
    PreprocessingConfig(
        name="adaptive_aspect_3way",
        course_topic="Adaptive preprocessing",
        description="Route tall/two-line, square, and wide/one-line crops by aspect ratio.",
        adaptive_policy="aspect_3way",
    ),
    PreprocessingConfig(
        name="adaptive_size_3way",
        course_topic="Adaptive preprocessing",
        description="Route very small, medium, and large-width crops to size-specific pipelines.",
        adaptive_policy="size_3way",
    ),
    PreprocessingConfig(
        name="adaptive_noise_3way",
        course_topic="Adaptive preprocessing",
        description="Route crops using a robust high-frequency residual noise estimate.",
        adaptive_policy="noise_3way",
    ),
    PreprocessingConfig(
        name="adaptive_dark_fraction_3way",
        course_topic="Adaptive preprocessing",
        description="Route crops according to the fraction of very dark pixels.",
        adaptive_policy="dark_fraction_3way",
    ),
    PreprocessingConfig(
        name="adaptive_small_then_quality",
        course_topic="Adaptive preprocessing",
        description="Use RL for tiny crops, otherwise apply the cross-validated quality router.",
        adaptive_policy="small_then_quality",
    ),
    # Small crops are common in this dataset. Upscaling is activated only below
    # the validation-derived size limits, before enhancement and final resize.
    PreprocessingConfig(
        name="small_lanczos_clahe",
        course_topic="7 Sampling and local enhancement",
        description="Lanczos 2x pre-upscale for tiny crops followed by gentle CLAHE.",
        pre_upscale_factor=2.0,
        pre_upscale_min_height=32,
        pre_upscale_min_width=65,
        pre_upscale_interpolation="lanczos",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
    ),
    PreprocessingConfig(
        name="small_cubic_clahe",
        course_topic="7 Sampling and local enhancement",
        description="Bicubic 2x pre-upscale for tiny crops followed by gentle CLAHE.",
        pre_upscale_factor=2.0,
        pre_upscale_min_height=32,
        pre_upscale_min_width=65,
        pre_upscale_interpolation="cubic",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
    ),
    PreprocessingConfig(
        name="small_lanczos_homomorphic",
        course_topic="3-7 Sampling and illumination",
        description="Lanczos pre-upscale for tiny crops followed by homomorphic correction.",
        pre_upscale_factor=2.0,
        pre_upscale_min_height=32,
        pre_upscale_min_width=65,
        illumination="homomorphic",
    ),
    PreprocessingConfig(
        name="small_lanczos_rl_bilateral",
        course_topic="3-7 Sampling and restoration",
        description="Lanczos pre-upscale for tiny crops followed by RL deblur and bilateral filtering.",
        pre_upscale_factor=2.0,
        pre_upscale_min_height=32,
        pre_upscale_min_width=65,
        deblur="richardson_lucy",
        deblur_sigma=0.7,
        deblur_iterations=3,
        denoise="bilateral",
        bilateral_d=3,
        bilateral_sigma_color=25.0,
        bilateral_sigma_space=25.0,
    ),
    PreprocessingConfig(
        name="small_lanczos_clahe_rl",
        course_topic="2-3-7 Sampling and restoration",
        description="Lanczos pre-upscale for tiny crops, CLAHE, RL deblur, then bilateral filtering.",
        pre_upscale_factor=2.0,
        pre_upscale_min_height=32,
        pre_upscale_min_width=65,
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
        deblur="richardson_lucy",
        deblur_sigma=0.7,
        deblur_iterations=3,
        denoise="bilateral",
        bilateral_d=3,
        bilateral_sigma_color=25.0,
        bilateral_sigma_space=25.0,
    ),
    # Finer CLAHE settings for the small native crop sizes in this dataset.
    PreprocessingConfig(
        name="clahe_clip05_tile2",
        course_topic="2 Local contrast enhancement",
        description="Very gentle CLAHE using a 2x2 tile grid.",
        clahe_clip_limit=0.5,
        clahe_tile_size=2,
    ),
    PreprocessingConfig(
        name="clahe_clip075_tile2",
        course_topic="2 Local contrast enhancement",
        description="Gentle CLAHE with clip 0.75 and a 2x2 tile grid.",
        clahe_clip_limit=0.75,
        clahe_tile_size=2,
    ),
    PreprocessingConfig(
        name="clahe_clip1_tile2",
        course_topic="2 Local contrast enhancement",
        description="CLAHE clip 1.0 with a coarse 2x2 tile grid.",
        clahe_clip_limit=1.0,
        clahe_tile_size=2,
    ),
    PreprocessingConfig(
        name="clahe_clip05_tile4",
        course_topic="2 Local contrast enhancement",
        description="Very gentle CLAHE using the established 4x4 tile grid.",
        clahe_clip_limit=0.5,
        clahe_tile_size=4,
    ),
    PreprocessingConfig(
        name="clahe_clip075_tile4",
        course_topic="2 Local contrast enhancement",
        description="CLAHE clip 0.75 using the established 4x4 tile grid.",
        clahe_clip_limit=0.75,
        clahe_tile_size=4,
    ),
    PreprocessingConfig(
        name="clahe_clip125_tile4",
        course_topic="2 Local contrast enhancement",
        description="Slightly stronger CLAHE clip 1.25 with a 4x4 tile grid.",
        clahe_clip_limit=1.25,
        clahe_tile_size=4,
    ),
    # Additional dataset-derived chains for uneven lighting, saturated plates,
    # and cases where deblurring benefits from a small final sharpening step.
    PreprocessingConfig(
        name="homomorphic_clahe05",
        course_topic="2-3 Illumination and local contrast",
        description="Homomorphic correction followed by very gentle CLAHE.",
        illumination="homomorphic",
        clahe_clip_limit=0.5,
        clahe_tile_size=4,
    ),
    PreprocessingConfig(
        name="homomorphic_unsharp_010",
        course_topic="3-4 Illumination and edge enhancement",
        description="Homomorphic correction with a conservative 10% unsharp mask.",
        illumination="homomorphic",
        sharpen_alpha=0.10,
    ),
    PreprocessingConfig(
        name="rl_bilateral_unsharp_010",
        course_topic="3-4 Restoration and edge enhancement",
        description="RL deblur, bilateral low-pass, then a conservative unsharp mask.",
        deblur="richardson_lucy",
        deblur_sigma=0.7,
        deblur_iterations=3,
        denoise="bilateral",
        bilateral_d=3,
        bilateral_sigma_color=25.0,
        bilateral_sigma_space=25.0,
        sharpen_alpha=0.10,
    ),
    PreprocessingConfig(
        name="clahe_rl_bilateral_unsharp_010",
        course_topic="2-4 Enhancement and restoration",
        description="CLAHE, RL deblur, bilateral low-pass, then a conservative unsharp mask.",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
        deblur="richardson_lucy",
        deblur_sigma=0.7,
        deblur_iterations=3,
        denoise="bilateral",
        bilateral_d=3,
        bilateral_sigma_color=25.0,
        bilateral_sigma_space=25.0,
        sharpen_alpha=0.10,
    ),
    PreprocessingConfig(
        name="percentile_rl_bilateral",
        course_topic="1-3 Contrast and restoration",
        description="Robust percentile stretch followed by RL deblur and bilateral filtering.",
        percentile_low=2.0,
        percentile_high=98.0,
        deblur="richardson_lucy",
        deblur_sigma=0.7,
        deblur_iterations=3,
        denoise="bilateral",
        bilateral_d=3,
        bilateral_sigma_color=25.0,
        bilateral_sigma_space=25.0,
    ),
    PreprocessingConfig(
        name="green_clahe1",
        course_topic="2 Multichannel and local contrast",
        description="Green channel extraction followed by gentle CLAHE.",
        gray_channel="green",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
    ),
    PreprocessingConfig(
        name="green_percentile_2_98",
        course_topic="1 Multichannel contrast enhancement",
        description="Green channel extraction followed by robust percentile stretching.",
        gray_channel="green",
        percentile_low=2.0,
        percentile_high=98.0,
    ),
    PreprocessingConfig(
        name="grayscale_letterbox",
        course_topic="7.1 Image sampling and interpolation",
        description="Preserve crop aspect ratio and pad before PARSeq normalization.",
        resize_mode="letterbox",
    ),
]

# Backwards-compatible aliases used by older commands in this repository.
_ALIASES = {
    "raw": "raw_rgb",
    "clahe_sharpen": "train_baseline",
    "clahe_1_5": "clahe_gray",
    "clahe_2_sharp_1_2": "clahe_unsharp",
    "clahe_3_sharp_1_5": "baseline_strong_unsharp",
    "adaptive_thresh": "adaptive_threshold",
}


def get_preprocessing_config(name: str) -> PreprocessingConfig:
    name = _ALIASES.get(name, name)
    for cfg in SWEEP_CONFIGS:
        if cfg.name == name:
            return cfg
    raise KeyError(f"Unknown preprocessing config: {name}")


def list_preprocessing_configs() -> list[str]:
    return [cfg.name for cfg in SWEEP_CONFIGS]


def _odd_at_least(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 else value + 1


def _haar_soft_threshold(channel: np.ndarray) -> np.ndarray:
    """Dependency-free, one-level orthonormal Haar wavelet denoising."""
    src = channel.astype(np.float32)
    orig_h, orig_w = src.shape
    if orig_h % 2 or orig_w % 2:
        src = np.pad(src, ((0, orig_h % 2), (0, orig_w % 2)), mode="reflect")
    a, b = src[0::2, 0::2], src[0::2, 1::2]
    c, d = src[1::2, 0::2], src[1::2, 1::2]
    ll = (a + b + c + d) / 2.0
    lh = (a - b + c - d) / 2.0
    hl = (a + b - c - d) / 2.0
    hh = (a - b - c + d) / 2.0
    sigma = float(np.median(np.abs(hh)) / 0.6745) if hh.size else 0.0
    threshold = sigma * np.sqrt(2.0 * np.log(max(src.size, 2)))

    def shrink(detail: np.ndarray) -> np.ndarray:
        return np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)

    lh, hl, hh = shrink(lh), shrink(hl), shrink(hh)
    out = np.empty_like(src)
    out[0::2, 0::2] = (ll + lh + hl + hh) / 2.0
    out[0::2, 1::2] = (ll - lh + hl - hh) / 2.0
    out[1::2, 0::2] = (ll + lh - hl - hh) / 2.0
    out[1::2, 1::2] = (ll - lh - hl + hh) / 2.0
    return np.clip(out[:orig_h, :orig_w], 0, 255).astype(np.uint8)


def _robust_rescale(channel: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    src = channel.astype(np.float32)
    low, high = np.percentile(src, [float(low_percentile), float(high_percentile)])
    if high <= low + 1e-6:
        return np.clip(src, 0, 255).astype(np.uint8)
    return np.clip((src - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def _select_gray_channel(arr: np.ndarray, mode: str) -> np.ndarray:
    import cv2

    channels = {
        "red": arr[:, :, 0],
        "green": arr[:, :, 1],
        "blue": arr[:, :, 2],
        "max": arr.max(axis=2),
        "luma": cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY),
        "hsv_v": cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[:, :, 2],
        "lab_l": cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[:, :, 0],
    }
    if mode == "best_contrast":
        candidates = [channels[key] for key in ("luma", "red", "green", "blue")]
        scores = [float(np.percentile(channel, 95) - np.percentile(channel, 5)) for channel in candidates]
        return candidates[int(np.argmax(scores))]
    if mode not in channels:
        raise ValueError(f"Unsupported grayscale channel: {mode}")
    return channels[mode]


def _retinex(channel: np.ndarray, sigmas: tuple[float, ...]) -> np.ndarray:
    import cv2

    src = channel.astype(np.float32) + 1.0
    response = np.zeros_like(src)
    for sigma in sigmas:
        illumination = cv2.GaussianBlur(src, (0, 0), float(sigma))
        response += np.log(src) - np.log(np.maximum(illumination, 1e-6))
    return _robust_rescale(response / len(sigmas), 1.0, 99.0)


def _homomorphic(channel: np.ndarray) -> np.ndarray:
    import cv2

    src = np.log1p(channel.astype(np.float32))
    rows, cols = src.shape
    yy, xx = np.ogrid[:rows, :cols]
    distance2 = (yy - rows / 2.0) ** 2 + (xx - cols / 2.0) ** 2
    cutoff = max(min(rows, cols) / 4.0, 2.0)
    low_gain, high_gain = 0.7, 1.4
    transfer = (high_gain - low_gain) * (1.0 - np.exp(-distance2 / (cutoff * cutoff))) + low_gain
    spectrum = np.fft.fftshift(np.fft.fft2(src))
    restored = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * transfer)))
    return _robust_rescale(np.expm1(restored), 1.0, 99.0)


def _local_contrast_normalize(channel: np.ndarray, sigma: float = 7.0) -> np.ndarray:
    import cv2

    src = channel.astype(np.float32)
    mean = cv2.GaussianBlur(src, (0, 0), sigma)
    mean_square = cv2.GaussianBlur(src * src, (0, 0), sigma)
    std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    normalized = 127.5 + 40.0 * (src - mean) / np.maximum(std, 10.0)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _adaptive_wiener(channel: np.ndarray, window: int) -> np.ndarray:
    import cv2

    src = channel.astype(np.float32)
    size = (int(window), int(window))
    mean = cv2.boxFilter(src, cv2.CV_32F, size, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_square = cv2.boxFilter(src * src, cv2.CV_32F, size, normalize=True, borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(mean_square - mean * mean, 0.0)
    noise_variance = float(np.mean(variance))
    gain = np.maximum(variance - noise_variance, 0.0) / np.maximum(variance, 1e-6)
    return np.clip(mean + gain * (src - mean), 0, 255).astype(np.uint8)


def _gaussian_psf(kernel_size: int, sigma: float) -> np.ndarray:
    import cv2

    size = _odd_at_least(kernel_size)
    vector = cv2.getGaussianKernel(size, max(float(sigma), 0.1))
    kernel = vector @ vector.T
    return (kernel / max(float(kernel.sum()), 1e-12)).astype(np.float32)


def _wiener_deconvolution(
    channel: np.ndarray, kernel_size: int, sigma: float, balance: float
) -> np.ndarray:
    """Small-PSF Wiener inverse filter with reflected padding to limit ringing."""

    src = channel.astype(np.float32) / 255.0
    pad = max(_odd_at_least(kernel_size) * 2, 4)
    padded = np.pad(src, ((pad, pad), (pad, pad)), mode="reflect")
    psf = _gaussian_psf(kernel_size, sigma)
    transfer = np.zeros_like(padded, dtype=np.float32)
    kh, kw = psf.shape
    transfer[:kh, :kw] = psf
    transfer = np.roll(transfer, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    transfer_fft = np.fft.fft2(transfer)
    restored_fft = (
        np.conj(transfer_fft)
        * np.fft.fft2(padded)
        / (np.abs(transfer_fft) ** 2 + max(float(balance), 1e-6))
    )
    restored = np.real(np.fft.ifft2(restored_fft))[pad:-pad, pad:-pad]
    return np.clip(restored * 255.0, 0, 255).astype(np.uint8)


def _richardson_lucy(
    channel: np.ndarray, kernel_size: int, sigma: float, iterations: int
) -> np.ndarray:
    """Conservative Richardson-Lucy deconvolution for an assumed Gaussian PSF."""

    import cv2

    observed = np.clip(channel.astype(np.float32) / 255.0, 1e-4, 1.0)
    estimate = observed.copy()
    psf = _gaussian_psf(kernel_size, sigma)
    mirrored = psf[::-1, ::-1]
    for _ in range(max(int(iterations), 1)):
        blurred = cv2.filter2D(estimate, cv2.CV_32F, psf, borderType=cv2.BORDER_REFLECT)
        relative = observed / np.maximum(blurred, 1e-4)
        correction = cv2.filter2D(relative, cv2.CV_32F, mirrored, borderType=cv2.BORDER_REFLECT)
        estimate = np.clip(estimate * correction, 0.0, 1.0)
    return np.clip(estimate * 255.0, 0, 255).astype(np.uint8)


def _apply_per_channel(work: np.ndarray, operation) -> np.ndarray:
    if work.ndim == 2:
        return operation(work)
    return np.stack([operation(work[:, :, index]) for index in range(work.shape[2])], axis=-1)


def _character_component_mask(gray: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Find plausible character components under both dark/light text polarities."""

    import cv2

    src = np.asarray(gray, dtype=np.uint8)
    height, width = src.shape
    if height < 8 or width < 16:
        return None, None
    smoothed = cv2.GaussianBlur(src, (3, 3), 0.5)
    candidates = []
    for threshold_type in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _level, binary = cv2.threshold(smoothed, 0, 255, threshold_type + cv2.THRESH_OTSU)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        selected = []
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            fill = area / max(component_width * component_height, 1)
            if not (max(3, int(0.18 * height)) <= component_height <= int(0.96 * height)):
                continue
            if not (max(1, int(0.008 * width)) <= component_width <= int(0.38 * width)):
                continue
            if area < max(4, int(0.0015 * height * width)) or not (0.08 <= fill <= 0.95):
                continue
            selected.append(label)
        if selected:
            mask = np.isin(labels, selected).astype(np.uint8) * 255
            chosen_stats = stats[selected]
            x0 = int(chosen_stats[:, 0].min())
            y0 = int(chosen_stats[:, 1].min())
            x1 = int((chosen_stats[:, 0] + chosen_stats[:, 2]).max())
            y1 = int((chosen_stats[:, 1] + chosen_stats[:, 3]).max())
            coverage = (x1 - x0) * (y1 - y0) / max(width * height, 1)
            score = len(selected) + min(coverage, 1.0)
            candidates.append((score, len(selected), mask, (x0, y0, x1, y1)))
    if not candidates:
        return None, None
    _score, component_count, mask, box = max(candidates, key=lambda item: item[0])
    if component_count < 3 or (box[2] - box[0]) < 0.30 * width:
        return None, None
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return mask, box


def _crop_character_region(arr: np.ndarray, margin: float) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _mask, box = _character_component_mask(gray)
    if box is None:
        return arr
    x0, y0, x1, y1 = box
    height, width = gray.shape
    margin_x = max(2, int((x1 - x0) * float(margin)))
    margin_y = max(1, int((y1 - y0) * float(margin)))
    x0, x1 = max(0, x0 - margin_x), min(width, x1 + margin_x)
    y0, y1 = max(0, y0 - margin_y), min(height, y1 + margin_y)
    if (x1 - x0) < 0.40 * width or (y1 - y0) < 0.30 * height:
        return arr
    return arr[y0:y1, x0:x1]


def _image_quality_features(image: Image.Image) -> dict[str, float]:
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    residual = np.abs(
        gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    )
    return {
        "width": float(gray.shape[1]),
        "height": float(gray.shape[0]),
        "aspect": float(gray.shape[1] / max(gray.shape[0], 1)),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "saturation": float(hsv[:, :, 1].mean()),
        "noise": float(np.median(residual)),
        "dark_fraction": float((gray < 50).mean()),
    }


def _adaptive_config_name(image: Image.Image, policy: str) -> str:
    """Route an image to a fixed pipeline using validation-locked thresholds."""

    feature = _image_quality_features(image)
    if policy == "quality_cv":
        low_quality = (
            feature["contrast"] < 60
            or feature["brightness"] < 90
            or feature["saturation"] > 55
        )
        high_detail = feature["brightness"] > 145 or feature["sharpness"] > 5000
        return "homomorphic_filter" if low_quality else "rl_deblur_bilateral_lowpass" if high_detail else "clahe_clip1_tile4"
    if policy == "quality_conservative":
        low_quality = (
            feature["contrast"] < 50
            or feature["brightness"] < 100
            or feature["saturation"] > 75
        )
        high_detail = feature["brightness"] > 165 or feature["sharpness"] > 12000
        return "homomorphic_filter" if low_quality else "rl_deblur_bilateral_lowpass" if high_detail else "clahe_clip1_tile4"
    if policy == "brightness_3way":
        if feature["brightness"] < 110:
            return "homomorphic_filter"
        return "rl_deblur_bilateral_lowpass" if feature["brightness"] > 155 else "clahe_clip1_tile4"
    if policy == "aspect_3way":
        if feature["aspect"] < 1.25:
            return "clahe_clip1_tile4"
        return "homomorphic_filter" if feature["aspect"] < 2.2 else "rl_deblur_bilateral_lowpass"
    if policy == "size_3way":
        if feature["width"] <= 65:
            return "rl_deblur_bilateral_lowpass"
        return "clahe_rl_deblur_bilateral" if feature["width"] <= 85 else "homomorphic_filter"
    if policy == "noise_3way":
        if feature["noise"] <= 5:
            return "homomorphic_filter"
        return "clahe_rl_deblur_bilateral" if feature["noise"] <= 10 else "rl_deblur_bilateral_lowpass"
    if policy == "dark_fraction_3way":
        if feature["dark_fraction"] <= 0.08:
            return "homomorphic_filter"
        return "clahe_rl_deblur_bilateral" if feature["dark_fraction"] <= 0.20 else "rl_deblur_bilateral_lowpass"
    if policy == "small_then_quality":
        if feature["width"] <= 65 or feature["height"] <= 28:
            return "rl_deblur_bilateral_lowpass"
        return _adaptive_config_name(image, "quality_cv")
    raise ValueError(f"Unsupported adaptive policy: {policy}")


def _opencv_preprocess(image: Image.Image, cfg: PreprocessingConfig) -> Image.Image:
    import cv2

    arr = np.asarray(image.convert("RGB"))
    should_upscale = cfg.pre_upscale_factor > 1.0 and (
        (cfg.pre_upscale_min_height > 0 and arr.shape[0] < cfg.pre_upscale_min_height)
        or (cfg.pre_upscale_min_width > 0 and arr.shape[1] < cfg.pre_upscale_min_width)
    )
    if should_upscale:
        interpolation = {
            "nearest": cv2.INTER_NEAREST,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }.get(cfg.pre_upscale_interpolation)
        if interpolation is None:
            raise ValueError(f"Unsupported pre-upscale interpolation: {cfg.pre_upscale_interpolation}")
        arr = cv2.resize(
            arr,
            None,
            fx=float(cfg.pre_upscale_factor),
            fy=float(cfg.pre_upscale_factor),
            interpolation=interpolation,
        )
    if cfg.character_isolation == "content_crop":
        arr = _crop_character_region(arr, cfg.character_margin)
    elif cfg.character_isolation not in {"none", "component_mask", "component_fusion"}:
        raise ValueError(f"Unsupported character isolation method: {cfg.character_isolation}")
    work = _select_gray_channel(arr, cfg.gray_channel) if cfg.grayscale else arr.copy()

    if cfg.gamma != 1.0:
        lut = np.clip((np.arange(256, dtype=np.float32) / 255.0) ** float(cfg.gamma) * 255.0, 0, 255)
        work = cv2.LUT(work, lut.astype(np.uint8))

    if cfg.percentile_low is not None and cfg.percentile_high is not None:
        if work.ndim == 2:
            work = _robust_rescale(work, cfg.percentile_low, cfg.percentile_high)
        else:
            work = np.stack(
                [_robust_rescale(work[:, :, idx], cfg.percentile_low, cfg.percentile_high) for idx in range(3)],
                axis=-1,
            )

    if cfg.illumination != "none":
        if work.ndim != 2:
            raise ValueError("Illumination normalization currently requires grayscale input")
        if cfg.illumination == "retinex":
            work = _retinex(work, (float(cfg.illumination_sigma),))
        elif cfg.illumination == "multiscale_retinex":
            work = _retinex(work, (5.0, 15.0, 40.0))
        elif cfg.illumination == "homomorphic":
            work = _homomorphic(work)
        elif cfg.illumination == "local_norm":
            work = _local_contrast_normalize(work)
        else:
            raise ValueError(f"Unsupported illumination method: {cfg.illumination}")

    if cfg.autocontrast:
        work = np.asarray(ImageOps.autocontrast(Image.fromarray(work)))
    if cfg.histogram_equalization:
        if work.ndim == 2:
            work = cv2.equalizeHist(work)
        else:
            lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            work = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if cfg.clahe_clip_limit is not None:
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg.clahe_clip_limit),
            tileGridSize=(int(cfg.clahe_tile_size), int(cfg.clahe_tile_size)),
        )
        if work.ndim == 2:
            work = clahe.apply(work)
        else:
            lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            work = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if cfg.deblur == "wiener_deconv":
        work = _apply_per_channel(
            work,
            lambda channel: _wiener_deconvolution(
                channel,
                cfg.deblur_kernel_size,
                cfg.deblur_sigma,
                cfg.deblur_balance,
            ),
        )
    elif cfg.deblur == "richardson_lucy":
        work = _apply_per_channel(
            work,
            lambda channel: _richardson_lucy(
                channel,
                cfg.deblur_kernel_size,
                cfg.deblur_sigma,
                cfg.deblur_iterations,
            ),
        )
    elif cfg.deblur != "none":
        raise ValueError(f"Unsupported deblur method: {cfg.deblur}")

    if cfg.denoise == "gaussian":
        work = cv2.GaussianBlur(work, (0, 0), float(cfg.gaussian_sigma))
    elif cfg.denoise == "median":
        work = cv2.medianBlur(work, _odd_at_least(cfg.median_ksize))
    elif cfg.denoise == "bilateral":
        work = cv2.bilateralFilter(
            work,
            int(cfg.bilateral_d),
            float(cfg.bilateral_sigma_color),
            float(cfg.bilateral_sigma_space),
        )
    elif cfg.denoise == "wavelet_haar":
        if work.ndim == 2:
            work = _haar_soft_threshold(work)
        else:
            work = np.stack([_haar_soft_threshold(work[:, :, idx]) for idx in range(3)], axis=-1)
    elif cfg.denoise == "nlm":
        if work.ndim == 2:
            work = cv2.fastNlMeansDenoising(work, None, float(cfg.nlm_h), 7, 21)
        else:
            work = cv2.fastNlMeansDenoisingColored(work, None, float(cfg.nlm_h), float(cfg.nlm_h), 7, 21)
    elif cfg.denoise == "wiener":
        window = _odd_at_least(cfg.wiener_ksize)
        if work.ndim == 2:
            work = _adaptive_wiener(work, window)
        else:
            work = np.stack(
                [_adaptive_wiener(work[:, :, idx], window) for idx in range(3)], axis=-1
            )
    elif cfg.denoise != "none":
        raise ValueError(f"Unsupported denoise method: {cfg.denoise}")

    if cfg.sharpen_alpha > 0:
        alpha = float(cfg.sharpen_alpha)
        if cfg.sharpen_method == "unsharp":
            blur = cv2.GaussianBlur(work, (0, 0), float(cfg.sharpen_sigma))
            work = cv2.addWeighted(work, 1.0 + alpha, blur, -alpha, 0)
        elif cfg.sharpen_method == "laplacian":
            laplacian = cv2.Laplacian(work, cv2.CV_32F, ksize=3)
            work = np.clip(work.astype(np.float32) - alpha * laplacian, 0, 255).astype(np.uint8)
        elif cfg.sharpen_method == "dog":
            small = cv2.GaussianBlur(work, (0, 0), float(cfg.sharpen_sigma))
            large = cv2.GaussianBlur(work, (0, 0), float(cfg.sharpen_sigma_large))
            detail = small.astype(np.float32) - large.astype(np.float32)
            work = np.clip(work.astype(np.float32) + alpha * detail, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported sharpen method: {cfg.sharpen_method}")

    if cfg.edge_enhancement == "sobel":
        gray_for_edges = work if work.ndim == 2 else cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray_for_edges, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_for_edges, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        scale = float(np.percentile(magnitude, 99.0))
        edge = np.clip(magnitude * (255.0 / max(scale, 1e-6)), 0, 255).astype(np.uint8)
        if work.ndim == 3:
            edge = np.repeat(edge[:, :, None], 3, axis=2)
        work = cv2.addWeighted(work, 1.0, edge, float(cfg.edge_strength), 0)
    elif cfg.edge_enhancement == "dual_stroke":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blackhat = cv2.morphologyEx(work, cv2.MORPH_BLACKHAT, kernel)
        tophat = cv2.morphologyEx(work, cv2.MORPH_TOPHAT, kernel)
        enhanced = (
            work.astype(np.float32)
            - float(cfg.edge_strength) * blackhat.astype(np.float32)
            + float(cfg.edge_strength) * tophat.astype(np.float32)
        )
        work = np.clip(enhanced, 0, 255).astype(np.uint8)
    elif cfg.edge_enhancement != "none":
        raise ValueError(f"Unsupported edge enhancement method: {cfg.edge_enhancement}")

    if cfg.morphology != "none":
        kernel_width = cfg.morphology_kernel_width or _odd_at_least(cfg.morphology_ksize)
        kernel_height = cfg.morphology_kernel_height or _odd_at_least(cfg.morphology_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kernel_width), int(kernel_height)))
        if cfg.morphology == "close":
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
        elif cfg.morphology == "blackhat":
            blackhat = cv2.morphologyEx(work, cv2.MORPH_BLACKHAT, kernel)
            work = cv2.addWeighted(work, 1.0, blackhat, -float(cfg.morphology_strength), 0)
        elif cfg.morphology == "gradient":
            gradient = cv2.morphologyEx(work, cv2.MORPH_GRADIENT, kernel)
            work = cv2.addWeighted(work, 1.0, gradient, float(cfg.morphology_strength), 0)
        else:
            raise ValueError(f"Unsupported morphology method: {cfg.morphology}")

    if cfg.character_isolation in {"component_mask", "component_fusion"}:
        gray_for_components = work if work.ndim == 2 else cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        component_mask, _box = _character_component_mask(gray_for_components)
        if component_mask is not None:
            if work.ndim == 3:
                component_mask = np.repeat(component_mask[:, :, None], 3, axis=2)
                median = np.median(work.reshape(-1, 3), axis=0).reshape(1, 1, 3)
            else:
                median = float(np.median(work))
            if cfg.character_isolation == "component_mask":
                work = np.where(component_mask > 0, work, median).astype(np.uint8)
            else:
                blurred = cv2.GaussianBlur(work, (0, 0), float(cfg.background_blur_sigma))
                background = 0.65 * blurred.astype(np.float32) + 0.35 * np.asarray(median, dtype=np.float32)
                work = np.where(component_mask > 0, work, background).astype(np.uint8)

    if cfg.threshold != "none":
        gray = work if work.ndim == 2 else cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        if cfg.threshold == "otsu":
            _level, work = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif cfg.threshold == "adaptive":
            work = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                _odd_at_least(cfg.adaptive_block_size),
                int(cfg.adaptive_c),
            )
        else:
            raise ValueError(f"Unsupported threshold method: {cfg.threshold}")

    if work.ndim == 2:
        work = np.repeat(work[:, :, None], 3, axis=2)
    return Image.fromarray(work.astype(np.uint8), mode="RGB")


def preprocess_plate_image(image: Image.Image, cfg: PreprocessingConfig | str | None = None) -> Image.Image:
    cfg = DEFAULT_CONFIG if cfg is None else get_preprocessing_config(cfg) if isinstance(cfg, str) else cfg
    if cfg.adaptive_policy != "none":
        selected = get_preprocessing_config(_adaptive_config_name(image, cfg.adaptive_policy))
        return _opencv_preprocess(image, selected)
    if cfg.name == RAW_CONFIG.name:
        return image.convert("RGB")
    try:
        return _opencv_preprocess(image, cfg)
    except ImportError:
        gray = ImageOps.grayscale(image)
        if cfg.autocontrast or cfg.histogram_equalization or cfg.clahe_clip_limit is not None:
            gray = ImageOps.autocontrast(gray)
        if cfg.sharpen_alpha > 0:
            gray = ImageEnhance.Sharpness(gray).enhance(1.0 + cfg.sharpen_alpha)
        return Image.merge("RGB", (gray, gray, gray))


def iter_named_configs(names: Iterable[str] | None = None) -> list[PreprocessingConfig]:
    if names is None:
        return list(SWEEP_CONFIGS)
    return [get_preprocessing_config(name) for name in names]
