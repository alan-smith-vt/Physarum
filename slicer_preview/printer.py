"""Printer specs and preview scale constants."""

# ── Hardware ──
REAL_PIXEL_X_UM = 14.0       # µm per pixel, horizontal
REAL_PIXEL_Y_UM = 19.0       # µm per pixel, vertical (non-square)
REAL_LAYER_UM   = 10.0       # µm per slice layer
PLATE_W_PX      = 15120      # plate resolution, pixels
PLATE_H_PX      = 6230

# ── Preview downsampling ──
PREVIEW_SCALE = 4

PIXEL_X_UM = REAL_PIXEL_X_UM * PREVIEW_SCALE   # 112 µm effective
PIXEL_Y_UM = REAL_PIXEL_Y_UM * PREVIEW_SCALE   # 152 µm effective
LAYER_UM   = REAL_LAYER_UM * PREVIEW_SCALE      #  80 µm effective

# Convenience (mm)
PX_MM = PIXEL_X_UM / 1000
PZ_MM = PIXEL_Y_UM / 1000
LY_MM = LAYER_UM / 1000
