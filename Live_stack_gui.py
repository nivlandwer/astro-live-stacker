# live_stack_gui.py — PySide6 GUI for Live Telescope Stacking
#
# v3.0 changes:
# - Created new `calibration_util.py` file.
# - Moved ASISource, ImageView, and helper functions to `calibration_util.py`.
# - Main app now imports these classes.
# - Added `CalibrationUtility` QDialog in `calibration_util.py`.
# - "Open Calibration Utility" button now imports and shows this dialog.
# - Modified ASISource in util to yield raw frames (needed for calib).
# - Main AcqWorker modified to handle the new Frame(image_f32, image_raw)
# - Added logic to AcqWorker to apply dark correction.

from __future__ import annotations
import sys, os, glob, json
from dataclasses import dataclass
from typing import Optional, Tuple, Generator

import numpy as np
import cv2

try:
    from astropy.io import fits
except Exception:
    fits = None

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QComboBox, QSlider, QLineEdit, QFileDialog,
    QCheckBox, QGroupBox, QMessageBox, QSpinBox, QScrollArea, QDoubleSpinBox,
    QSplitter
)

# --- NEW: Import shared classes from calibration_util.py ---
try:
    from calibration_util import (
        robust_min_max, to_display_float, auto_contrast_u8, linear_u8, 
        np_to_qimage, Frame, Source, ASISource, ImageView, CalibrationUtility
    )
except ImportError:
    # This try-except block is crucial for user-friendliness
    print("ERROR: Could not find 'calibration_util.py'.")
    print("Make sure 'calibration_util.py' is in the same folder as this script.")
    # We can't show a QMessageBox yet because QApplication isn't running
    # So we'll show it right after app = QApplication() in main()
    IMPORT_ERROR_MESSAGE = "Could not find 'calibration_util.py'.\nMake sure it is in the same folder as this script."
    # Define dummy classes so the rest of the file can be parsed
    class Source: pass
    class ImageView(QLabel): pass
    class Frame: pass
    class ASISource(Source): pass
    class CalibrationUtility(QDialog): pass
    def robust_min_max(x, lo=1.0, hi=99.0): return 0.0, 1.0
    def to_display_float(img_any): return np.zeros((10,10,3), dtype=np.float32)
    def auto_contrast_u8(img_f32): return np.zeros((10,10,3), dtype=np.uint8)
    def linear_u8(img_f32, visual_gain=1.0): return np.zeros((10,10,3), dtype=np.uint8)
    def np_to_qimage(img_f32, auto_contrast=True, visual_gain=1.0): return QImage()
else:
    IMPORT_ERROR_MESSAGE = None


# --------------- sources (Local ones) ----------------
class FolderSource(Source):
    def __init__(self, path: str, pattern="*.png"):
        files = sorted(glob.glob(os.path.join(path, pattern)))
        if not files:
            pats = ["*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp"]
            f=[]; [f.extend(glob.glob(os.path.join(path,p))) for p in pats]
            files = sorted(f)
        if not files: raise FileNotFoundError(f"No images found in {path}")
        self.files = files
    def frames(self):
        for fp in self.files:
            img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if img is None: continue
            # Emulate the new Frame structure (no raw data)
            yield Frame(image_f32=to_display_float(img), image_raw=img)

class FitsSource(Source):
    def __init__(self, path: str):
        if fits is None: raise RuntimeError("astropy required for FITS")
        files = sorted(glob.glob(os.path.join(path, "*.fits")))
        if not files: raise FileNotFoundError(f"No FITS in {path}")
        self.files = files
    def frames(self):
        for fp in self.files:
            with fits.open(fp, memmap=True) as hdul:
                data = hdul[0].data
            # Don't auto-scale FITS, assume they are 0-1 or 0-65535
            data_f = data.astype(np.float32)
            if data.dtype == np.uint16: data_f /= 65535.0
            elif data.dtype == np.uint8: data_f /= 255.0
            
            if data_f.ndim == 2:
                img_f32 = np.dstack([data_f,data_f,data_f])
            else:
                img_f32 = data_f # Assume it's already 3-channel
                
            yield Frame(image_f32=img_f32, image_raw=data)

class WebcamSource(Source):
    def __init__(self, cam_index=0, width=0, height=0):
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if width>0:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height>0: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened(): raise RuntimeError("Could not open webcam")
    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok: break
            yield Frame(image_f32=to_display_float(frame), image_raw=frame)

class VideoSource(Source):
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): raise RuntimeError(f"Could not open {path}")
    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok: break
            yield Frame(image_f32=to_display_float(frame), image_raw=frame)

# ----- registration/stacking -----
class Registrar:
    def __init__(self, mode="ecc_affine"):
        if mode not in {"ecc_affine","phase"}: raise ValueError("registrar bad")
        self.mode = mode
    def _gray(self, img):
        if img.ndim==3 and img.shape[2]==3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return img.astype(np.float32)
    def _warp(self, img, M, shape):
        h,w = shape
        if img.ndim==2:
            return cv2.warpAffine(img, M, (w,h),
                                  flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REFLECT)
        chs = [cv2.warpAffine(img[...,c], M, (w,h),
                              flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT) for c in range(img.shape[2])]
        return np.stack(chs, axis=2)
    def register(self, moving, fixed):
        f = self._gray(fixed); m = self._gray(moving)
        if self.mode=="phase":
            shifts,_ = cv2.phaseCorrelate(f,m); dx,dy = shifts
            M = np.float32([[1,0,dx],[0,1,dy]])
            return self._warp(moving, M, f.shape), M
        warp_mode = cv2.MOTION_AFFINE
        fb = cv2.GaussianBlur(f,(0,0),1.0); mb = cv2.GaussianBlur(m,(0,0),1.0)
        W = np.eye(2,3,dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-6)
        try:
            _,W = cv2.findTransformECC(fb,mb,W,warp_mode,crit,None,5)
            return self._warp(moving,W,f.shape), W
        except cv2.error:
            return moving.copy(), np.eye(2,3,dtype=np.float32)

class Stacker:
    def __init__(self, sigma_clip=False, clip_sigma=3.0, stack_gain=1.0):
        self.count=0; self.mean=None; self.m2=None
        self.sigma_clip=sigma_clip; self.clip_sigma=clip_sigma
        self.stack_gain = float(stack_gain) # NEW: Accumulation gain

    def add(self, frame):
        x = frame.astype(np.float32) # Base frame
        
        if self.mean is None:
            # Store the first frame, scaled by the gain
            self.mean = x * self.stack_gain
            self.m2 = np.zeros_like(self.mean)
            self.count = 1
            return

        self.count += 1
        
        # Scale the incoming frame by the gain for accumulation
        x_gained = x * self.stack_gain
        
        if self.sigma_clip and self.count > 2:
            # Compare gained frame to gained mean, using gained std
            std = np.sqrt(self.m2 / (self.count - 1)) + 1e-6
            mask = np.abs(x_gained - self.mean) <= (self.clip_sigma * std)
            delta = (x_gained - self.mean) * mask
        else:
            delta = (x_gained - self.mean)
            
        self.mean += delta / self.count
        self.m2 += delta * (x_gained - self.mean) # new_mean is self.mean

    def reset(self):
        self.count=0; self.mean=None; self.m2=None
    
    def get(self)->Optional[np.ndarray]: 
        return self.mean # Mean already contains the gain

# --------------- worker ----------------
class AcqWorker(QObject):
    sig_raw = Signal(object); sig_reg = Signal(object); sig_stack = Signal(object)
    sig_error = Signal(str); sig_stopped = Signal()
    def __init__(self, cfg: dict, resume=False, parent=None):
        super().__init__(parent)
        self.cfg = cfg; self._running=False; self._resume=resume
        self._ref: Optional[np.ndarray] = None
        self._stacker = Stacker(
            sigma_clip=cfg.get("sigma_clip", False),
            stack_gain=cfg.get("stack_gain", 1.0) # Pass gain to stacker
        )
        self._registrar = Registrar(mode=cfg.get("registrar","ecc_affine"))
        self._last_raw: Optional[np.ndarray] = None
        
        # --- Dark Frame state ---
        self.dark_calib_enabled = cfg.get("enable_dark_calib", False)
        self.master_bias = None
        self.master_dark_rate = None
        self.needs_calib_load = self.dark_calib_enabled

    def set_resume_state(self, ref: Optional[np.ndarray], stacker: Stacker):
        self._ref = None if ref is None else ref.copy()
        self._stacker = stacker
        
    def last_raw(self): return self._last_raw # Corrected
    
    def build_source(self)->Source:
        src = self.cfg.get("source","webcam")
        if src=="folder": return FolderSource(self.cfg.get("path",""))
        if src=="fits":   return FitsSource(self.cfg.get("path",""))
        if src=="webcam": return WebcamSource(self.cfg.get("cam_index",0), self.cfg.get("width",0), self.cfg.get("height",0))
        if src=="video":  return VideoSource(self.cfg.get("path",""))
        if src=="asi":
            # --- ASISource is now imported ---
            return ASISource(
                asi_dll_path=self.cfg.get("asi_dll_path",""),
                gain=self.cfg.get("gain",200),
                exposure_ms=self.cfg.get("exposure_ms",200),
                binning=self.cfg.get("asi_bin",1),
                raw16=self.cfg.get("asi_raw16",False),
                bayer=(self.cfg.get("asi_bayer","rggb")),
                usb_bandwidth=self.cfg.get("asi_usb",80),
                timeout_ms=self.cfg.get("asi_timeout_ms") or None
            )
        raise ValueError("Unknown source")

    def _load_calibration_frames(self):
        """Load bias and calculate dark rate."""
        try:
            gain = self.cfg.get("gain")
            binning = self.cfg.get("asi_bin")
            calib_path = self.cfg.get("calib_path")
            
            if not calib_path:
                raise FileNotFoundError("Calibration path is not set.")
                
            if fits is None:
                raise RuntimeError("astropy is required to load FITS files.")

            bias_file = os.path.join(calib_path, f"master_bias_g{gain}_b{binning}.fits")
            dark_file = os.path.join(calib_path, f"master_dark_g{gain}_b{binning}.fits")

            if not os.path.exists(bias_file):
                raise FileNotFoundError(f"Master Bias file not found for G:{gain} B:{binning}\n{bias_file}")
            if not os.path.exists(dark_file):
                raise FileNotFoundError(f"Master Dark file not found for G:{gain} B:{binning}\n{dark_file}")
            
            # Load Bias
            with fits.open(bias_file) as hdul:
                self.master_bias = hdul[0].data.astype(np.float64)
                
            # Load Dark and calculate Rate
            with fits.open(dark_file) as hdul:
                master_dark_data = hdul[0].data.astype(np.float64)
                dark_exp_s = float(hdul[0].header['EXPOSURE'])
                if dark_exp_s <= 0:
                    raise ValueError("Master Dark file has invalid exposure time <= 0.")

            # Rate = (Dark - Bias) / Exposure
            # We clip at 0 to avoid negative dark current values
            self.master_dark_rate = np.clip(
                (master_dark_data - self.master_bias) / dark_exp_s,
                0, 
                np.inf
            )
            
            # Convert bias to float64 for calculations
            self.master_bias = self.master_bias.astype(np.float64)
            self.needs_calib_load = False # Success!
            
        except Exception as e:
            self.dark_calib_enabled = False # Disable on error
            self.needs_calib_load = False
            self.sig_error.emit(f"Dark calib. disabled: {e}")

    def _apply_dark_correction(self, raw_frame: np.ndarray) -> np.ndarray:
        """Subtracts synthetic dark frame from the raw_frame."""
        if not self.dark_calib_enabled or self.master_bias is None or self.master_dark_rate is None:
            return raw_frame
            
        # Get current exposure in seconds
        exp_s = self.cfg.get("exposure_ms", 1) / 1000.0
        
        # Synthetic_Dark = Bias + (Rate * Exposure)
        synthetic_dark = self.master_bias + (self.master_dark_rate * exp_s)
        
        # Corrected = Raw - Synthetic_Dark
        # Clip at 0. Convert to float first to avoid wrap-around.
        corrected_f64 = raw_frame.astype(np.float64) - synthetic_dark
        corrected_f64 = np.clip(corrected_f64, 0, 65535.0) 
        
        return corrected_f64.astype(raw_frame.dtype)

    def _debayer(self, raw: np.ndarray) -> np.ndarray:
        # --- This function now lives in the worker ---
        # It takes the (potentially corrected) raw frame and returns a float32
        
        # Get config
        is_raw16 = self.cfg.get("asi_raw16", False)
        preblur_sigma = float(self.cfg.get("asi_preblur_sigma", 0.0))
        debayer_method = self.cfg.get("asi_debayer_method", "bilinear")
        bayer_pattern = self.cfg.get("asi_bayer", "rggb").lower()
        no_debayer = self.cfg.get("asi_no_debayer", False)
        no_swap_rb = self.cfg.get("asi_no_swap_rb", False)
        
        if no_debayer:
            g = raw.astype(np.float32) / (65535.0 if is_raw16 else 255.0)
            return np.dstack([g,g,g]).astype(np.float32)
            
        work = raw
        if preblur_sigma > 0.0:
            work = cv2.GaussianBlur(raw, (0,0), preblur_sigma)
            
        if debayer_method == "superpixel":
            # Need to import demosaic_superpixel from util
            from calibration_util import demosaic_superpixel
            img_f = demosaic_superpixel(work, bayer_pattern)
        else:
            # Need to import maps from util
            from calibration_util import _OCV_BILINEAR, _OCV_VNG, _OCV_EA
            
            cvt_bilinear = _OCV_BILINEAR.get(bayer_pattern, cv2.COLOR_BayerRG2BGR)
            code = cvt_bilinear
            if debayer_method == "vng":
                code = _OCV_VNG.get(bayer_pattern, cvt_bilinear)
            elif debayer_method in ("edge_aware","ea","edgeaware"):
                code = _OCV_EA.get(bayer_pattern, cvt_bilinear)
                
            bgr = cv2.cvtColor(work, code)
            img_f = bgr.astype(np.float32) / (65535.0 if is_raw16 else 255.0)
            
        if not no_swap_rb: # Logic is "No R/B swap"
            img_f = img_f[..., ::-1] # BGR -> RGB
            
        return img_f

    @Slot()
    def run(self):
        self._running=True
        if not self._resume:
            # self._stacker.reset() # Reset is implicit in new Stacker creation
            self._ref=None
            
        # --- Load calibration frames if needed ---
        if self.needs_calib_load:
            self._load_calibration_frames()
            
        try:
            src = self.build_source()
        except Exception as e:
            self.sig_error.emit(str(e)); self.sig_stopped.emit(); return
        try:
            for fr in src.frames(): # fr is now a Frame(image_f32, image_raw)
                if not self._running: break
                
                # ASI source gives raw data, other sources give BGR
                # We need to handle this
                source_is_asi = (self.cfg.get("source") == "asi")
                
                if source_is_asi:
                    raw_for_processing = fr.image_raw
                    
                    # --- 1. Apply Dark Correction (only for ASI) ---
                    if self.dark_calib_enabled:
                        raw_for_processing = self._apply_dark_correction(raw_for_processing)
                    
                    # --- 2. Debayer ---
                    img_f32 = self._debayer(raw_for_processing)
                else:
                    # Other sources (webcam, video, etc.) are already debayered
                    img_f32 = fr.image_f32 # Use the pre-converted float image
                
                # --- 3. Emit raw (debayered) frame ---
                self._last_raw = img_f32
                self.sig_raw.emit(img_f32) 
                
                if self._ref is None: 
                    self._ref = img_f32.copy()
                    
                # --- 4. Register ---
                moved,_ = self._registrar.register(img_f32, self._ref)
                self.sig_reg.emit(moved)
                
                # --- 5. Stack ---
                self._stacker.add(moved)
                st = self._stacker.get(); 
                if st is not None: 
                    self.sig_stack.emit(st)
                    
        except Exception as e:
            import traceback
            print(traceback.format_exc()) # Print full traceback
            self.sig_error.emit(f"{type(e).__name__}: {e}") # Show cleaner error
        finally:
            if hasattr(src, 'close'): # Close ASI camera
                src.close()
            self.sig_stopped.emit()
            
    def stop(self): self._running=False
    def get_state(self): return self._ref, self._stacker, self._last_raw

# --------------- GUI ----------------
# ImageView is now imported from calibration_util

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Live Telescope Stacking — PySide6")
        self.worker_thread=None; self.worker=None
        self._ref=None; self._stacker=Stacker(False); self._last_raw=None
        self._last_reg=None; self._last_stack=None
        self.calib_dialog = None # NEW: Placeholder for dialog
        
        self._build_ui(); self._bind_actions(); self._apply_default_ui_values()
        self._on_maxsize_toggled(self.chk_maxsize.isChecked())

    def _build_ui(self):
        central = QWidget(); root = QHBoxLayout(central)

        # --- Create views first, so they exist for _build_display_group ---
        self.raw_view = ImageView("Raw"); self.reg_view = ImageView("Registered"); self.stack_view = ImageView("Stack")

        # Create a content widget and layout for all control panels
        controls_content_widget = QWidget()
        controls_layout = QVBoxLayout(controls_content_widget)
        controls_layout.addWidget(self._build_config_group())
        controls_layout.addWidget(self._build_dark_calib_group()) # NEW
        controls_layout.addWidget(self._build_ops_group())
        controls_layout.addWidget(self._build_display_group()) # NEW Display Group
        controls_layout.addWidget(self._build_save_group())
        controls_layout.addStretch(1) # Add stretch inside the layout

        # Create the scroll area and add the content widget to it
        controls_scroll = QScrollArea()
        controls_scroll.setWidget(controls_content_widget)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Give the scroll area a reasonable minimum width
        controls_scroll.setMinimumWidth(360) 
        # Set a fixed width to ensure it doesn't try to expand
        controls_scroll.setMaximumWidth(420) 

        # --- NEW: Use a QSplitter for the views ---
        views_splitter = QSplitter(Qt.Vertical)
        views_splitter.addWidget(self.raw_view)
        views_splitter.addWidget(self.reg_view)
        views_splitter.addWidget(self.stack_view)
        # Optional: Set initial relative sizes (e.g., evenly)
        views_splitter.setSizes([300, 300, 300]) 
        
        root.addWidget(controls_scroll, 0) # Add the scroll area to the root layout
        root.addWidget(views_splitter, 1) # Add the new splitter
        
        self.setCentralWidget(central); self.resize(1200,900)

    def _build_config_group(self)->QGroupBox:
        g = QGroupBox("Configuration"); grid = QGridLayout(g); row=0
        # Dropdowns
        self.cb_source = QComboBox(); self.cb_source.addItems(["webcam","video","folder","fits","asi"])
        self.cb_registrar = QComboBox(); self.cb_registrar.addItems(["ecc_affine","phase"])
        self.cb_bayer = QComboBox(); self.cb_bayer.addItems(["rggb","bggr","grbg","gbrg"])  # default rggb
        self.cb_asi_bin = QComboBox(); self.cb_asi_bin.addItems(["1","2"])
        grid.addWidget(QLabel("Source"), row,0); grid.addWidget(self.cb_source,row,1); row+=1
        grid.addWidget(QLabel("Registrar"), row,0); grid.addWidget(self.cb_registrar,row,1); row+=1
        grid.addWidget(QLabel("ASI Bayer"), row,0); grid.addWidget(self.cb_bayer,row,1); row+=1
        grid.addWidget(QLabel("ASI Bin"), row,0); grid.addWidget(self.cb_asi_bin,row,1); row+=1
        # Path
        self.ed_path = QLineEdit(); btn_browse = QPushButton("Browse…")
        hb = QHBoxLayout(); hb.addWidget(self.ed_path); hb.addWidget(btn_browse)
        w_path = QWidget(); w_path.setLayout(hb)
        grid.addWidget(QLabel("Path (video/folder/FITS)"), row,0); grid.addWidget(w_path,row,1); row+=1
        btn_browse.clicked.connect(self._browse_path)
        # Width/Height + Max
        self.chk_maxsize = QCheckBox("Max")
        def add_slider(name, minv, maxv, init, step=1):
            nonlocal row
            lab = QLabel(name)
            sld = QSlider(Qt.Horizontal); sld.setRange(minv,maxv); sld.setValue(init); sld.setSingleStep(step); sld.setPageStep(step)
            edt = QSpinBox(); edt.setRange(minv,maxv); edt.setValue(init); edt.setSingleStep(step)
            sld.valueChanged.connect(edt.setValue); edt.valueChanged.connect(sld.setValue)
            hb2 = QHBoxLayout(); hb2.addWidget(sld); hb2.addWidget(edt); w = QWidget(); w.setLayout(hb2)
            grid.addWidget(lab,row,0); grid.addWidget(w,row,1); row+=1
            return sld, edt
        # These controls are now for Webcam/Video only
        self.sld_w, self.ed_w = add_slider("Width (Webcam)", 0, 4096, 0, 2)
        self.sld_h, self.ed_h = add_slider("Height (Webcam)", 0, 4096, 0, 2)
        grid.addWidget(self.chk_maxsize, row-2, 2, 2, 1)
        self.chk_maxsize.toggled.connect(self._on_maxsize_toggled)
        # Webcam extras
        self.sld_camidx, self.ed_camidx = add_slider("Cam Index", 0, 5, 0, 1)
        # Demosaic controls
        self.cb_debayer_method = QComboBox(); self.cb_debayer_method.addItems(["bilinear","vng","edge_aware","superpixel"]) 
        grid.addWidget(QLabel("ASI Debayer"), row,0); grid.addWidget(self.cb_debayer_method,row,1); row+=1
        self.sld_preblur, self.ed_preblur = add_slider("Pre-debayer blur σ", 0, 3, 0, 1)
        # Registrar / stack
        self.chk_sigma = QCheckBox("Sigma clip")
        grid.addWidget(self.chk_sigma, row,0,1,2); row+=1
        # NEW: Stack Gain (N)
        self.sld_stack_gain, self.ed_stack_gain = add_slider("Stack Gain (N)", 1, 100, 1, 1)
        # ASI DLL
        self.ed_dll = QLineEdit(); btn_dll = QPushButton("ASICamera2.dll…")
        hb2 = QHBoxLayout(); hb2.addWidget(self.ed_dll); hb2.addWidget(btn_dll)
        w_dll = QWidget(); w_dll.setLayout(hb2)
        grid.addWidget(QLabel("ASI DLL path"), row,0); grid.addWidget(w_dll,row,1); row+=1
        btn_dll.clicked.connect(self._browse_dll)
        # ASI numerics
        self.sld_gain, self.ed_gain = add_slider("Gain", 0, 600, 200, 1)
        self.sld_exp, self.ed_exp = add_slider("Exposure (ms)", 1, 60000, 200, 1)
        self.sld_usb, self.ed_usb = add_slider("USB bandwidth", 40, 100, 80, 1)
        self.sld_timeout, self.ed_timeout = add_slider("Timeout (ms, 0=auto)", 0, 20000, 0, 50)
        # ASI checkboxes
        self.chk_raw16 = QCheckBox("RAW16")
        self.chk_no_debayer = QCheckBox("No debayer (grayscale)")
        self.chk_no_swap_rb = QCheckBox("No R/B swap")
        grid.addWidget(self.chk_raw16, row,0); grid.addWidget(self.chk_no_debayer,row,1); row+=1
        grid.addWidget(self.chk_no_swap_rb, row,0); row+=1
        g.setLayout(grid); return g

    def _build_dark_calib_group(self)->QGroupBox:
        g = QGroupBox("Dark Correction"); grid = QGridLayout(g); row=0
        
        self.chk_enable_dark_calib = QCheckBox("Enable Dark Correction")
        grid.addWidget(self.chk_enable_dark_calib, row, 0, 1, 2); row+=1

        self.btn_open_calib = QPushButton("Open Calibration Utility...")
        grid.addWidget(self.btn_open_calib, row, 0, 1, 2); row+=1

        lab_calib_path = QLabel("Calibration Library Path")
        self.ed_calib_path = QLineEdit()
        self.btn_browse_calib_path = QPushButton("Browse…")
        
        hb_path = QHBoxLayout()
        hb_path.addWidget(self.ed_calib_path)
        hb_path.addWidget(self.btn_browse_calib_path)
        w_path = QWidget(); w_path.setLayout(hb_path)
        
        grid.addWidget(lab_calib_path, row, 0); grid.addWidget(w_path, row, 1); row+=1
        
        g.setLayout(grid)
        return g

    def _build_ops_group(self)->QGroupBox:
        g = QGroupBox("Operation"); hb = QHBoxLayout(g)
        self.btn_start = QPushButton("Start"); self.btn_stop = QPushButton("Stop"); self.btn_resume = QPushButton("Resume")
        hb.addWidget(self.btn_start); hb.addWidget(self.btn_stop); hb.addWidget(self.btn_resume); return g

    def _build_display_group(self)->QGroupBox:
        g = QGroupBox("Display"); grid = QGridLayout(g); row=0
        
        self.chk_autocontrast = QCheckBox("Auto-contrast display")
        grid.addWidget(self.chk_autocontrast, row, 0, 1, 2); row+=1

        def add_gain_control(name, view: ImageView): # MODIFIED
            nonlocal row
            lab = QLabel(name)
            # Slider from 1 to 500 (representing 0.1 to 50.0)
            sld = QSlider(Qt.Horizontal); sld.setRange(1, 500); sld.setValue(10); # 10 = 1.0
            
            # NEW: QDoubleSpinBox
            edt = QDoubleSpinBox()
            edt.setRange(0.1, 50.0); edt.setValue(1.0); edt.setSingleStep(0.1); edt.setDecimals(1)
            edt.setFixedWidth(60) # A bit wider for "50.0"
            
            hb = QHBoxLayout(); hb.addWidget(sld); hb.addWidget(edt) # Replaced QLabel with QDoubleSpinBox
            w = QWidget(); w.setLayout(hb)
            grid.addWidget(lab,row,0); grid.addWidget(w,row,1); row+=1
            
            # --- MODIFIED Connections ---
            def sld_to_edt_and_view(value_int):
                gain_float = value_int / 10.0
                edt.blockSignals(True)
                edt.setValue(gain_float)
                edt.blockSignals(False)
                view.set_visual_gain(gain_float) # Update view
            sld.valueChanged.connect(sld_to_edt_and_view)

            def edt_to_sld_and_view(value_float):
                sld.blockSignals(True)
                sld.setValue(int(value_float * 10))
                sld.blockSignals(False)
                view.set_visual_gain(value_float) # Update view
            edt.valueChanged.connect(edt_to_sld_and_view)
            
            return sld, edt # Return slider and spinbox

        # MODIFIED: Pass views
        self.sld_gain_raw, self.edt_gain_raw = add_gain_control("Raw Visual Gain", self.raw_view)
        self.sld_gain_reg, self.edt_gain_reg = add_gain_control("Reg Visual Gain", self.reg_view)
        self.sld_gain_stack, self.edt_gain_stack = add_gain_control("Stack Visual Gain", self.stack_view)

        g.setLayout(grid)
        return g

    def _build_save_group(self)->QGroupBox:
        g = QGroupBox("Save / Config"); hb = QHBoxLayout(g)
        self.btn_save = QPushButton("Save image (stack/last)")
        self.btn_save_cfg = QPushButton("Save Config…")
        self.btn_load_cfg = QPushButton("Load Config…")
        hb.addWidget(self.btn_save); hb.addWidget(self.btn_save_cfg); hb.addWidget(self.btn_load_cfg)
        return g

    def _bind_actions(self):
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_resume.clicked.connect(self.on_resume)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_save_cfg.clicked.connect(self.on_save_config)
        self.btn_load_cfg.clicked.connect(self.on_load_config)
        
        # --- NEW Calibration binds ---
        self.btn_browse_calib_path.clicked.connect(self._browse_calib_path) 
        self.btn_open_calib.clicked.connect(self.on_open_calib_util)
        
        # Display controls
        self.chk_autocontrast.stateChanged.connect(self._on_toggle_autocontrast)

    def _apply_default_ui_values(self):
        self.cb_source.setCurrentText("webcam")
        self.cb_registrar.setCurrentText("ecc_affine")
        self.chk_sigma.setChecked(False)
        self.ed_stack_gain.setValue(1) # NEW
        self.ed_camidx.setValue(0); self.ed_w.setValue(0); self.ed_h.setValue(0); self.chk_maxsize.setChecked(True)
        # Display
        self.chk_autocontrast.setChecked(False)
        self.edt_gain_raw.setValue(1.0) # This will auto-update sld_gain_raw
        self.edt_gain_reg.setValue(1.0)
        self.edt_gain_stack.setValue(1.0)
        self._on_toggle_autocontrast(False) # Set initial slider state
        
        for v in (self.raw_view, self.reg_view, self.stack_view):
            v.set_auto_contrast(False)
            v.set_visual_gain(1.0)
            
        # Demosaic defaults
        self.cb_debayer_method.setCurrentText("bilinear")
        self.ed_preblur.setValue(0)
        # ASI defaults
        self.cb_bayer.setCurrentText("rggb")
        self.cb_asi_bin.setCurrentText("1")
        self.chk_raw16.setChecked(False)
        self.ed_gain.setValue(200); self.ed_exp.setValue(200)
        self.ed_usb.setValue(80); self.chk_no_debayer.setChecked(False)
        self.chk_no_swap_rb.setChecked(True)
        self.ed_timeout.setValue(0)

    def _on_maxsize_toggled(self, checked: bool):
        for w in (self.sld_w, self.ed_w, self.sld_h, self.ed_h):
            w.setEnabled(not checked)
        if checked:
            self.ed_w.setValue(0); self.ed_h.setValue(0)

    # helpers
    def _browse_path(self):
        src = self.cb_source.currentText()
        if src in ("folder","fits"):
            d = QFileDialog.getExistingDirectory(self, "Choose folder")
            if d: self.ed_path.setText(d)
        else:
            f,_ = QFileDialog.getOpenFileName(self, "Choose file (video/image)", "", "All (*.*)")
            if f: self.ed_path.setText(f)
            
    def _browse_calib_path(self): # NEW
        d = QFileDialog.getExistingDirectory(self, "Choose Calibration Library Folder")
        if d: self.ed_calib_path.setText(d)
            
    def _browse_dll(self):
        f,_ = QFileDialog.getOpenFileName(self, "Select ASICamera2.dll/.so", "", "Library (*.dll *.so *.dylib);;All (*.*)")
        if f: self.ed_dll.setText(f)

    # -------- Config persistence --------
    def gather_cfg(self)->dict:
        width = int(self.ed_w.value()); height = int(self.ed_h.value())
        if width and (width % 2): width -= 1
        if height and (height % 2): height -= 1
        cfg = dict(
            source=self.cb_source.currentText(),
            path=self.ed_path.text().strip(),
            cam_index=int(self.ed_camidx.value()),
            width=width, # For webcam
            height=height, # For webcam
            registrar=self.cb_registrar.currentText(),
            sigma_clip=self.chk_sigma.isChecked(),
            stack_gain=int(self.ed_stack_gain.value()), # NEW
            # Dark Correction
            enable_dark_calib=self.chk_enable_dark_calib.isChecked(), # NEW
            calib_path=self.ed_calib_path.text().strip(), # NEW
            # Display
            display_auto_contrast=self.chk_autocontrast.isChecked(),
            display_gain_raw=float(self.edt_gain_raw.value()), # UPDATED
            display_gain_reg=float(self.edt_gain_reg.value()), # UPDATED
            display_gain_stack=float(self.edt_gain_stack.value()), # UPDATED
            # Demosaic
            asi_debayer_method=self.cb_debayer_method.currentText(),
            asi_preblur_sigma=int(self.ed_preblur.value()),
            # ASI
            asi_dll_path=self.ed_dll.text().strip(),
            gain=int(self.ed_gain.value()),
            exposure_ms=int(self.ed_exp.value()),
            asi_bin=int(self.cb_asi_bin.currentText()),
            asi_raw16=self.chk_raw16.isChecked(),
            asi_bayer=self.cb_bayer.currentText(),
            asi_usb=int(self.ed_usb.value()),
            asi_no_debayer=self.chk_no_debayer.isChecked(),
            asi_no_swap_rb=self.chk_no_swap_rb.isChecked(),
            asi_timeout_ms=int(self.ed_timeout.value()),
        )
        return cfg

    def apply_cfg(self, cfg: dict):
        def set_combo(cb: QComboBox, val: str):
            idx = cb.findText(str(val))
            if idx >= 0: cb.setCurrentIndex(idx)
        set_combo(self.cb_source, cfg.get("source", self.cb_source.currentText()))
        set_combo(self.cb_registrar, cfg.get("registrar", self.cb_registrar.currentText()))
        set_combo(self.cb_bayer, cfg.get("asi_bayer", self.cb_bayer.currentText()))
        set_combo(self.cb_asi_bin, str(cfg.get("asi_bin", int(self.cb_asi_bin.currentText()))))
        self.ed_path.setText(str(cfg.get("path", self.ed_path.text())))
        w = int(cfg.get("width", 0)); h = int(cfg.get("height", 0))
        is_max = (w==0 or h==0)
        self.chk_maxsize.setChecked(is_max)
        if not is_max:
            self.ed_w.setValue(max(0, w - (w % 2)))
            self.ed_h.setValue(max(0, h - (h % 2)))
        self.ed_camidx.setValue(int(cfg.get("cam_index", self.ed_camidx.value())))
        self.chk_sigma.setChecked(bool(cfg.get("sigma_clip", self.chk_sigma.isChecked())))
        self.ed_stack_gain.setValue(int(cfg.get("stack_gain", self.ed_stack_gain.value()))) # NEW
        
        # Dark Correction
        self.chk_enable_dark_calib.setChecked(bool(cfg.get("enable_dark_calib", False))) # NEW
        self.ed_calib_path.setText(str(cfg.get("calib_path", ""))) # NEW
        
        # Display
        ac = bool(cfg.get("display_auto_contrast", self.chk_autocontrast.isChecked()))
        self.chk_autocontrast.setChecked(ac)
        
        raw_gain = float(cfg.get("display_gain_raw", 1.0)) # UPDATED
        reg_gain = float(cfg.get("display_gain_reg", 1.0)) # UPDATED
        stack_gain_val = float(cfg.get("display_gain_stack", 1.0)) # UPDATED
        
        self.edt_gain_raw.setValue(raw_gain) # UPDATED
        self.edt_gain_reg.setValue(reg_gain) # UPDATED
        self.edt_gain_stack.setValue(stack_gain_val) # UPDATED
        
        self.raw_view.set_visual_gain(raw_gain) # UPDATED
        self.reg_view.set_visual_gain(reg_gain) # UPDATED
        self.stack_view.set_visual_gain(stack_gain_val) # UPDATED

        for v in (self.raw_view, self.reg_view, self.stack_view):
            v.set_auto_contrast(ac)
            
        self._on_toggle_autocontrast(ac) # Set slider enabled state
            
        # Demosaic
        set_combo(self.cb_debayer_method, cfg.get("asi_debayer_method", self.cb_debayer_method.currentText()))
        self.ed_preblur.setValue(int(cfg.get("asi_preblur_sigma", self.ed_preblur.value())))
        # ASI
        self.ed_dll.setText(str(cfg.get("asi_dll_path", self.ed_dll.text())))
        self.ed_gain.setValue(int(cfg.get("gain", self.ed_gain.value())))
        self.ed_exp.setValue(int(cfg.get("exposure_ms", self.ed_exp.value())))
        self.ed_usb.setValue(int(cfg.get("asi_usb", self.ed_usb.value())))
        self.chk_raw16.setChecked(bool(cfg.get("asi_raw16", self.chk_raw16.isChecked())))
        self.chk_no_debayer.setChecked(bool(cfg.get("asi_no_debayer", self.chk_no_debayer.isChecked())))
        self.chk_no_swap_rb.setChecked(bool(cfg.get("asi_no_swap_rb", self.chk_no_swap_rb.isChecked())))
        self.ed_timeout.setValue(int(cfg.get("asi_timeout_ms", self.ed_timeout.value())))

    def on_save_config(self):
        cfg = self.gather_cfg()
        fn, _ = QFileDialog.getSaveFileName(self, "Save configuration", "config.json", "JSON (*.json)")
        if not fn:
            return
        try:
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save Config", f"Failed to save config:\n{e}")
        else:
            QMessageBox.information(self, "Save Config", f"Saved configuration to:\n{fn}")

    def on_load_config(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load configuration", "", "JSON (*.json)")
        if not fn:
            return
        try:
            with open(fn, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if not isinstance(cfg, dict):
                raise ValueError("Config root must be a JSON object")
        except Exception as e:
            QMessageBox.critical(self, "Load Config", f"Failed to load config:\n{e}")
            return
        try:
            self.apply_cfg(cfg)
        except Exception as e:
            QMessageBox.critical(self, "Load Config", f"Config applied with errors:\n{e}")
        else:
            QMessageBox.information(self, "Load Config", f"Loaded configuration from:\n{fn}")

    # --- NEW: Open Calibration Utility ---
    def on_open_calib_util(self):
        asi_dll_path = self.ed_dll.text().strip()
        calib_path = self.ed_calib_path.text().strip()
        
        if not calib_path:
            QMessageBox.warning(self, "Calibration", "Please set a Calibration Library Path first.")
            return
            
        if not os.path.exists(calib_path):
            try:
                os.makedirs(calib_path)
            except Exception as e:
                QMessageBox.critical(self, "Calibration Path", f"Could not create path:\n{calib_path}\n{e}")
                return

        if self.cb_source.currentText() != "asi":
            QMessageBox.warning(self, "Calibration", "Calibration utility only works when 'asi' is the selected source.")
            return
            
        if not asi_dll_path or not os.path.exists(asi_dll_path):
             QMessageBox.warning(self, "Calibration", "ASI DLL path is not set or not valid. Please set it in the Configuration panel.")
             return

        # Pass current ASI settings to the dialog
        current_gain = self.ed_gain.value()
        current_bin = int(self.cb_asi_bin.currentText())

        try:
            # We create a new dialog each time.
            # We pass 'self' as the parent so it's centered on the main window.
            self.calib_dialog = CalibrationUtility(
                asi_dll_path=asi_dll_path,
                calib_path=calib_path,
                initial_gain=current_gain,
                initial_bin=current_bin,
                parent=self
            )
            self.calib_dialog.show() # Use show() for non-modal
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open calibration utility:\n{e}")
            import traceback
            print(traceback.format_exc())

    # threading
    def _connect_worker_signals(self):
        self.worker.sig_raw.connect(self.on_raw)
        self.worker.sig_reg.connect(self.on_reg)
        self.worker.sig_stack.connect(self.on_stack)
        self.worker.sig_error.connect(self.on_error)
        self.worker.sig_stopped.connect(self.on_worker_stopped)
        
    def _start_worker(self, resume=False):
        if self.worker_thread: # Stop any existing worker first
            self.on_stop()
            
        cfg = self.gather_cfg()
        self.worker = AcqWorker(cfg, resume=resume)
        if resume: self.worker.set_resume_state(self._ref, self._stacker)
        self.worker_thread = QThread(); self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self._connect_worker_signals(); self.worker_thread.start()
        
    def on_start(self):
        self.on_stop() # Stop any previous run
        # Create new stacker with new gain setting
        self._stacker=Stacker(
            sigma_clip=self.chk_sigma.isChecked(), 
            stack_gain=int(self.ed_stack_gain.value())
        )
        self._ref=None; self._last_raw=None; self._last_reg=None; self._last_stack=None
        self._start_worker(resume=False)
        
    def on_stop(self):
        if self.worker: 
            # Save state before stopping
            self._ref, self._stacker, self._last_raw = self.worker.get_state()
            self.worker.stop()
        if self.worker_thread:
            self.worker_thread.quit(); self.worker_thread.wait(2000) # Wait 2s
        self.worker=None; self.worker_thread=None
        
    def on_resume(self):
        if self.worker: return # Already running
        self._start_worker(resume=True)
        
    def on_save(self):
        img=None
        if self._stacker and self._stacker.get() is not None: img=self._stacker.get(); self._last_stack = img
        elif self._last_raw is not None: img=self.last_raw
        if img is None:
            QMessageBox.information(self,"Save","No image to save yet."); return
        f,_ = QFileDialog.getSaveFileName(self,"Save image","stack.png","PNG (*.png);;JPEG (*.jpg)")
        if not f: return
        
        # Save based on current display settings (auto or manual gain)
        if self.chk_autocontrast.isChecked():
            view_u8 = auto_contrast_u8(img)
        else:
            # Use the stack view's visual gain for saving
            stack_gain = self.edt_gain_stack.value() # UPDATED
            view_u8 = linear_u8(img, visual_gain=stack_gain)
            
        if view_u8.ndim==3 and view_u8.shape[2]==3:
            # Convert to BGR for cv2.imwrite
            out = cv2.cvtColor(view_u8, cv2.COLOR_RGB2BGR)
        else: 
            out = view_u8
        cv2.imwrite(f, out)
        
    # worker signals
    def on_raw(self, img): 
        self._last_raw=img; 
        self.raw_view.set_image(img)
    def on_reg(self, img): 
        self._last_reg=img; 
        self.reg_view.set_image(img)
    def on_stack(self, img):
        self._last_stack = img
        self.stack_view.set_image(img)
        if self.worker: 
            # This might be slow, consider doing it less often
            try:
                self._ref, self._stacker, self._last_raw = self.worker.get_state()
            except Exception:
                pass # worker might be shutting down
            
    def on_error(self, msg): 
        QMessageBox.critical(self,"Error",msg)
        self.on_stop() # Stop worker on error
        
    def on_worker_stopped(self): 
        # Clean up worker/thread
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def _on_toggle_autocontrast(self, state):
        enabled = bool(state)
        # Disable gain sliders AND spinboxes if auto-contrast is on
        self.sld_gain_raw.setEnabled(not enabled)
        self.sld_gain_reg.setEnabled(not enabled)
        self.sld_gain_stack.setEnabled(not enabled)
        self.edt_gain_raw.setEnabled(not enabled) # NEW
        self.edt_gain_reg.setEnabled(not enabled) # NEW
        self.edt_gain_stack.setEnabled(not enabled) # NEW

        for v in (self.raw_view, self.reg_view, self.stack_view):
            v.set_auto_contrast(enabled)
            
        # Re-render all views with the new setting
        if self._last_raw is not None:
            self.raw_view.set_image(self._last_raw)
        if self._last_reg is not None:
            self.reg_view.set_image(self._last_reg)
        if self._last_stack is not None:
            self.stack_view.set_image(self._last_stack)
            
    def closeEvent(self, event):
        # Clean up calibration dialog if it's open
        if self.calib_dialog:
            self.calib_dialog.close()
        # Clean up worker thread
        self.on_stop()
        event.accept()

# --------------- main ----------------
def main():
    app = QApplication(sys.argv)
    
    # Handle the import error here
    if IMPORT_ERROR_MESSAGE:
        QMessageBox.critical(None, "Import Error", IMPORT_ERROR_MESSAGE)
        sys.exit(1)
        
    w = MainWindow(); w.show(); sys.exit(app.exec())

if __name__=="__main__":
    main()
