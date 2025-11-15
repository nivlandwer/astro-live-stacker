# calibration_util.py — Calibration Utility for Live Stacking
# Contains the QDialog for the calibration UI, a worker for capturing,
# and shared classes like ImageView and ASISource.

#from __future{annotations}
import sys, os, time, json
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
    QApplication, QDialog, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QComboBox, QSlider, QLineEdit, QFileDialog,
    QCheckBox, QGroupBox, QMessageBox, QSpinBox, QProgressBar
)

# ---------------- utils ----------------
# Moved from main app
def robust_min_max(x: np.ndarray, lo=1.0, hi=99.0) -> Tuple[float, float]:
    lo_v = np.percentile(x, lo); hi_v = np.percentile(x, hi)
    if hi_v <= lo_v: hi_v = lo_v + 1e-6
    return float(lo_v), float(hi_v)

@dataclass
class Frame:
    image_f32: np.ndarray
    image_raw: np.ndarray # We need the raw data for calibration

def to_display_float(img_any: np.ndarray) -> np.ndarray:
    """Convert BGR/Gray/u16/u8 → float32 [0..1], 3 channels."""
    scale = 65535.0 if img_any.dtype == np.uint16 else 255.0
    if img_any.ndim == 2:
        f = img_any.astype(np.float32)/scale
        return np.dstack([f,f,f])
    if img_any.ndim == 3 and img_any.shape[2] == 3:
        return img_any.astype(np.float32)/scale
    if img_any.ndim == 3 and img_any.shape[2] == 4:
        return img_any[..., :3].astype(np.float32)/scale
    return img_any.astype(np.float32)/scale

def auto_contrast_u8(img_f32: np.ndarray) -> np.ndarray:
    lo, hi = robust_min_max(img_f32)
    view = np.clip((img_f32 - lo)/(hi-lo), 0, 1)
    return (view*255).astype(np.uint8)

def linear_u8(img_f32: np.ndarray, visual_gain: float = 1.0) -> np.ndarray:
    view = np.clip(img_f32 * visual_gain, 0, 1)
    return (view * 255).astype(np.uint8)

def np_to_qimage(img_f32: np.ndarray, auto_contrast=True, visual_gain=1.0) -> QImage:
    if auto_contrast:
        proc = auto_contrast_u8(img_f32)
    else:
        proc = linear_u8(img_f32, visual_gain=visual_gain)
    if proc.ndim == 2:
        h, w = proc.shape
        return QImage(proc.data, w, h, w, QImage.Format_Grayscale8).copy()
    if proc.shape[2] == 3:
        proc = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        h, w, ch = proc.shape
        return QImage(proc.data, w, h, ch*w, QImage.Format_RGB888).copy()
    h, w = proc.shape[:2]
    return QImage(proc.data, w, h, w, QImage.Format_Grayscale8).copy()

# ------- Demosaic helpers (Needed for ASISource) -------
_OCV_BILINEAR = {
    "rggb": cv2.COLOR_BayerRG2BGR, "bggr": cv2.COLOR_BayerBG2BGR,
    "grbg": cv2.COLOR_BayerGR2BGR, "gbrg": cv2.COLOR_BayerGB2BGR,
}
_OCV_VNG = {
    k: getattr(cv2, f"COLOR_Bayer{pat.upper()}2BGR_VNG", _OCV_BILINEAR[k])
    for k, pat in {"rggb":"RG", "bggr":"BG", "grbg":"GR", "gbrg":"GB"}.items()
}
_OCV_EA = {
    k: getattr(cv2, f"COLOR_Bayer{pat.upper()}2BGR_EA", _OCV_BILINEAR[k])
    for k, pat in {"rggb":"RG", "bggr":"BG", "grbg":"GR", "gbrg":"GB"}.items()
}

def demosaic_superpixel(raw: np.ndarray, pattern: str) -> np.ndarray:
    pat = pattern.lower(); H, W = raw.shape
    H2, W2 = H - (H % 2), W - (W % 2); r = g1 = g2 = b = None
    if pat == "rggb":
        r  = raw[0:H2:2, 0:W2:2]; g1 = raw[0:H2:2, 1:W2:2]; g2 = raw[1:H2:2, 0:W2:2]; b  = raw[1:H2:2, 1:W2:2]
    elif pat == "bggr":
        b  = raw[0:H2:2, 0:W2:2]; g1 = raw[0:H2:2, 1:W2:2]; g2 = raw[1:H2:2, 0:W2:2]; r  = raw[1:H2:2, 1:W2:2]
    elif pat == "grbg":
        g1 = raw[0:H2:2, 0:W2:2]; r  = raw[0:H2:2, 1:W2:2]; b  = raw[1:H2:2, 0:W2:2]; g2 = raw[1:H2:2, 1:W2:2]
    elif pat == "gbrg":
        g1 = raw[0:H2:2, 0:W2:2]; b  = raw[0:H2:2, 1:W2:2]; r  = raw[1:H2:2, 0:W2:2]; g2 = raw[1:H2:2, 1:W2:2]
    else: # fallback assume rggb
        r  = raw[0:H2:2, 0:W2:2]; g1 = raw[0:H2:2, 1:W2:2]; g2 = raw[1:H2:2, 0:W2:2]; b  = raw[1:H2:2, 1:W2:2]
    g = ((g1.astype(np.float32) + g2.astype(np.float32)) * 0.5)
    out = np.stack([r.astype(np.float32), g, b.astype(np.float32)], axis=2)
    return out / (65535.0 if raw.dtype == np.uint16 else 255.0)

# --------------- Source Base ----------------
class Source:
    def frames(self) -> Generator[Frame, None, None]: raise NotImplementedError

# ----- ASI (Moved from main app) -----
BAYER_CODE_MAP_BY_STR = _OCV_BILINEAR
BAYER_CODE_MAP_BY_ENUM = {
    0: cv2.COLOR_BayerRG2BGR, 1: cv2.COLOR_BayerBG2BGR,
    2: cv2.COLOR_BayerGR2BGR, 3: cv2.COLOR_BayerGB2BGR,
}

class ASISource(Source):
    # --- NOTE: This is a modified ASISource ---
    # It does NOT apply any debayering itself,
    # but yields the RAW frame and a PREVIEW (debayered) frame.
    def __init__(self,
                 asi_dll_path: str = "",
                 gain: int = 200,
                 exposure_ms: int = 200,
                 binning: int = 1,
                 raw16: bool = False,
                 bayer: Optional[str] = "rggb",
                 usb_bandwidth: int = 80,
                 timeout_ms: Optional[int] = None):
        try:
            import zwoasi as asi
        except Exception:
            raise RuntimeError("zwoasi not installed")
        self.asi = asi
        if asi_dll_path and os.path.exists(asi_dll_path): asi.init(asi_dll_path)
        else: asi.init()
        if asi.get_num_cameras()==0: raise RuntimeError("No ASI cameras detected")
        self.cam = asi.Camera(0)
        for f in ("stop_video_capture","stop_exposure"):
            try: getattr(self.cam,f)()
            except Exception: pass
        info = self.cam.get_camera_property()
        max_w, max_h = int(info['MaxWidth']), int(info['MaxHeight'])
        bayer_enum = int(info.get('BayerPattern', 0))
        self.raw16 = bool(raw16)
        # Calibration MUST use RAW16
        img_type = asi.ASI_IMG_RAW16 if self.raw16 else asi.ASI_IMG_RAW8 
        
        w = max_w; h = max_h # Use max resolution
        if binning not in (1,2): binning = 1
        self.cam.set_roi_format(width=w, height=h, bins=binning, image_type=img_type)
        try:
            fmt = self.cam.get_roi_format()
            w = int(fmt.get('Width', w)); h = int(fmt.get('Height', h))
        except Exception:
            pass
        self.w, self.h = w, h
        self.cam.set_control_value(asi.ASI_EXPOSURE, int(exposure_ms*1000), auto=False)
        self.cam.set_control_value(asi.ASI_GAIN, int(gain), auto=False)
        self.cam.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, int(usb_bandwidth))
        
        if bayer is None:
            self.cvt_bilinear = BAYER_CODE_MAP_BY_ENUM.get(bayer_enum, cv2.COLOR_BayerRG2BGR)
        else:
            self.cvt_bilinear = BAYER_CODE_MAP_BY_STR.get(bayer.lower(), cv2.COLOR_BayerRG2BGR)
            
        self.cam.start_video_capture()
        self.timeout_ms = int(timeout_ms if timeout_ms else max(1000, 2*exposure_ms + 500))

    def _bytes_to_ndarray(self, frame):
        dtype = np.uint16 if self.raw16 else np.uint8
        arr = np.frombuffer(frame, dtype=dtype)
        expected = self.w * self.h
        if arr.size != expected:
             # Try to recover
            if self.w and arr.size % self.w == 0: self.h = arr.size // self.w
            elif self.h and arr.size % self.h == 0: self.w = arr.size // self.h
            else: raise RuntimeError(f"ASI buffer mismatch: {arr.size} vs {expected}")
        return arr.reshape((self.h, self.w))

    def _debayer_for_preview(self, raw: np.ndarray) -> np.ndarray:
        # Simple bilinear debayer for preview only
        bgr = cv2.cvtColor(raw, self.cvt_bilinear)
        img_f = bgr.astype(np.float32) / (65535.0 if self.raw16 else 255.0)
        return img_f

    def frames(self):
        asi = self.asi
        try:
            while True:
                try:
                    frame = self.cam.get_video_data(timeout=self.timeout_ms)
                except asi.ZWO_Error:
                    continue
                raw = self._bytes_to_ndarray(frame)
                img_f_preview = self._debayer_for_preview(raw)
                # Yield both raw and preview
                yield Frame(image_f32=img_f_preview, image_raw=raw) 
        finally:
            try: self.cam.stop_video_capture()
            except Exception: pass
            self.cam.close()
            
    def close(self):
        try: self.cam.stop_video_capture()
        except Exception: pass
        self.cam.close()

# --------------- ImageView (Moved from main app) ----------------
class ImageView(QLabel):
    def __init__(self, title=""):
        super().__init__(title)
        self.setMinimumSize(240, 180)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self._pix = None; self._fit_to_window = True
        self._zoom = 1.0; self._min_zoom = 0.05; self._max_zoom = 40.0
        self._pan = np.array([0.0, 0.0], dtype=float)
        self._dragging = False; self._last_mouse = None
        self._auto_contrast = False
        self._visual_gain = 1.0
        self._last_img_f32: Optional[np.ndarray] = None

    def set_auto_contrast(self, enabled: bool):
        self._auto_contrast = enabled

    def set_visual_gain(self, gain: float):
        self._visual_gain = gain
        if self._last_img_f32 is not None and not self._auto_contrast:
            qimg = np_to_qimage(self._last_img_f32, auto_contrast=self._auto_contrast, visual_gain=self._visual_gain)
            self._pix = QPixmap.fromImage(qimg)
            self.update()

    def set_image(self, img_f32: np.ndarray):
        self._last_img_f32 = img_f32
        qimg = np_to_qimage(img_f32, auto_contrast=self._auto_contrast, visual_gain=self._visual_gain)
        new_pix = QPixmap.fromImage(qimg)
        if self._pix is None:
            self._pix = new_pix; self._fit_to_window = True
            self._zoom = 1.0; self._pan[:] = 0
            self.update(); return
        old_w, old_h = self._pix.width(), self._pix.height()
        new_w, new_h = new_pix.width(), new_pix.height()
        self._pix = new_pix
        if not self._fit_to_window and (old_w, old_h) != (new_w, new_h):
            self._pan[:] = 0
            if old_w > 0 and new_w > 0: self._zoom *= (old_w / new_w)
        self.update()

    def _fit_scale(self) -> float:
        if not self._pix or self.width() <= 0 or self.height() <= 0: return 1.0
        pw = self._pix.width(); ph = self._pix.height()
        if pw == 0 or ph == 0: return 1.0
        return min(self.width() / pw, self.height() / ph)

    def _image_rect(self):
        if not self._pix: return None
        if self._fit_to_window:
            s = self._fit_scale(); tw = int(self._pix.width() * s); th = int(self._pix.height() * s)
            x = (self.width() - tw) // 2; y = (self.height() - th) // 2
            return x, y, tw, th
        tw = int(self._pix.width() * self._zoom); th = int(self._pix.height() * self._zoom)
        cx = self.width() // 2; cy = self.height() // 2
        x = int(cx - tw // 2 + self._pan[0]); y = int(cy - th // 2 + self._pan[1])
        return x, y, tw, th

    def paintEvent(self, e):
        super().paintEvent(e);
        if not self._pix: return
        painter = QPainter(self); x, y, w, h = self._image_rect()
        painter.drawPixmap(x, y, w, h, self._pix)

    def wheelEvent(self, ev):
        if not self._pix: return
        angle = ev.angleDelta().y();
        if angle == 0: return
        base = 1.15
        if ev.modifiers() & Qt.ControlModifier: base = 1.3
        elif ev.modifiers() & Qt.ShiftModifier: base = 1.07
        factor = base if angle > 0 else 1.0 / base
        if self._fit_to_window:
            self._fit_to_window = False; self._zoom = self._fit_scale(); self._pan[:] = 0
        mx = ev.position().x(); my = ev.position().y()
        x, y, w, h = self._image_rect(); rx = (mx - x) / max(1, w); ry = (my - y) / max(1, h)
        new_zoom = float(np.clip(self._zoom * factor, self._min_zoom, self._max_zoom))
        if new_zoom == self._zoom: return
        self._zoom = new_zoom; nx, ny, nw, nh = self._image_rect()
        desired_x = mx - rx * nw; desired_y = my - ry * nh
        self._pan += np.array([desired_x - nx, desired_y - ny]); self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and not self._fit_to_window:
            self._dragging = True; self._last_mouse = np.array([ev.position().x(), ev.position().y()])
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._dragging and self._last_mouse is not None:
            pos = np.array([ev.position().x(), ev.position().y()]); delta = pos - self._last_mouse
            self._pan += delta; self._last_mouse = pos; self.update()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._dragging:
            self._dragging = False; self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        self._fit_to_window = not self._fit_to_window
        if self._fit_to_window: self._pan[:] = 0; self._zoom = 1.0
        else: self._zoom = self._fit_scale()
        self.update(); super().mouseDoubleClickEvent(ev)

# --------------- Calibration Worker ----------------
class CalibWorker(QObject):
    sig_progress = Signal(int, str) # percent, status_text
    sig_frame = Signal(object)      # preview frame
    sig_finished = Signal(str, object) # "bias" or "dark", master_frame
    sig_error = Signal(str)
    
    def __init__(self, asi_dll_path, calib_path, gain, binning, exposure_ms, num_frames, mode, parent=None):
        super().__init__(parent)
        self.asi_dll_path = asi_dll_path
        self.calib_path = calib_path
        self.gain = gain
        self.binning = binning
        self.exposure_ms = exposure_ms
        self.num_frames = num_frames
        self.mode = mode # "bias" or "dark"
        self._running = False
        self.source = None

    @Slot()
    def run(self):
        self._running = True
        try:
            self.sig_progress.emit(0, f"Initializing camera...")
            self.source = ASISource(
                asi_dll_path=self.asi_dll_path,
                gain=self.gain,
                exposure_ms=self.exposure_ms,
                binning=self.binning,
                raw16=True, # Calibration frames MUST be 16-bit
                bayer=None, # Use camera default
                usb_bandwidth=40, # Low bandwidth is fine for calibration
                timeout_ms=self.exposure_ms * 2 + 2000
            )
            
            master_frame_f64 = None
            
            for i, frame in enumerate(self.source.frames()):
                if not self._running:
                    self.sig_progress.emit(0, "Cancelled")
                    return
                
                if i >= self.num_frames:
                    break
                    
                self.sig_frame.emit(frame.image_f32) # Send preview
                
                raw = frame.image_raw.astype(np.float64)
                
                if master_frame_f64 is None:
                    master_frame_f64 = raw
                else:
                    # Simple averaging
                    master_frame_f64 = (master_frame_f64 * i + raw) / (i + 1)
                    
                pct = int(100 * (i + 1) / self.num_frames)
                self.sig_progress.emit(pct, f"Captured frame {i+1} / {self.num_frames}")

            if master_frame_f64 is None:
                raise RuntimeError("No frames were captured.")

            master_frame_u16 = master_frame_f64.astype(np.uint16)
            
            # --- Save the frame ---
            # Filename format: master_bias_g<gain>_b<bin>.fits
            filename = f"master_{self.mode}_g{self.gain}_b{self.binning}.fits"
            filepath = os.path.join(self.calib_path, filename)
            
            if fits is None:
                raise RuntimeError("astropy is required to save FITS files.")
                
            self.sig_progress.emit(100, f"Saving to {filename}...")
            
            hdu = fits.PrimaryHDU(master_frame_u16)
            hdu.header['CAL_TYPE'] = (self.mode, 'Calibration frame type')
            hdu.header['GAIN'] = (self.gain, 'Camera gain')
            hdu.header['BINNING'] = (self.binning, 'Camera binning')
            hdu.header['EXPOSURE'] = (self.exposure_ms / 1000.0, 'Exposure time in seconds')
            hdu.header['NFRAMES'] = (self.num_frames, 'Number of averaged frames')
            
            hdul = fits.HDUList([hdu])
            hdul.writeto(filepath, overwrite=True)
            hdul.close()

            self.sig_progress.emit(100, f"Finished: {filename}")
            self.sig_finished.emit(self.mode, master_frame_u16)

        except Exception as e:
            self.sig_error.emit(str(e))
        finally:
            if self.source:
                self.source.close()
                
    def stop(self):
        self._running = False

# --------------- Calibration Dialog ----------------
class CalibrationUtility(QDialog):
    def __init__(self, asi_dll_path, calib_path, initial_gain, initial_bin, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dark Frame Calibration Utility")
        self.asi_dll_path = asi_dll_path
        self.calib_path = calib_path
        self.initial_gain = initial_gain
        self.initial_bin = initial_bin
        
        self.worker_thread = None
        self.worker = None
        
        self._build_ui()
        self._bind_actions()
        self.setMinimumSize(800, 600)

    def _build_ui(self):
        root = QHBoxLayout(self)
        
        # --- Left Controls ---
        controls = QVBoxLayout()
        g_params = QGroupBox("Capture Parameters")
        grid = QGridLayout(g_params)
        
        row = 0
        grid.addWidget(QLabel("Gain"), row, 0)
        self.ed_gain = QSpinBox(); self.ed_gain.setRange(0, 600); self.ed_gain.setValue(self.initial_gain)
        grid.addWidget(self.ed_gain, row, 1); row+=1
        
        grid.addWidget(QLabel("Binning"), row, 0)
        self.cb_bin = QComboBox(); self.cb_bin.addItems(["1", "2"])
        self.cb_bin.setCurrentText(str(self.initial_bin))
        grid.addWidget(self.cb_bin, row, 1); row+=1

        grid.addWidget(QLabel("Num. Frames"), row, 0)
        self.ed_num_frames = QSpinBox(); self.ed_num_frames.setRange(1, 500); self.ed_num_frames.setValue(20)
        grid.addWidget(self.ed_num_frames, row, 1); row+=1

        grid.addWidget(QLabel("Dark Exp. (ms)"), row, 0)
        self.ed_dark_exp = QSpinBox(); self.ed_dark_exp.setRange(100, 120000); self.ed_dark_exp.setValue(30000)
        grid.addWidget(self.ed_dark_exp, row, 1); row+=1
        
        controls.addWidget(g_params)
        
        g_actions = QGroupBox("Actions")
        vbox_actions = QVBoxLayout(g_actions)
        self.btn_capture_bias = QPushButton("Capture Master Bias")
        self.btn_capture_bias.setToolTip("Captures frames at the shortest possible exposure (0ms).")
        self.btn_capture_dark = QPushButton("Capture Master Dark")
        self.btn_capture_dark.setToolTip("Captures frames at the 'Dark Exp (ms)' setting.")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        vbox_actions.addWidget(self.btn_capture_bias)
        vbox_actions.addWidget(self.btn_capture_dark)
        vbox_actions.addWidget(self.btn_cancel)
        controls.addWidget(g_actions)

        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0,100); self.progress_bar.setValue(0)
        controls.addWidget(self.progress_bar)
        
        self.lab_status = QLabel("Status: Idle. Cover telescope/camera lens.")
        self.lab_status.setWordWrap(True)
        controls.addWidget(self.lab_status)
        controls.addStretch(1)
        
        w_controls = QWidget(); w_controls.setLayout(controls)
        w_controls.setMaximumWidth(300)
        
        # --- Right Image View ---
        self.image_view = ImageView("Calibration Preview")
        self.image_view.set_auto_contrast(True) # Always autocontrast preview
        
        root.addWidget(w_controls)
        root.addWidget(self.image_view, 1) # Give image view stretch factor

    def _bind_actions(self):
        self.btn_capture_bias.clicked.connect(self.on_capture_bias)
        self.btn_capture_dark.clicked.connect(self.on_capture_dark)
        self.btn_cancel.clicked.connect(self.on_cancel)
        
    def _set_controls_enabled(self, enabled):
        self.g_params.setEnabled(enabled)
        self.btn_capture_bias.setEnabled(enabled)
        self.btn_capture_dark.setEnabled(enabled)
        self.btn_cancel.setEnabled(not enabled)

    def on_capture_bias(self):
        self._start_capture(mode="bias")

    def on_capture_dark(self):
        self._start_capture(mode="dark")

    def on_cancel(self):
        if self.worker:
            self.lab_status.setText("Status: Cancelling...")
            self.worker.stop()
        self._set_controls_enabled(True)

    def _start_capture(self, mode):
        if self.worker_thread is not None:
            QMessageBox.warning(self, "Busy", "A capture is already in progress.")
            return

        gain = self.ed_gain.value()
        binning = int(self.cb_bin.currentText())
        num_frames = self.ed_num_frames.value()
        
        if mode == "bias":
            # ASI driver interprets 0 as shortest possible exposure
            exposure_ms = 0 
        else: # "dark"
            exposure_ms = self.ed_dark_exp.value()
            
        self._set_controls_enabled(False)
        self.lab_status.setText("Status: Starting...")
        self.progress_bar.setValue(0)
        
        self.worker = CalibWorker(
            asi_dll_path=self.asi_dll_path,
            calib_path=self.calib_path,
            gain=gain,
            binning=binning,
            exposure_ms=exposure_ms,
            num_frames=num_frames,
            mode=mode
        )
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        
        self.worker.sig_progress.connect(self.on_progress)
        self.worker.sig_frame.connect(self.on_frame)
        self.worker.sig_finished.connect(self.on_finished)
        self.worker.sig_error.connect(self.on_error)
        
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.finished.connect(self.on_thread_finished)
        
        self.worker_thread.start()

    @Slot(int, str)
    def on_progress(self, percent, status):
        self.progress_bar.setValue(percent)
        self.lab_status.setText(f"Status: {status}")

    @Slot(object)
    def on_frame(self, preview_img_f32):
        self.image_view.set_image(preview_img_f32)

    @Slot(str, object)
    def on_finished(self, mode, master_frame):
        self.lab_status.setText(f"Status: Finished Master {mode.capitalize()}!")
        self.progress_bar.setValue(100)
        # Show the final master frame (as preview)
        self.image_view.set_image(to_display_float(master_frame))

    @Slot(str)
    def on_error(self, message):
        QMessageBox.critical(self, "Calibration Error", message)
        self.lab_status.setText(f"Error: {message}")
        # Make sure thread cleans up
        if self.worker:
            self.worker.stop()

    def on_thread_finished(self):
        self.worker_thread.deleteLater()
        self.worker.deleteLater()
        self.worker_thread = None
        self.worker = None
        self._set_controls_enabled(True)

    def closeEvent(self, event):
        # Ensure worker is stopped when dialog is closed
        if self.worker_thread is not None:
            self.lab_status.setText("Status: Stopping...")
            self.worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait(2000) # Wait up to 2s
        event.accept()

# --- Main for testing this file directly ---
if __name__ == "__main__":
    # This allows you to run and test the calibration dialog by itself
    app = QApplication(sys.argv)
    
    # --- Mock data for testing ---
    # IMPORTANT: Change this to your actual ASI DLL path for testing
    TEST_ASI_DLL = r"C:\path\to\ASICamera2.dll" 
    TEST_CALIB_PATH = "./CALIB_TEST"
    
    if not os.path.exists(TEST_CALIB_PATH):
        os.makedirs(TEST_CALIB_PATH)
        
    if not os.path.exists(TEST_ASI_DLL):
        QMessageBox.critical(None, "Test Error", f"Please update TEST_ASI_DLL in calibration_util.py to your ASICamera2.dll path.\n\nLooked for:\n{TEST_ASI_DLL}")
        sys.exit(1)

    dialog = CalibrationUtility(
        asi_dll_path=TEST_ASI_DLL,
        calib_path=TEST_CALIB_PATH,
        initial_gain=100,
        initial_bin=1
    )
    dialog.show()
    sys.exit(app.exec())
