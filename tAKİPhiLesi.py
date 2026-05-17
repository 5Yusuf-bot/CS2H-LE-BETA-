
import sys
import cv2
import mss
import numpy as np
import mediapipe as mp
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QSlider, 
                             QCheckBox, QPushButton, QTabWidget, QColorDialog, 
                             QDoubleSpinBox, QGroupBox, QFrame, QGridLayout)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QBrush, QKeySequence, QShortcut

# --- 1. BÖLÜM: YÜKSEK PERFORMANSLI MULTI-THREAD AI MOTORU ---
class AIScannerThread(QThread):
    """
    Ekranı saniyede onlarca kez yakalayıp MediaPipe ile analiz eden bağımsız iş parçacığı.
    Bu sayede ana menü (GUI) ve çizim katmanı asla donmaz veya kasmaz.
    """
    data_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.confidence = 0.40
        self.monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        self.sct = mss.mss()
        
        # Sadece insan tespiti için optimize edilmiş MediaPipe Pose modeli
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.confidence, 
            min_tracking_confidence=0.40
        )

    def run(self):
        while self.is_running:
            # Hızlı ekran görüntüsü alma
            img = np.array(self.sct.grab(self.monitor))
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            h, w, _ = frame_rgb.shape

            self.pose.min_detection_confidence = self.confidence
            results = self.pose.process(frame_rgb)
            
            detected_targets = []
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                x_pts = [int(lm.x * w) for lm in landmarks]
                y_pts = [int(lm.y * h) for lm in landmarks]
                
                # Hedef sınır kutusu (Bounding Box)
                box = [min(x_pts), min(y_pts), max(x_pts), max(y_pts)]
                
                # Dinamik mesafe simülasyonu (Kutu boyutuna göre)
                w_box = box[2] - box[0]
                dist_est = max(1, int((1920 - w_box) / 100))

                detected_targets.append({
                    'id': 77,
                    'box': box,
                    'landmarks': list(zip(x_pts, y_pts)),
                    'distance': dist_est,
                    'hp': 100
                })

            # Tespit edilen hedefleri ana arayüze ve çizim katmanına gönder
            self.data_signal.emit(detected_targets)
            time.sleep(0.002) # CPU optimizasyonu

    def stop(self):
        self.is_running = False
        self.wait()


# --- 2. BÖLÜM: GELİŞMİŞ KOORDİNAT YUMUŞATMA VE TAHMİN MOTORU ---
class CoordinateSmoother:
    """
    En ufak hareketlerde veya ani kayıplarda yeşil kutunun kaybolmasını, 
    titremesini (flicker) engellemek için kullanılan filtreleme sistemi.
    """
    def __init__(self):
        self.last_box = None
        self.velocity = [0, 0, 0, 0]
        self.smooth_factor = 0.25 # Düşük değer = Daha yumuşak/yavaş, Yüksek = Daha keskin/hızlı

    def filter(self, new_box, smoothing_active, alpha=0.25):
        self.smooth_factor = alpha
        if new_box is None:
            if self.last_box is not None:
                # Hedef milisaniyelik kaybolduysa yön ve hızına göre tahmin çiz (Prediction)
                predicted = [
                    int(self.last_box[0] + self.velocity[0]),
                    int(self.last_box[1] + self.velocity[1]),
                    int(self.last_box[2] + self.velocity[2]),
                    int(self.last_box[3] + self.velocity[3])
                ]
                self.last_box = predicted
                return predicted
            return None

        if self.last_box is None or not smoothing_active:
            self.last_box = list(new_box)
            return new_box

        # Üstel Hareketli Ortalama (EMA) Filtresi ile pürüzsüzleştirme
        smoothed = []
        for i in range(4):
            val = int(self.last_box[i] * (1 - self.smooth_factor) + new_box[i] * self.smooth_factor)
            smoothed.append(val)

        # Hız vektörünü güncelle
        self.velocity = [smoothed[i] - self.last_box[i] for i in range(4)]
        self.last_box = smoothed
        return smoothed


# --- 3. BÖLÜM: ULTRA-AKICI ŞEFFAF ÇİZİM OVERLAY KATMANI ---
class ESPOverlay(QWidget):
    def __init__(self):
        super().__init__()
        # Şeffaf, tıklama geçiren ve her şeyin en üstünde duran katman ayarları
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | 
                            Qt.WindowType.WindowStaysOnTopHint | 
                            Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setGeometry(0, 0, 1920, 1080)

        self.targets = []
        self.config = {}
        self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS

    def update_data(self, targets, config):
        self.targets = targets
        self.config = config
        self.update() # paintEvent'i tetikler

    def paintEvent(self, event):
        if not self.config:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. FOV Dairesi Çizimi
        if self.config.get('show_fov', False):
            cx, cy = self.width() // 2, self.height() // 2
            fov_r = self.config.get('fov_radius', 120)
            painter.setPen(QPen(QColor(255, 51, 102, 150), 1, Qt.PenStyle.DashLine))
            painter.drawEllipse(cx - fov_r, cy - fov_r, fov_r * 2, fov_r * 2)

        # 2. Ekran Ortası Crosshair
        if self.config.get('show_cross', False):
            cx, cy = self.width() // 2, self.height() // 2
            painter.setPen(QPen(QColor(255, 51, 102), 2))
            painter.drawLine(cx - 10, cy, cx + 10, cy)
            painter.drawLine(cx, cy - 10, cx, cy + 10)

        if not self.targets:
            return

        font = QFont("Consolas", 9, QFont.Weight.Bold)
        painter.setFont(font)

        for target in self.targets:
            box = target['box']
            landmarks = target['landmarks']
            w_box = box[2] - box[0]
            h_box = box[3] - box[1]

            # Dinamik Ölçekleme: Hedef uzaklaştıkça çizimlerin incelmesini sağlar
            scale = max(0.3, min(1.0, w_box / 150))
            
            # Renk ve Kalınlık Tanımları
            esp_color = self.config.get('color', QColor(0, 255, 102))
            thick = max(1, int(self.config.get('thick', 2) * scale))
            painter.setPen(QPen(esp_color, thick))

            # 3. Kutu ESP (Klasik)
            if self.config.get('show_box', True) and not self.config.get('show_corners', False):
                margin = int(12 * scale)
                painter.drawRect(box[0] - margin, box[1] - margin, w_box + margin*2, h_box + margin*2)

            # 4. Köşeli Kutu ESP
            elif self.config.get('show_corners', False):
                m = int(12 * scale)
                l = int(min(w_box, h_box) / 4)
                # Sol Üst
                painter.drawLine(box[0]-m, box[1]-m, box[0]-m + l, box[1]-m)
                painter.drawLine(box[0]-m, box[1]-m, box[0]-m, box[1]-m + l)
                # Sağ Üst
                painter.drawLine(box[2]+m, box[1]-m, box[2]+m - l, box[1]-m)
                painter.drawLine(box[2]+m, box[1]-m, box[2]+m, box[1]-m + l)
                # Sol Alt
                painter.drawLine(box[0]-m, box[3]+m, box[0]-m + l, box[3]+m)
                painter.drawLine(box[0]-m, box[3]+m, box[0]-m, box[3]+m - l)
                # Sağ Alt
                painter.drawLine(box[2]+m, box[3]+m, box[2]+m - l, box[3]+m)
                painter.drawLine(box[2]+m, box[3]+m, box[2]+m, box[3]+m - l)

            # 5. Snaplines (Ekran Altından Hedefe Çizgi)
            if self.config.get('show_snaplines', False):
                cx = self.width() // 2
                cy = self.height()
                painter.setPen(QPen(QColor(esp_color.red(), esp_color.green(), esp_color.blue(), 100), 1))
                painter.drawLine(cx, cy, box[0] + w_box // 2, box[3])
                painter.setPen(QPen(esp_color, thick)) # Rengi geri yükle

            # 6. İskelet Sistemi (AI Bones)
            if self.config.get('show_skeleton', True) and landmarks:
                for conn in self.pose_connections:
                    if conn[0] < len(landmarks) and conn[1] < len(landmarks):
                        pt1, pt2 = landmarks[conn[0]], landmarks[conn[1]]
                        painter.drawLine(pt1[0], pt1[1], pt2[0], pt2[1])

                # Eklem Noktaları
                if self.config.get('show_joints', True):
                    painter.setBrush(self.config.get('joint_color', QColor(255, 0, 0)))
                    painter.setPen(Qt.PenStyle.NoPen)
                    j_rad = max(1, int(self.config.get('joint_radius', 4) * scale))
                    for pt in landmarks:
                        painter.drawEllipse(pt[0] - j_rad, pt[1] - j_rad, j_rad*2, j_rad*2)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(esp_color, thick))

            # 7. Kafa Dairesi
            if self.config.get('show_head', True) and landmarks:
                head_pt = landmarks[0]
                h_rad = int(self.config.get('head_radius', 15) * scale)
                painter.drawEllipse(head_pt[0] - h_rad, head_pt[1] - h_rad, h_rad*2, h_rad*2)

            # 8. Temsili Can Barı
            if self.config.get('show_health', False):
                bar_x = box[0] - int(25 * scale)
                bar_y = box[1] - int(12 * scale)
                bar_h = h_box + int(24 * scale)
                painter.fillRect(bar_x, bar_y, 4, bar_h, QColor(255, 51, 51)) # Boş Can (Kırmızı)
                painter.fillRect(bar_x, bar_y, 4, int(bar_h * 0.85), QColor(0, 255, 102)) # %85 Can (Yeşil)

            # 9. Oyuncu Bilgi HUD Kartı
            painter.setPen(QPen(esp_color))
            y_offset = box[1] - int(25 * scale)
            
            info_list = []
            if self.config.get('show_name', True):
                info_list.append(f"NAME: [TGT_{target['id']}]")
            if self.config.get('show_dist', True):
                info_list.append(f"DIST: {target['distance']}m")
                
            info_txt = " | ".join(info_list)
            status_txt = f"STATE: [{target['status']}]"
            
            painter.drawText(box[0] - int(10 * scale), y_offset - 12, info_txt)
            painter.drawText(box[0] - int(10 * scale), y_offset, status_txt)


# --- 4. BÖLÜM: GERÇEKÇİ RADAR WIDGET PANELİ ---
class RadarWidget(QWidget):
    """
    Sacracia menüsünde yer alan, hedeflerin yönünü ve uzaklığını 
    merkeze göre canlı simüle eden özel radar bileşeni.
    """
    def __init__(self):
        super().__init__()
        self.setMinimumSize(160, 160)
        self.target_dist = None

    def update_radar(self, distance):
        self.target_dist = distance
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        cx = self.width() // 2
        cy = self.height() // 2
        radius = min(cx, cy) - 10

        # Radar Gövdesi (Karanlık Halkalar)
        painter.setBrush(QBrush(QColor(12, 13, 20)))
        painter.setPen(QPen(QColor(28, 29, 38), 1))
        painter.drawEllipse(cx - radius, cy - radius, radius*2, radius*2)
        painter.drawEllipse(cx - radius//2, cy - radius//2, radius, radius)
        
        # Grid Çizgileri
        painter.drawLine(cx - radius, cy, cx + radius, cy)
        painter.drawLine(cx, cy - radius, cx, cy + radius)

        # Kilitli hedef varsa radarda yeşil nokta olarak işaretle
        if self.target_dist is not None:
            painter.setBrush(QBrush(QColor(0, 255, 102)))
            painter.setPen(Qt.PenStyle.NoPen)
            
            # Mesafeyi radara göre ölçekle
            scale_dist = min(radius - 6, int(self.target_dist * 4))
            pt_x = cx + int(scale_dist * 0.6)
            pt_y = cy - int(scale_dist * 0.5)
            
            painter.drawEllipse(pt_x - 4, pt_y - 4, 8, 8)


# --- 5. BÖLÜM: PREMİUM SACRACIA MODERN KONTROL PANELİ (GUI) ---
class SacraciaPremiumController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sacracia Sensory & AI Controller v5.5")
        self.setGeometry(100, 100, 520, 800)
        self.setStyleSheet(self.get_sacracia_stylesheet())

        # ESP Değişkenleri
        self.esp_color = QColor(0, 255, 102) # Parlak Neon Yeşil
        self.joint_color = QColor(255, 51, 102) # Neon Pembe
        self.is_running = False

        # Stabilite Filtresi ve Kararlılık Sayacı
        self.smoother = CoordinateSmoother()
        self.lost_frames_counter = 0
        self.max_lost_frames = 20

        # Overlay ve AI İş Parçacığı Kurulumu
        self.overlay = ESPOverlay()
        self.ai_thread = AICoreThread()
        self.ai_thread.data_signal.connect(self.on_data_received)

        # Arayüzü Tasarla
        self.init_ui()

        # Klavye Kısayolu (Panik Tuşu - F12 ile anında hileyi kapatma/gizleme)
        self.panic_shortcut = QShortcut(QKeySequence("F12"), self)
        self.panic_shortcut.activated.connect(self.panic_action)

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Sacracia Logo Header
        header_frame = QFrame()
        header_frame.setObjectName("HeaderFrame")
        header_layout = QHBoxLayout()
        logo_lbl = QLabel("SACRACIA PRO SENSORY")
        logo_lbl.setStyleSheet("font-size: 20px; font-weight: bold; color: #00ff66; letter-spacing: 3px;")
        ver_lbl = QLabel("v5.5 premium")
        ver_lbl.setStyleSheet("font-size: 9px; color: #ff3366; font-family: 'Consolas';")
        header_layout.addWidget(logo_lbl)
        header_layout.addStretch()
        header_layout.addWidget(ver_lbl)
        header_frame.setLayout(header_layout)
        main_layout.addWidget(header_frame)

        tabs = QTabWidget()

        # TAB 1: VISUALS (ESP & Renkler)
        tab_visuals = QWidget()
        v_layout = QVBoxLayout()
        
        grp_visuals = QGroupBox("Görsel ESP Taktikleri")
        grp_v_layout = QVBoxLayout()
        self.chk_box = QCheckBox("Dinamik Kutu ESP (Box)"); self.chk_box.setChecked(True)
        self.chk_corners = QCheckBox("Köşe Hatları ESP (Corners)")
        self.chk_head = QCheckBox("Kafa Dairesi ESP"); self.chk_head.setChecked(True)
        self.chk_snap = QCheckBox("Snaplines (Kılavuz Çizgileri)")
        self.chk_cross = QCheckBox("Ekran Ortası Crosshair"); self.chk_cross.setChecked(True)
        
        self.chk_box.stateChanged.connect(lambda s: self.chk_corners.setChecked(False) if s == 2 else None)
        self.chk_corners.stateChanged.connect(lambda s: self.chk_box.setChecked(False) if s == 2 else None)

        grp_v_layout.addWidget(self.chk_box)
        grp_v_layout.addWidget(self.chk_corners)
        grp_v_layout.addWidget(self.chk_head)
        grp_v_layout.addWidget(self.chk_snap)
        grp_v_layout.addWidget(self.chk_cross)
        grp_visuals.setLayout(grp_v_layout)
        v_layout.addWidget(grp_visuals)

        # Ayar Sürgüler (Sliders)
        v_layout.addWidget(QLabel("ESP Çizgi Kalınlığı:"))
        self.sld_thick = QSlider(Qt.Orientation.Horizontal)
        self.sld_thick.setRange(1, 6); self.sld_thick.setValue(2)
        v_layout.addWidget(self.sld_thick)

        v_layout.addWidget(QLabel("Kafa Bölgesi Çapı (Head Radius):"))
        self.sld_head_r = QSlider(Qt.Orientation.Horizontal)
        self.sld_head_r.setRange(5, 30); self.sld_head_r.setValue(15)
        v_layout.addWidget(self.sld_head_r)

        # Renk Butonları
        color_layout = QHBoxLayout()
        self.btn_color = QPushButton("ESP Rengini Seç")
        self.btn_color.clicked.connect(self.pick_esp_color)
        self.btn_jcolor = QPushButton("Eklem Rengini Seç")
        self.btn_jcolor.clicked.connect(self.pick_joint_color)
        color_layout.addWidget(self.btn_color)
        color_layout.addWidget(self.btn_jcolor)
        v_layout.addLayout(color_layout)

        tab_visuals.setLayout(v_layout)
        tabs.addTab(tab_visuals, "Visuals")

        # TAB 2: AI CORE & STABILITY (Yapay Zeka ve Titreme Önleme)
        tab_ai = QWidget()
        ai_layout = QVBoxLayout()

        grp_ai = QGroupBox("Yapay Zeka & Anti-Flicker Motoru")
        grp_ai_layout = QVBoxLayout()
        self.chk_skeleton = QCheckBox("Kemik İskelet ESP (AI Skeleton)"); self.chk_skeleton.setChecked(True)
        self.chk_joints = QCheckBox("Eklem Noktalarını Çiz"); self.chk_joints.setChecked(True)
        self.chk_smooth = QCheckBox("Koordinat Yumuşatma (Anti-Flicker)"); self.chk_smooth.setChecked(True)
        
        grp_ai_layout.addWidget(self.chk_skeleton)
        grp_ai_layout.addWidget(self.chk_joints)
        grp_ai_layout.addWidget(self.chk_smooth)
        grp_ai.setLayout(grp_ai_layout)
        ai_layout.addWidget(grp_ai)

        ai_layout.addWidget(QLabel("Yumuşatma Hassasiyeti (Alpha):"))
        self.spn_smooth = QDoubleSpinBox()
        self.spn_smooth.setRange(0.05, 1.00); self.spn_smooth.setSingleStep(0.05); self.spn_smooth.setValue(0.25)
        ai_layout.addWidget(self.spn_smooth)

        ai_layout.addWidget(QLabel("Kayıp Koruma Süresi (Maksimum Kare):"))
        self.sld_persistence = QSlider(Qt.Orientation.Horizontal)
        self.sld_persistence.setRange(5, 50); self.sld_persistence.setValue(20)
        ai_layout.addWidget(self.sld_persistence)

        ai_layout.addWidget(QLabel("Yapay Zeka Güven Eşiği (Confidence):"))
        self.spn_conf = QDoubleSpinBox()
        self.spn_conf.setRange(0.1, 0.9); self.spn_conf.setSingleStep(0.05); self.spn_conf.setValue(0.40)
        ai_layout.addWidget(self.spn_conf)

        tab_ai.setLayout(ai_layout)
        tabs.addTab(tab_ai, "AI Core")

        # TAB 3: TELEMETRY (Aimbot FOV, Can Barı, Radar)
        tab_tel = QWidget()
        tel_layout = QVBoxLayout()

        # Radar Bölümü
        grp_radar = QGroupBox("Canlı Radar Analizörü")
        rad_h_layout = QHBoxLayout()
        self.radar_widget = RadarWidget()
        rad_h_layout.addStretch()
        rad_h_layout.addWidget(self.radar_widget)
        rad_h_layout.addStretch()
        grp_radar.setLayout(rad_h_layout)
        tel_layout.addWidget(grp_radar)

        # Diğer Seçenekler
        grp_extras = QGroupBox("Ekstra HUD Ayarları")
        extras_layout = QVBoxLayout()
        self.chk_fov = QCheckBox("Aimbot FOV Dairesini Göster"); self.chk_fov.setChecked(True)
        self.chk_health = QCheckBox("Temsili Oyuncu Can Barı"); self.chk_health.setChecked(True)
        self.chk_name = QCheckBox("Oyuncu İsmini Göster"); self.chk_name.setChecked(True)
        self.chk_dist = QCheckBox("Mesafe Bilgisi Yaz"); self.chk_dist.setChecked(True)
        
        extras_layout.addWidget(self.chk_fov)
        extras_layout.addWidget(self.chk_health)
        extras_layout.addWidget(self.chk_name)
        extras_layout.addWidget(self.chk_dist)
        grp_extras.setLayout(extras_layout)
        tel_layout.addWidget(grp_extras)

        # FOV Sürgüsü
        tel_layout.addWidget(QLabel("Aimbot Görüş Açısı (FOV Radius):"))
        self.sld_fov = QSlider(Qt.Orientation.Horizontal)
        self.sld_fov.setRange(30, 300); self.sld_fov.setValue(120)
        tel_layout.addWidget(self.sld_fov)

        # Durum Panel Gridleri
        grp_status = QGroupBox("Sistem Verileri")
        status_grid = QGridLayout()
        self.lbl_status = QLabel("DURUM: BEKLEMEDE")
        self.lbl_latency = QLabel("GECİKME: 0ms")
        self.lbl_target = QLabel("HEDEF: YOK")
        status_grid.addWidget(self.lbl_status, 0, 0)
        status_grid.addWidget(self.lbl_latency, 0, 1)
        status_grid.addWidget(self.lbl_target, 1, 0, 1, 2)
        grp_status.setLayout(status_grid)
        tel_layout.addWidget(grp_status)

        tab_tel.setLayout(tel_layout)
        tabs.addTab(tab_tel, "Telemetry")

        main_layout.addWidget(tabs)

        # Sacracia Güçlü Başlat Butonu
        self.btn_toggle = QPushButton("SİSTEMİ BAŞLAT (OVERLAY DIRECT)")
        self.btn_toggle.setObjectName("btn_toggle")
        self.btn_toggle.clicked.connect(self.toggle_system)
        main_layout.addWidget(self.btn_toggle)

        # Panik Tuşu Bilgilendirmesi
        panic_lbl = QLabel("Hızlı Gizleme/Panik Modu için F12 tuşunu kullanın.")
        panic_lbl.setStyleSheet("font-size: 10px; color: #8f909c; text-align: center; font-family: 'Consolas';")
        main_layout.addWidget(panic_lbl)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def pick_esp_color(self):
        color = QColorDialog.getColor()
        if color.isValid(): self.esp_color = color

    def pick_joint_color(self):
        color = QColorDialog.getColor()
        if color.isValid(): self.joint_color = color

    def panic_action(self):
        """ F12 Panik Tuşu tetiklendiğinde hileyi anında kapatır ve gizler """
        if self.is_running:
            self.toggle_system()
            self.lbl_status.setText("DURUM: PANIK AKTIF (KAPATILDI)")

    def toggle_system(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_toggle.setText("SİSTEMİ DURDUR")
            self.btn_toggle.setStyleSheet("background-color: #ff3366; color: white;")
            self.lbl_status.setText("DURUM: AKTİF (OVERLAY AÇIK)")
            self.overlay.show()
            
            # AI İş Parçacığını Başlat
            self.ai_thread.confidence = self.spn_conf.value()
            self.ai_thread.is_running = True
            self.ai_thread.start()
        else:
            self.btn_toggle.setText("SİSTEMİ BAŞLAT (OVERLAY DIRECT)")
            self.btn_toggle.setStyleSheet("background-color: #00ff66; color: black;")
            self.lbl_status.setText("DURUM: BEKLEMEDE")
            self.lbl_target.setText("HEDEF: YOK")
            self.overlay.hide()
            self.ai_thread.stop()
            self.smoother.last_box = None
            self.lost_frames_counter = 0
            self.radar_widget.update_radar(None)

    def on_data_received(self, targets):
        """ Arka plandaki AI iş parçacığından gelen verileri alır ve süzgeçten geçirir """
        start_time = time.time()
        self.max_lost_frames = self.sld_persistence.value()

        detected_box = None
        detected_landmarks = None
        status = "SEARCHING"

        if targets:
            # Hedef tespit edildi
            detected_box = targets[0]['box']
            detected_landmarks = targets[0]['landmarks']
            self.lost_frames_counter = 0
            status = "LOCKED"
            self.lbl_target.setText("HEDEF: KİLİTLENDİ (AI ONARLANAN)")
        else:
            # Yapay zeka hedefi o karede kaybettiyse Kalman/Tahmin filtresini devreye al
            if self.lost_frames_counter < self.max_lost_frames:
                self.lost_frames_counter += 1
                status = f"PREDICTING ({self.lost_frames_counter})"
                self.lbl_target.setText(f"HEDEF: YOL TAHMİNİ YAPILIYOR ({self.lost_frames_counter})")
            else:
                self.smoother.last_box = None
                self.lbl_target.setText("HEDEF: YOK (ARIYOR...)")

        # Filtreleme İşlemi (Titreme ve Kapanma önleyici)
        final_box = self.smoother.filter(
            detected_box if status == "LOCKED" else None, 
            self.chk_smooth.isChecked(),
            self.spn_smooth.value()
        )

        processed_targets = []
        if final_box and (detected_landmarks or self.smoother.last_box):
            # Eğer landmarks kaybolduysa boş listeye çek, sadece tahmini kutu kalsın
            if not detected_landmarks:
                detected_landmarks = []

            # Dinamik mesafe tahmini
            w_box = final_box[2] - final_box[0]
            dist_est = max(1, int((1920 - w_box) / 105))

            processed_targets.append({
                'id': 77,
                'box': final_box,
                'landmarks': detected_landmarks,
                'distance': dist_est,
                'status': status,
                'hp': 100
            })
            self.radar_widget.update_radar(dist_est)
        else:
            self.radar_widget.update_radar(None)

        # Tüm Arayüz Konfigürasyonunu Paketle
        config = {
            'color': self.esp_color,
            'joint_color': self.joint_color,
            'thick': self.sld_thick.value(),
            'show_box': self.chk_box.isChecked(),
            'show_corners': self.chk_corners.isChecked(),
            'show_snaplines': self.chk_snap.isChecked(),
            'show_skeleton': self.chk_skeleton.isChecked(),
            'show_joints': self.chk_joints.isChecked(),
            'joint_radius': 4,
            'show_head': self.chk_head.isChecked(),
            'head_radius': self.sld_head_r.value(),
            'show_fov': self.chk_fov.isChecked(),
            'fov_radius': self.sld_fov.value(),
            'show_cross': self.chk_cross.isChecked(),
            'show_health': self.chk_health.isChecked(),
            'show_name': self.chk_name.isChecked(),
            'show_dist': self.chk_dist.isChecked()
        }

        # Overlay Çizimini Güncelle
        self.overlay.update_data(processed_targets, config)

        # Gecikme Hızı Analizi
        latency = int((time.time() - start_time) * 1000)
        self.lbl_latency.setText(f"GECİKME: {latency}ms")

    def closeEvent(self, event):
        self.overlay.close()
        self.ai_thread.stop()
        event.accept()

    # --- SACRACIA LUXURY DARK THEME STYLESHEET ---
    def get_sacracia_stylesheet(self):
        return """
            QMainWindow {
                background-color: #0c0d12;
            }
            QFrame#HeaderFrame {
                background-color: #12131a;
                border-bottom: 2px solid #1c1d24;
                padding: 10px;
                margin-bottom: 10px;
            }
            QLabel {
                color: #e1e1e8;
                font-family: 'Consolas', sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #1c1d24;
                background-color: #12131a;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #12131a;
                border: 1px solid #1c1d24;
                color: #8f909c;
                padding: 12px 20px;
                font-family: 'Consolas';
                font-weight: bold;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #00ff66;
                color: #0c0d12;
                border: 1px solid #00ff66;
            }
            QGroupBox {
                border: 1px solid #1c1d24;
                border-radius: 6px;
                margin-top: 15px;
                font-family: 'Consolas';
                font-weight: bold;
                color: #00ff66;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QCheckBox {
                color: #e1e1e8;
                font-family: 'Consolas';
                spacing: 8px;
                padding: 4px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #1c1d24;
                background: #0c0d12;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #00ff66;
                border: 1px solid #00ff66;
            }
            QPushButton {
                background-color: #1c1d24;
                color: #00ff66;
                font-weight: bold;
                font-family: 'Consolas';
                border: 1px solid #00ff66;
                border-radius: 4px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #00ff66;
                color: #0c0d12;
            }
            QPushButton#btn_toggle {
                background-color: #00ff66;
                color: #0c0d12;
                font-weight: bold;
                font-family: 'Consolas';
                font-size: 14px;
                border: none;
                border-radius: 4px;
                padding: 15px;
                margin-top: 10px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #1c1d24;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00ff66;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QDoubleSpinBox {
                background-color: #0c0d12;
                color: #00ff66;
                border: 1px solid #1c1d24;
                font-family: 'Consolas';
                padding: 6px;
                border-radius: 4px;
            }
        """


# --- 6. BÖLÜM: YÜKSEK PERFORMANSLI EKRAN YAKALAMA SINIFI ---
class AICoreThread(QThread):
    """
    Ekran görüntüsü taramasını PyQt GUI döngüsünden tamamen ayıran 
    asenkron Thread sınıfı. Donmaları kökten çözer.
    """
    data_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.confidence = 0.40
        self.monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        self.sct = mss.mss()
        
        # Sadece insan tespiti için optimize edilmiş MediaPipe Pose modeli
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.confidence, 
            min_tracking_confidence=0.40
        )

    def run(self):
        while self.is_running:
            # Saniyede 60+ kare hızında ultra akıcı ekran görüntüsü yakalama
            img = np.array(self.sct.grab(self.monitor))
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            h, w, _ = frame_rgb.shape

            self.pose.min_detection_confidence = self.confidence
            results = self.pose.process(frame_rgb)
            
            detected_targets = []
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                x_pts = [int(lm.x * w) for lm in landmarks]
                y_pts = [int(lm.y * h) for lm in landmarks]
                
                # Sınır kutusunu çıkar
                box = [min(x_pts), min(y_pts), max(x_pts), max(y_pts)]
                w_box = box[2] - box[0]
                dist_est = max(1, int((1920 - w_box) / 100))

                detected_targets.append({
                    'id': 77,
                    'box': box,
                    'landmarks': list(zip(x_pts, y_pts)),
                    'distance': dist_est,
                    'hp': 100
                })

            self.data_signal.emit(detected_targets)
            time.sleep(0.005) # İşlemciyi koruma beklemesi

    def stop(self):
        self.is_running = False
        self.wait()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = SacraciaPremiumController()
    controller.show()
    sys.exit(app.exec())

