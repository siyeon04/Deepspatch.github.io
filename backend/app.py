from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
import subprocess
import librosa
from PIL import Image
import base64
from io import BytesIO

# TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow 없음 - 오디오 분석 불가")

# FastAPI 앱
app = FastAPI(title="딥페이크 탐지 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VIDEO_MODEL_PATH = "models/GANomaly_xai_model.pth"
AUDIO_MODEL_PATH = "models/ganomaly_model_full_dataset.h5"

# 비디오 설정 (Document 4)
VIDEO_IMAGE_SIZE = 64
VIDEO_LATENT_DIM = 128

# 오디오 설정 (Document 5)
AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MELS = 128
AUDIO_HOP_LENGTH = 512
AUDIO_N_FFT = 2048
AUDIO_LENGTH = 48640
AUDIO_WIDTH = 96
AUDIO_HEIGHT = 128
AUDIO_CHANNELS = 3  # Mel, Delta, Delta-Delta

# ==================== Config 클래스 ====================
class Config:
    """Document 4의 Config 클래스 (checkpoint 역직렬화용)"""
    TRAIN_DATA_PATH = '/content/dataset/train'
    TEST_DATA_PATH = '/content/dataset/test'
    IMAGE_SIZE = 64
    BATCH_SIZE = 32
    LATENT_DIM = 128
    EPOCHS = 150
    LR_G = 0.0002
    LR_D = 0.0001
    BETA1 = 0.5
    BETA2 = 0.999
    W_ADV = 1
    W_CON = 100
    W_ENC = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== PyTorch GANomaly (비디오) ====================

class Generator(nn.Module):
    """Document 4의 정확한 Generator 구조"""
    def __init__(self):
        super(Generator, self).__init__()
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, VIDEO_LATENT_DIM, 4, 1, 0, bias=False),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(VIDEO_LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, VIDEO_LATENT_DIM, 4, 1, 0, bias=False),
        )
    
    def forward(self, x):
        z = self.encoder1(x)
        x_hat = self.decoder(z)
        z_hat = self.encoder2(x_hat)
        return x_hat, z, z_hat


# ==================== 모델 로드 ====================

video_model = None
video_threshold = 50.0
video_score_5th = 0.0
video_score_95th = 1.0

audio_model = None
audio_g_encoder = None

def load_video_model():
    """비디오 모델 로드"""
    global video_model, video_threshold, video_score_5th, video_score_95th
    
    try:
        print("📦 비디오 모델 로드 중...")
        
        checkpoint = torch.load(VIDEO_MODEL_PATH, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'generator' in checkpoint:
                state_dict = checkpoint['generator']
            else:
                state_dict = checkpoint
            
            video_model = Generator()
            video_model.load_state_dict(state_dict, strict=True)
            video_model.to(device)
            video_model.eval()
            
            if 'threshold_normalized' in checkpoint:
                video_threshold = float(checkpoint['threshold_normalized'])
            
            if 'score_5th' in checkpoint and 'score_95th' in checkpoint:
                video_score_5th = float(checkpoint['score_5th'])
                video_score_95th = float(checkpoint['score_95th'])
            
            print("✅ 비디오 모델 로드 완료")
            print(f"   임계값: {video_threshold:.1f}%")
            print(f"   정규화 범위: [{video_score_5th:.4f}, {video_score_95th:.4f}]")
            
    except Exception as e:
        print(f"❌ 비디오 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()


def load_audio_model():
    """오디오 모델 로드 - 저장된 구조 확인 후 로드"""
    global audio_model, audio_g_encoder
    
    if not TF_AVAILABLE:
        return
    
    try:
        print("📦 오디오 모델 로드 중...")
        
        # Step 1: 저장된 h5 파일 구조 확인
        import h5py
        print("   📋 저장된 모델 구조 분석 중...")
        
        try:
            with h5py.File(AUDIO_MODEL_PATH, 'r') as f:
                print(f"   h5 파일 키: {list(f.keys())}")
                
                if 'model_weights' in f.keys():
                    model_weights = f['model_weights']
                    saved_layers = list(model_weights.keys())
                    print(f"   저장된 레이어({len(saved_layers)}개): {saved_layers}")
        except Exception as e:
            print(f"   ⚠️  h5 구조 분석 실패: {e}")
        
        # Step 2: Document 5 구조 재현
        from tensorflow.keras import regularizers
        
        height, width, channels = AUDIO_HEIGHT, AUDIO_WIDTH, AUDIO_CHANNELS
        
        # g_e (encoder)
        input_layer_g_e = layers.Input(name='input_g_e', shape=(height, width, channels))
        x = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='conv_1', 
                         kernel_regularizer=regularizers.l2(0.0001))(input_layer_g_e)
        x = layers.LeakyReLU(name='leaky_1')(x)
        x = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='conv_2', 
                         kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization(name='norm_1')(x)
        x = layers.LeakyReLU(name='leaky_2')(x)
        x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_3', 
                         kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization(name='norm_2')(x)
        x = layers.LeakyReLU(name='leaky_3')(x)
        x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_4', 
                         kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization(name='norm_3')(x)
        x = layers.LeakyReLU(name='leaky_4')(x)
        x = layers.GlobalAveragePooling2D(name='g_encoder_output')(x)
        g_e = models.Model(inputs=input_layer_g_e, outputs=x, name="g_encoder")
        
        # g (generator = g_e + decoder)
        last_conv_shape = g_e.get_layer('leaky_4').output.shape
        dense_units = int(last_conv_shape[1] * last_conv_shape[2] * last_conv_shape[3])
        reshape_shape = (int(last_conv_shape[1]), int(last_conv_shape[2]), int(last_conv_shape[3]))
        
        input_layer_g_d = layers.Input(name='input_g_d', shape=(height, width, channels))
        x_g_d = g_e(input_layer_g_d)
        y = layers.Dense(dense_units, name='dense')(x_g_d)
        y = layers.Reshape(reshape_shape, name='de_reshape')(y)
        y = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', name='deconv_1', 
                                  kernel_regularizer=regularizers.l2(0.0001))(y)
        y = layers.LeakyReLU(name='de_leaky_1')(y)
        y = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', name='deconv_2', 
                                  kernel_regularizer=regularizers.l2(0.0001))(y)
        y = layers.LeakyReLU(name='de_leaky_2')(y)
        y = layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', name='deconv_3', 
                                  kernel_regularizer=regularizers.l2(0.0001))(y)
        y = layers.LeakyReLU(name='de_leaky_3')(y)
        y = layers.Conv2DTranspose(channels, (1, 1), strides=(1,1), padding='same', 
                                  name='decoder_deconv_output', 
                                  kernel_regularizer=regularizers.l2(0.0001), 
                                  activation='tanh')(y)
        g = models.Model(inputs=input_layer_g_d, outputs=y, name="generator")
        
        print(f"   재구성된 모델 레이어 수: {len(g.layers)}")
        
        # Step 3: 가중치 로드 (여러 방법 시도)
        loaded = False
        
        # 방법 1: by_name=True
        if not loaded:
            try:
                print(f"   [1/3] by_name=True로 로드 시도...")
                g.load_weights(AUDIO_MODEL_PATH, by_name=True, skip_mismatch=True)
                print(f"   ✅ by_name 로드 성공")
                loaded = True
            except Exception as e:
                print(f"   ✗ by_name 로드 실패: {e}")
        
        # 방법 2: 전체 로드
        if not loaded:
            try:
                print(f"   [2/3] 전체 로드 시도...")
                g.load_weights(AUDIO_MODEL_PATH)
                print(f"   ✅ 전체 로드 성공")
                loaded = True
            except Exception as e:
                print(f"   ✗ 전체 로드 실패: {e}")
        
        # 방법 3: h5py로 수동 로드
        if not loaded:
            try:
                print(f"   [3/3] 수동 로드 시도...")
                with h5py.File(AUDIO_MODEL_PATH, 'r') as f:
                    if 'model_weights' in f.keys():
                        model_weights = f['model_weights']
                        
                        for layer_name in model_weights.keys():
                            try:
                                layer = g.get_layer(layer_name)
                                layer_weights_group = model_weights[layer_name][layer_name]
                                
                                weights = []
                                for weight_name in layer_weights_group.keys():
                                    weights.append(np.array(layer_weights_group[weight_name]))
                                
                                if weights:
                                    layer.set_weights(weights)
                            except:
                                pass
                        
                        print(f"   ✅ 수동 로드 성공")
                        loaded = True
            except Exception as e:
                print(f"   ✗ 수동 로드 실패: {e}")
        
        if not loaded:
            raise Exception("모든 로드 방법 실패")
        
        audio_model = g
        audio_g_encoder = g_e
        
        print("✅ 오디오 모델 로드 완료")
        print(f"   입력: {height}x{width}x{channels}")
        
    except Exception as e:
        print(f"❌ 오디오 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()


# 모델 로드 실행
print("📦 모델 로드 중...")
load_video_model()
load_audio_model()

# ==================== 전처리 ====================

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((VIDEO_IMAGE_SIZE, VIDEO_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==================== 유틸리티 함수 ====================

def download_video_from_url(url):
    try:
        tmp_dir = tempfile.gettempdir()
        output_template = os.path.join(tmp_dir, 'downloaded_video.%(ext)s')
        
        command = [
            'yt-dlp',
            '-f', 'best[ext=mp4]/best',
            '-o', output_template,
            '--no-playlist',
            '--max-filesize', '100M',
            url
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise Exception(f"다운로드 실패: {result.stderr}")
        
        downloaded_files = [
            os.path.join(tmp_dir, f) 
            for f in os.listdir(tmp_dir) 
            if f.startswith('downloaded_video.')
        ]
        
        if downloaded_files:
            return downloaded_files[0]
        
        raise Exception("다운로드된 파일을 찾을 수 없습니다.")
        
    except Exception as e:
        raise Exception(f"다운로드 오류: {str(e)}")


def extract_frames_per_second(video_path):
    """1초당 1프레임씩 추출"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"  📊 영상 정보: {total_frames}프레임, {fps:.1f}fps, {duration:.1f}초")
    
    frames_with_timestamps = []
    current_second = 0
    
    while True:
        frame_number = int(current_second * fps)
        
        if frame_number >= total_frames:
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if frame_rgb.dtype != np.uint8:
                frame_rgb = frame_rgb.astype(np.uint8)
            
            frames_with_timestamps.append((current_second, frame_rgb))
            
            if (current_second + 1) % 10 == 0:
                print(f"      ... {current_second + 1}초 프레임 추출 중")
        
        current_second += 1
    
    cap.release()
    
    print(f"  ✅ {len(frames_with_timestamps)}개 프레임 추출 완료 (1fps)")
    return frames_with_timestamps


def create_heatmap(error_map):
    """Document 4의 히트맵 생성"""
    error_normalized = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
    heatmap = cv2.applyColorMap((error_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def analyze_facial_regions_grid(frame, error_map_resized):
    """
    그리드 기반 얼굴 부위 분석 (MediaPipe 없이)
    Document 4의 FaceRegionDetector.get_region_masks 방식
    """
    h, w = frame.shape[:2]
    
    # Document 4의 그리드 기반 영역
    regions = {
        '왼쪽 눈': (int(h*0.3), int(h*0.5), int(w*0.2), int(w*0.4)),
        '오른쪽 눈': (int(h*0.3), int(h*0.5), int(w*0.6), int(w*0.8)),
        '코': (int(h*0.4), int(h*0.65), int(w*0.4), int(w*0.6)),
        '입': (int(h*0.65), int(h*0.85), int(w*0.3), int(w*0.7)),
        '턱': (int(h*0.75), int(h*0.95), int(w*0.35), int(w*0.65)),
        '이마': (int(h*0.1), int(h*0.35), int(w*0.25), int(w*0.75))
    }
    
    # 부위별 점수 계산
    region_scores = {}
    for region_name, (y1, y2, x1, x2) in regions.items():
        region_error = error_map_resized[y1:y2, x1:x2].mean()
        # Document 4 방식
        score = region_error * 100
        region_scores[region_name] = round(float(score), 1)
    
    return region_scores


def normalize_channel(data):
    """Document 5의 정규화 함수"""
    min_val = np.min(data)
    max_val = np.max(data)
    if (max_val - min_val) > 1e-6:
        return (data - min_val) / (max_val - min_val) * 2 - 1
    else:
        return data - np.mean(data)


def extract_audio_to_spectrogram_3channel(video_path):
    """Document 5의 방식으로 3채널 스펙트로그램 생성"""
    audio_path = None
    
    try:
        audio_path = video_path.replace(Path(video_path).suffix, '_audio.wav')
        
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(AUDIO_SAMPLE_RATE),
            '-ac', '1',
            '-y',
            audio_path
        ]
        
        result = subprocess.run(
            command, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )
        
        try:
            import soundfile as sf
            y, sr= sf.read(audio_path)

            if len(y.shape) > 1:
                y = y.mean(axis=1)

            if sr != AUDIO_SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
                sr = AUDIO_SAMPLE_RATE

        except Exception as e:
            print(f"   ⚠️  soundfile 로드 실패, librosa로 재시도: {e}")
            y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)

        if len(y) <sr * 0.1:
            return None
        
        if len(y) > AUDIO_LENGTH:
            y = y[:AUDIO_LENGTH]
        else:
            y = np.pad(y, (0, max(0, AUDIO_LENGTH - len(y))), "constant")
        
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=AUDIO_N_FFT, 
            hop_length=AUDIO_HOP_LENGTH, 
            n_mels=AUDIO_N_MELS,
            center=False
        )
        
        if mel_spec.shape[1] < AUDIO_WIDTH:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, AUDIO_WIDTH - mel_spec.shape[1])), mode='constant')
        elif mel_spec.shape[1] > AUDIO_WIDTH:
            mel_spec = mel_spec[:, :AUDIO_WIDTH]
        
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        delta_spec = librosa.feature.delta(log_mel_spec)
        delta2_spec = librosa.feature.delta(log_mel_spec, order=2)
        
        norm_log_mel_spec = normalize_channel(log_mel_spec)
        norm_delta_spec = normalize_channel(delta_spec)
        norm_delta2_spec = normalize_channel(delta2_spec)
        
        spec_3d = np.stack([norm_log_mel_spec, norm_delta_spec, norm_delta2_spec], axis=-1)
        
        return spec_3d
        
    except Exception as e:
        print(f"  ⚠️  오디오 스펙트로그램 생성 실패: {e}")
        return None
        
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

# ==================== 탐지 함수 ====================

def calculate_anomaly_score_video(gen_img, latent_i, latent_o, input_tensor):
    """Document 4의 Anomaly Score 계산"""
    error_recon = torch.mean(torch.abs(input_tensor - gen_img), dim=[1, 2, 3])
    error_latent = torch.mean((latent_i - latent_o) ** 2, dim=[1, 2, 3])
    
    anomaly_score_raw = (error_recon + 0.1 * error_latent).item()
    
    # Percentile 정규화
    anomaly_score_normalized = (anomaly_score_raw - video_score_5th) / (video_score_95th - video_score_5th + 1e-8) * 100
    anomaly_score_normalized = np.clip(anomaly_score_normalized, 0, 100)
    
    return anomaly_score_normalized


def detect_deepfake_video_with_timeline(frames_with_timestamps):
    """비디오 딥페이크 탐지"""
    if video_model is None:
        return 50.0, [], 0, None, None
    
    try:
        timeline_data = []
        all_scores = []
        max_score = 0
        max_timestamp = 0
        max_frame = None
        max_tensor = None
        max_gen_tensor = None
        
        print(f"  📊 {len(frames_with_timestamps)}개 프레임 분석 중...")
        
        with torch.no_grad():
            for i, (timestamp, frame) in enumerate(frames_with_timestamps):
                frame_tensor = image_transform(frame).unsqueeze(0).to(device)
                
                try:
                    gen_img, latent_i, latent_o = video_model(frame_tensor)
                    fake_prob = calculate_anomaly_score_video(
                        gen_img, latent_i, latent_o, frame_tensor
                    )
                    
                    timeline_data.append({
                        'timestamp': int(timestamp),
                        'score': round(float(fake_prob), 2)
                    })
                    
                    all_scores.append(fake_prob)
                    
                    if fake_prob > max_score:
                        max_score = fake_prob
                        max_timestamp = timestamp
                        max_frame = frame
                        max_tensor = frame_tensor
                        max_gen_tensor = gen_img
                    
                    if (i + 1) % 10 == 0:
                        print(f"      {i+1}초: 딥페이크={fake_prob:.1f}%")
                    
                except Exception as e:
                    timeline_data.append({
                        'timestamp': int(timestamp),
                        'score': 50.0
                    })
                    all_scores.append(50.0)
        
        avg_score = np.mean(all_scores) if all_scores else 50.0
        
        # XAI: 그리드 기반 얼굴 분석 + 히트맵
        facial_regions = None
        heatmap_base64 = None
        
        if max_frame is not None and max_gen_tensor is not None:
            try:
                error_map = torch.abs(max_tensor - max_gen_tensor).mean(dim=1).squeeze().cpu().numpy()
                error_map_resized = cv2.resize(error_map, (max_frame.shape[1], max_frame.shape[0]))
                
                heatmap = create_heatmap(error_map_resized)
                facial_regions = analyze_facial_regions_grid(max_frame, error_map_resized)
                
                heatmap_pil = Image.fromarray(heatmap)
                buffered = BytesIO()
                heatmap_pil.save(buffered, format="PNG")
                heatmap_base64 = base64.b64encode(buffered.getvalue()).decode()
            except Exception as e:
                print(f"  ⚠️  XAI 분석 실패: {e}")
        
        print(f"\n  ⚠️  가장 의심스러운 시점: {max_timestamp}초 ({max_score:.1f}%)")
        
        return float(avg_score), timeline_data, int(max_timestamp), facial_regions, heatmap_base64
        
    except Exception as e:
        print(f"  ❌ 비디오 분석 오류: {e}")
        import traceback
        traceback.print_exc()
        return 50.0, [], 0, None, None


def detect_deepvoice_audio(spectrogram_3d):
    """Document 5의 방식으로 딥보이스 탐지"""
    if audio_model is None or audio_g_encoder is None or spectrogram_3d is None:
        return 0.0
    
    try:
        spec_batch = np.expand_dims(spectrogram_3d, axis=0)
        
        if spec_batch.shape != (1, AUDIO_HEIGHT, AUDIO_WIDTH, AUDIO_CHANNELS):
            print(f"  ⚠️  스펙트로그램 크기 불일치: {spec_batch.shape}")
            return 0.0
        
        # Document 5의 방식
        encoded_original = audio_g_encoder.predict(spec_batch, verbose=0)
        reconstructed_spec = audio_model.predict(spec_batch, verbose=0)
        encoded_reconstructed = audio_g_encoder.predict(reconstructed_spec, verbose=0)
        
        score_raw = np.sum(np.absolute(encoded_original - encoded_reconstructed), axis=-1)[0]
        
        # 0-100% 정규화
        score_normalized = np.clip(score_raw * 10, 0, 100)
        
        return float(score_normalized)
        
    except Exception as e:
        print(f"  ❌ 오디오 분석 오류: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

# ==================== API 엔드포인트 ====================

@app.post("/api/analyze-url")
async def analyze_video_url(video_url: str = Form(...)):
    video_path = None
    
    try:
        video_path = download_video_from_url(video_url)
        
        print("\n🎬 [1/3] 프레임 추출 중 (1fps)...")
        frames_with_timestamps = extract_frames_per_second(video_path)
        
        print("\n🔍 [2/3] 비디오 딥페이크 분석 중...")
        video_score, timeline, max_time, facial_regions, heatmap = detect_deepfake_video_with_timeline(frames_with_timestamps)
        
        print("\n🎵 [3/3] 오디오 딥보이스 분석 중...")
        spectrogram_3d = extract_audio_to_spectrogram_3channel(video_path)
        audio_score = detect_deepvoice_audio(spectrogram_3d)
        
        overall_score = (video_score + audio_score) / 2
        
        print(f"\n✅ 분석 완료!")
        print(f"  🎬 비디오 딥페이크: {video_score:.2f}%")
        print(f"  🎵 오디오 딥보이스: {audio_score:.2f}%")
        print(f"  📊 종합 점수: {overall_score:.2f}%")
        
        return JSONResponse(content={
            "success": True,
            "video_deepfake": float(video_score),
            "audio_deepfake": float(audio_score),
            "overall_score": float(overall_score),
            "timeline": timeline,
            "most_suspicious_time": int(max_time),
            "facial_regions": facial_regions,
            "heatmap": heatmap,
            "threshold": float(video_threshold),
            "frames_analyzed": len(frames_with_timestamps),
            "audio_available": audio_model is not None and spectrogram_3d is not None
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {str(e)}")
    
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass


@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    file_ext = Path(video.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_video:
        content = await video.read()
        tmp_video.write(content)
        tmp_video_path = tmp_video.name
    
    try:
        print(f"\n📹 파일 분석 시작: {video.filename}")
        
        print("\n🎬 [1/3] 프레임 추출 중 (1fps)...")
        frames_with_timestamps = extract_frames_per_second(tmp_video_path)
        
        print("\n🔍 [2/3] 비디오 딥페이크 분석 중...")
        video_score, timeline, max_time, facial_regions, heatmap = detect_deepfake_video_with_timeline(frames_with_timestamps)
        
        print("\n🎵 [3/3] 오디오 딥보이스 분석 중...")
        spectrogram_3d = extract_audio_to_spectrogram_3channel(tmp_video_path)
        audio_score = detect_deepvoice_audio(spectrogram_3d)
        
        overall_score = (video_score + audio_score) / 2
        
        print(f"\n✅ 분석 완료!")
        print(f"  🎬 비디오 딥페이크: {video_score:.2f}%")
        print(f"  🎵 오디오 딥보이스: {audio_score:.2f}%")
        print(f"  📊 종합 점수: {overall_score:.2f}%")
        
        return JSONResponse(content={
            "success": True,
            "video_deepfake": float(video_score),
            "audio_deepfake": float(audio_score),
            "overall_score": float(overall_score),
            "timeline": timeline,
            "most_suspicious_time": int(max_time),
            "facial_regions": facial_regions,
            "heatmap": heatmap,
            "threshold": float(video_threshold),
            "frames_analyzed": len(frames_with_timestamps),
            "audio_available": audio_model is not None and spectrogram_3d is not None
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {str(e)}")
    
    finally:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)


@app.get("/")
async def root():
    return {
        "message": "🎭 딥페이크 탐지 API (GANomaly)",
        "status": "running",
        "models": {
            "video": {
                "loaded": video_model is not None,
                "image_size": VIDEO_IMAGE_SIZE,
                "threshold": float(video_threshold)
            },
            "audio": {
                "loaded": audio_model is not None and audio_g_encoder is not None,
                "input_shape": f"{AUDIO_HEIGHT}x{AUDIO_WIDTH}x{AUDIO_CHANNELS}"
            }
        }
    }

# ==================== 정적 파일 서빙 ====================

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
    print(f"📁 Frontend 경로: {FRONTEND_DIR}")
else:
    print(f"⚠️  Frontend 폴더를 찾을 수 없습니다: {FRONTEND_DIR}")


# ==================== 서버 시작 ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 딥페이크 탐지 서버 시작")
    print("="*60)
    print(f"\n📊 비디오 모델:")
    print(f"   - 상태: {'✅' if video_model else '❌'}")
    print(f"   - 임계값: {video_threshold:.1f}%")
    
    print(f"\n🎵 오디오 모델:")
    print(f"   - 상태: {'✅' if (audio_model and audio_g_encoder) else '❌'}")
    print(f"   - 입력: {AUDIO_HEIGHT}x{AUDIO_WIDTH}x{AUDIO_CHANNELS}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=5000)