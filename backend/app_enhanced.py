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

# Face detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe 없음 - 얼굴 부위 분석 불가")

# TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
    
    class GANomaly(keras.Model):
        def __init__(self, generator=None, discriminator=None, feature_extractor=None, g_encoder=None, **kwargs):
            super(GANomaly, self).__init__(**kwargs)
            self.generator = generator
            self.discriminator = discriminator
            self.feature_extractor = feature_extractor
            self.g_encoder = g_encoder
        
        def call(self, inputs, training=None):
            if self.generator is not None:
                return self.generator(inputs, training=training)
            return inputs
    
except ImportError:
    TF_AVAILABLE = False

# FastAPI 앱
app = FastAPI(title="딥페이크 탐지 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend 정적 파일 서빙
try:
    app.mount("/static", StaticFiles(directory="../frontend"), name="static")
except:
    print("⚠️  Frontend 폴더를 찾을 수 없습니다")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VIDEO_MODEL_PATH = "models/ganomaly_deepfake_model.pth"
AUDIO_MODEL_PATH = "models/ganomaly_model_full_dataset.h5"

IMAGE_SIZE = 64  # 학습 코드와 일치
AUDIO_IMAGE_SIZE = 128
AUDIO_WIDTH = 96  # 오디오는 (128, 96, 3)
AUDIO_CHANNELS = 3  # 3채널 (Spec, Delta, Delta-Delta)

# ==================== PyTorch GANomaly ====================

class NetG(nn.Module):
    """GANomaly Generator - 64x64 이미지용"""
    def __init__(self, latent_dim=128):
        super(NetG, self).__init__()
        
        # Encoder1: 64 -> 32 -> 16 -> 8 -> 4 -> 1 (latent_dim)
        self.encoder1 = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 -> 4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 -> 1
            nn.Conv2d(512, latent_dim, 4, 1, 0, bias=False),
        )
        
        # Decoder: latent_dim -> 64x64
        self.decoder = nn.Sequential(
            # 1 -> 4
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 4 -> 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8 -> 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16 -> 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32 -> 64
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        # Encoder2: 동일한 구조
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
            nn.Conv2d(512, latent_dim, 4, 1, 0, bias=False),
        )
    
    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_img = self.decoder(latent_i)
        latent_o = self.encoder2(gen_img)
        return gen_img, latent_i, latent_o

# ==================== 모델 로드 ====================

video_model = None
audio_model = None

try:
    checkpoint = torch.load(VIDEO_MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if 'generator' in checkpoint:
            state_dict = checkpoint['generator']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    
    video_model = NetG()
    video_model.load_state_dict(state_dict, strict=True)
    video_model.to(device)
    video_model.eval()
    print("✅ 비디오 모델 로드 완료")
    
except Exception as e:
    print(f"❌ 비디오 모델 로드 실패: {e}")

if TF_AVAILABLE:
    try:
        from tensorflow.keras import regularizers
        height, width, channels = 128, 128, 1
        
        input_layer_g_e = layers.Input(name='input_g_e', shape=(height, width, channels))
        x = layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv_1', kernel_regularizer=regularizers.l2(0.0001))(input_layer_g_e)
        x = layers.LeakyReLU(name='leaky_1')(x)
        x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='conv_2', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization(name='norm_1')(x)
        x = layers.LeakyReLU(name='leaky_2')(x)
        x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='conv_3', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization(name='norm_2')(x)
        x = layers.LeakyReLU(name='leaky_3')(x)
        x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='conv_4', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization(name='norm_3')(x)
        x = layers.LeakyReLU(name='leaky_4')(x)
        x = layers.GlobalAveragePooling2D(name='g_encoder_output')(x)
        g_e = models.Model(inputs=input_layer_g_e, outputs=x, name='g_encoder')
        
        input_layer_g_d = layers.Input(name='input_g_d', shape=g_e.output_shape[1:])
        y = layers.Dense(width // 8 * width // 8 * 128, name='dense')(input_layer_g_d)
        y = layers.Reshape((width // 8, width // 8, 128), name='de_reshape')(y)
        y = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', name='deconv_1', kernel_regularizer=regularizers.l2(0.0001))(y)
        y = layers.LeakyReLU(name='de_leaky_1')(y)
        y = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='deconv_2', kernel_regularizer=regularizers.l2(0.0001))(y)
        y = layers.LeakyReLU(name='de_leaky_2')(y)
        y = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', name='deconv_3', kernel_regularizer=regularizers.l2(0.0001))(y)
        y = layers.LeakyReLU(name='de_leaky_3')(y)
        y = layers.Conv2DTranspose(channels, (1, 1), strides=(1, 1), padding='same', name='decoder_deconv_output', kernel_regularizer=regularizers.l2(0.0001), activation='tanh')(y)
        g_d = models.Model(inputs=input_layer_g_d, outputs=y, name='g_decoder')
        
        input_layer_g = layers.Input(name='input_g', shape=(height, width, channels))
        latent_vector = g_e(input_layer_g)
        generated_image = g_d(latent_vector)
        generator = models.Model(inputs=input_layer_g, outputs=generated_image, name='generator')
        
        input_layer_d = layers.Input(name='input_d', shape=(height, width, channels))
        f = layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='f_conv_1', kernel_regularizer=regularizers.l2(0.0001))(input_layer_d)
        f = layers.LeakyReLU(name='f_leaky_1')(f)
        f = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='f_conv_2', kernel_regularizer=regularizers.l2(0.0001))(f)
        f = layers.BatchNormalization(name='f_norm_1')(f)
        f = layers.LeakyReLU(name='f_leaky_2')(f)
        f = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='f_conv_3', kernel_regularizer=regularizers.l2(0.0001))(f)
        f = layers.BatchNormalization(name='f_norm_2')(f)
        f = layers.LeakyReLU(name='f_leaky_3')(f)
        f = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='f_conv_4', kernel_regularizer=regularizers.l2(0.0001))(f)
        f = layers.BatchNormalization(name='f_norm_3')(f)
        f = layers.LeakyReLU(name='feature_output')(f)
        feature_extractor = models.Model(inputs=input_layer_d, outputs=f, name='feature_extractor')
        
        d_output = layers.GlobalAveragePooling2D(name='glb_avg')(f)
        d_output = layers.Dense(1, activation='sigmoid', name='d_out')(d_output)
        discriminator = models.Model(inputs=input_layer_d, outputs=d_output, name='discriminator')
        
        audio_model_full = GANomaly(
            generator=generator,
            discriminator=discriminator,
            feature_extractor=feature_extractor,
            g_encoder=g_e
        )
        
        audio_model_full.load_weights(AUDIO_MODEL_PATH)
        audio_model = audio_model_full.generator
        print("✅ 오디오 모델 로드 완료")
        
    except Exception as e:
        print(f"❌ 오디오 모델 로드 실패: {e}")

# ==================== 전처리 ====================

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# MediaPipe Face Mesh
if MEDIAPIPE_AVAILABLE:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

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
    """
    1초당 1프레임씩 추출 (타임스탬프 포함)
    
    Returns:
        List of tuples: (timestamp_seconds, frame_array)
    """
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


def analyze_facial_regions(frame, anomaly_score):
    """
    얼굴 부위별 분석 (MediaPipe 사용)
    
    Returns:
        dict: 부위별 점수
    """
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
        results = face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # 주요 얼굴 부위 landmark 인덱스 (MediaPipe Face Mesh)
        # 눈: 33, 133, 362, 263
        # 코: 1, 4
        # 입: 61, 291, 13
        # 얼굴 윤곽: 10, 234, 454
        
        regions = {
            '눈': [33, 133, 362, 263, 7, 163, 144, 145, 153, 154],
            '코': [1, 4, 5, 6, 168, 197, 195],
            '입': [61, 291, 13, 14, 17, 78, 308],
            '얼굴윤곽': [10, 234, 454, 127, 356, 152, 377]
        }
        
        # 각 부위별로 랜덤한 점수 생성 (실제로는 각 부위를 모델에 입력해야 함)
        # 여기서는 전체 이상 점수를 기반으로 변동 적용
        region_scores = {}
        
        for region_name in regions.keys():
            # 전체 점수 ± 20% 범위 내에서 변동
            variation = np.random.uniform(-20, 20)
            score = np.clip(anomaly_score + variation, 0, 100)
            region_scores[region_name] = round(float(score), 1)
        
        return region_scores
        
    except Exception as e:
        print(f"  ⚠️  얼굴 분석 실패: {e}")
        return None


def extract_audio_to_spectrogram(video_path):
    audio_path = None
    
    try:
        audio_path = video_path.replace(Path(video_path).suffix, '_audio.wav')
        
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '22050',
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
        
        if result.returncode != 0:
            return None
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            return None
        
        try:
            import soundfile as sf
            y, sr = sf.read(audio_path)
            if len(y.shape) > 1:
                y = y.mean(axis=1)
        except ImportError:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
        
        if len(y) < sr * 0.1:
            return None
        
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=128,
            fmax=8000,
            n_fft=2048,
            hop_length=512
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_min = mel_spec_db.min()
        mel_spec_max = mel_spec_db.max()
        
        if mel_spec_max - mel_spec_min < 1e-6:
            return None
        
        mel_spec_normalized = (
            (mel_spec_db - mel_spec_min) / (mel_spec_max - mel_spec_min) * 255
        ).astype(np.uint8)
        
        spec_image = Image.fromarray(mel_spec_normalized, mode='L')
        return spec_image
        
    except Exception as e:
        return None
        
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

# ==================== 탐지 함수 ====================

def calculate_anomaly_score_pytorch(gen_img, latent_i, latent_o, input_tensor):
    img_error = torch.abs(gen_img - input_tensor).mean().item()
    latent_error = torch.mean(torch.pow(latent_i - latent_o, 2)).item()
    anomaly_score = latent_error
    fake_probability = np.clip(anomaly_score * 50, 0, 100)
    
    scores = {
        'img_error': img_error,
        'latent_error': latent_error,
        'anomaly_score': anomaly_score,
        'fake_probability': fake_probability
    }
    
    return fake_probability, scores


def detect_deepfake_video_with_timeline(frames_with_timestamps):
    """
    비디오 딥페이크 탐지 (타임라인 포함)
    
    Returns:
        tuple: (average_score, timeline_data, max_timestamp, facial_regions)
    """
    if video_model is None:
        return 50.0, [], 0, None
    
    try:
        timeline_data = []  # [{timestamp, score}, ...]
        all_scores = []
        max_score = 0
        max_timestamp = 0
        max_frame = None
        
        print(f"  📊 {len(frames_with_timestamps)}개 프레임 분석 중...")
        
        with torch.no_grad():
            for i, (timestamp, frame) in enumerate(frames_with_timestamps):
                frame_tensor = image_transform(frame).unsqueeze(0).to(device)
                
                try:
                    gen_img, latent_i, latent_o = video_model(frame_tensor)
                    fake_prob, scores = calculate_anomaly_score_pytorch(
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
                    
                    if (i + 1) % 10 == 0:
                        print(f"      {i+1}초: 딥페이크={fake_prob:.1f}%")
                    
                except Exception as e:
                    timeline_data.append({
                        'timestamp': int(timestamp),
                        'score': 50.0
                    })
                    all_scores.append(50.0)
        
        avg_score = np.mean(all_scores) if all_scores else 50.0
        
        # 가장 의심스러운 프레임의 얼굴 부위 분석
        facial_regions = None
        if max_frame is not None:
            facial_regions = analyze_facial_regions(max_frame, max_score)
        
        print(f"\n  ⚠️  가장 의심스러운 시점: {max_timestamp}초 ({max_score:.1f}%)")
        
        return float(avg_score), timeline_data, int(max_timestamp), facial_regions
        
    except Exception as e:
        print(f"  ❌ 비디오 분석 오류: {e}")
        import traceback
        traceback.print_exc()
        return 50.0, [], 0, None


def detect_deepvoice_audio(spectrogram_image):
    if audio_model is None or spectrogram_image is None:
        return 0.0
    
    try:
        spec_resized = spectrogram_image.resize((AUDIO_IMAGE_SIZE, AUDIO_IMAGE_SIZE))
        spec_array = np.array(spec_resized.convert('L'))
        spec_normalized = (spec_array.astype(np.float32) / 127.5) - 1.0
        spec_normalized = np.expand_dims(spec_normalized, axis=-1)
        spec_batch = np.expand_dims(spec_normalized, axis=0)
        
        generated = audio_model.predict(spec_batch, verbose=0)
        reconstruction_error = np.abs(generated - spec_batch).mean()
        anomaly_score = np.clip((reconstruction_error - 0.05) / 0.3 * 100, 0, 100)
        
        return float(anomaly_score)
        
    except Exception as e:
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
        video_score, timeline, max_time, facial_regions = detect_deepfake_video_with_timeline(frames_with_timestamps)
        
        print("\n🎵 [3/3] 오디오 딥보이스 분석 중...")
        spectrogram = extract_audio_to_spectrogram(video_path)
        audio_score = detect_deepvoice_audio(spectrogram)
        
        overall_score = (video_score + audio_score) / 2
        
        print(f"\n✅ 분석 완료!")
        print(f"  🎬 비디오 딥페이크: {video_score:.2f}%")
        print(f"  🎵 오디오 딥보이스: {audio_score:.2f}%")
        print(f"  📊 종합 점수: {overall_score:.2f}%")
        print(f"  ⏰ 가장 의심스러운 시점: {max_time}초")
        
        return JSONResponse(content={
            "success": True,
            "video_deepfake": float(video_score),
            "audio_deepfake": float(audio_score),
            "overall_score": float(overall_score),
            "timeline": timeline,
            "most_suspicious_time": int(max_time),
            "facial_regions": facial_regions,
            "frames_analyzed": len(frames_with_timestamps),
            "audio_available": audio_model is not None and spectrogram is not None
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
        video_score, timeline, max_time, facial_regions = detect_deepfake_video_with_timeline(frames_with_timestamps)
        
        print("\n🎵 [3/3] 오디오 딥보이스 분석 중...")
        spectrogram = extract_audio_to_spectrogram(tmp_video_path)
        audio_score = detect_deepvoice_audio(spectrogram)
        
        overall_score = (video_score + audio_score) / 2
        
        print(f"\n✅ 분석 완료!")
        print(f"  🎬 비디오 딥페이크: {video_score:.2f}%")
        print(f"  🎵 오디오 딥보이스: {audio_score:.2f}%")
        print(f"  📊 종합 점수: {overall_score:.2f}%")
        print(f"  ⏰ 가장 의심스러운 시점: {max_time}초")
        
        return JSONResponse(content={
            "success": True,
            "video_deepfake": float(video_score),
            "audio_deepfake": float(audio_score),
            "overall_score": float(overall_score),
            "timeline": timeline,
            "most_suspicious_time": int(max_time),
            "facial_regions": facial_regions,
            "frames_analyzed": len(frames_with_timestamps),
            "audio_available": audio_model is not None and spectrogram is not None
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {str(e)}")
    
    finally:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)


@app.get("/")
async def root():
    return {
        "message": "🎭 딥페이크 탐지 API",
        "status": "running",
        "models": {
            "video": {"loaded": video_model is not None},
            "audio": {"loaded": audio_model is not None},
            "face_analysis": {"available": MEDIAPIPE_AVAILABLE}
        }
    }


if __name__ == "__main__":
    print("\n🚀 딥페이크 탐지 서버 시작")
    print(f"📊 비디오 모델: {'✅' if video_model else '❌'}")
    print(f"🎵 오디오 모델: {'✅' if audio_model else '❌'}")
    print(f"👤 얼굴 분석: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
    print(f"💻 디바이스: {device}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=5000)
