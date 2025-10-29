import torch
import sys

# Config 클래스 더미 정의 (모델 로드용)
class Config:
    """GANomaly 설정 클래스 더미"""
    def __init__(self):
        self.isize = 256  # 기본값
        self.nc = 3
        self.nz = 100
        pass

# 모델 파일 경로
model_path = r"C:\Users\seann\OneDrive\바탕 화면\PBL\웹페이지\backend\models\ganomaly_deepfake_model.pth"

print("=" * 60)
print("🔍 GANomaly 딥페이크 모델 정보 확인")
print("=" * 60)

try:
    # 방법 1: 일반 로드 시도
    print("\n📦 모델 로드 중 (방법 1)...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ 모델 로드 성공!")
        
        # 체크포인트 타입 확인
        print(f"\n📋 체크포인트 타입: {type(checkpoint)}")
        
        # 딕셔너리인 경우
        if isinstance(checkpoint, dict):
            print(f"\n🔑 체크포인트 키:")
            for key in checkpoint.keys():
                print(f"  - {key}")
                
                # 각 키의 타입 확인
                if key == 'opt':
                    opt = checkpoint['opt']
                    print(f"\n⚙️  설정 정보 (opt):")
                    print(f"    타입: {type(opt)}")
                    
                    # Config 객체인 경우
                    if hasattr(opt, '__dict__'):
                        print(f"    속성들:")
                        for attr, value in opt.__dict__.items():
                            print(f"      - {attr}: {value}")
                            if attr == 'isize':
                                print(f"\n✨ 입력 이미지 크기: {value}x{value}")
                
                # state_dict 확인
                if 'state_dict' in key or 'netg' in key.lower():
                    state_dict = checkpoint[key]
                    if isinstance(state_dict, dict):
                        print(f"\n  {key} 레이어 수: {len(state_dict)}")
                        print(f"  처음 3개 레이어:")
                        for i, (name, param) in enumerate(list(state_dict.items())[:3]):
                            print(f"    {i+1}. {name}: {list(param.shape)}")
        
        print("\n" + "=" * 60)
        print("✅ 모델 정보 확인 완료!")
        print("=" * 60)
        
    except AttributeError as e:
        print(f"⚠️  방법 1 실패: {e}")
        print("\n📦 모델 로드 중 (방법 2 - state_dict만)...")
        
        # 방법 2: state_dict만 추출
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # state_dict 직접 접근
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"✅ state_dict 추출 성공!")
                print(f"레이어 수: {len(state_dict)}")
                
                print(f"\n처음 5개 레이어:")
                for i, (name, param) in enumerate(list(state_dict.items())[:5]):
                    print(f"  {i+1}. {name}: {list(param.shape)}")
                
                # 첫 번째 conv 레이어에서 입력 크기 추정
                first_layer = list(state_dict.items())[0]
                print(f"\n💡 첫 번째 레이어로부터 입력 정보 추정:")
                print(f"   이름: {first_layer[0]}")
                print(f"   shape: {list(first_layer[1].shape)}")
                
                print(f"\n⚠️  정확한 입력 크기는 학습 시 코드를 확인해야 합니다.")
                print(f"💡 일반적인 GANomaly는 64, 128, 256 중 하나를 사용합니다.")
            
except FileNotFoundError:
    print(f"\n❌ 오류: 모델 파일을 찾을 수 없습니다.")
    print(f"경로: {model_path}")
    
except Exception as e:
    print(f"\n❌ 오류 발생: {str(e)}")
    print(f"오류 타입: {type(e).__name__}")
    
    print("\n" + "=" * 60)
    print("🔍 zip 파일로 직접 확인 시도...")
    print("=" * 60)
    
    try:
        import zipfile
        with zipfile.ZipFile(model_path, 'r') as z:
            print("\n📦 체크포인트 파일 구조:")
            for name in z.namelist():
                print(f"  - {name}")
                if name == 'data.pkl':
                    print("\n💡 이 모델은 학습 시 사용한 원본 코드가 필요합니다.")
                    print("학습할 때 사용한 GANomaly 코드 파일을 알려주세요!")
    except:
        pass

print("\n" + "=" * 60)
print("💡 다음 단계:")
print("=" * 60)
print("1. 모델 학습 시 사용한 코드 파일 확인")
print("2. 학습 시 설정한 이미지 크기 확인 (보통 64, 128, 256 중 하나)")
print("3. 또는 일단 256으로 시도해보기 (GANomaly 기본값)")
print("=" * 60)