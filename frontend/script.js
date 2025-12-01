// DOM Elements
const uploadToggle = document.getElementById('uploadToggle');
const fileUploadSection = document.getElementById('fileUploadSection');
const urlUploadSection = document.getElementById('urlUploadSection');
const dropZone = document.getElementById('dropZone');
const videoFile = document.getElementById('videoFile');
const videoUrl = document.getElementById('videoUrl');
const uploadText = document.getElementById('uploadText');
const uploadCheck = document.getElementById('uploadCheck');
const analyzeButton = document.getElementById('analyzeButton');
const startAnalysis = document.getElementById('startAnalysis');
const processingSection = document.getElementById('processingSection');
const resultSection = document.getElementById('resultSection');
const newAnalysis = document.getElementById('newAnalysis');

// State
let currentMethod = 'file';
let selectedFile = null;

// Upload Method Toggle
if (uploadToggle) {
    uploadToggle.addEventListener('click', (e) => {
        if (e.target.classList.contains('toggle-btn')) {
            document.querySelectorAll('.toggle-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            e.target.classList.add('active');
            
            currentMethod = e.target.dataset.method;
            
            if (currentMethod === 'file') {
                fileUploadSection.classList.remove('hidden');
                urlUploadSection.classList.add('hidden');
            } else {
                fileUploadSection.classList.add('hidden');
                urlUploadSection.classList.remove('hidden');
            }
            
            analyzeButton.classList.add('hidden');
        }
    });
}

// File Upload - Drag and Drop
if (dropZone) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', handleDrop);
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
}

// File Input Change
if (videoFile) {
    videoFile.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

function handleFileSelect(file) {
    selectedFile = file;
    uploadText.textContent = file.name;
    uploadCheck.style.display = 'flex';
    analyzeButton.classList.remove('hidden');
}

// URL Input Change
if (videoUrl) {
    videoUrl.addEventListener('input', (e) => {
        if (e.target.value.trim()) {
            analyzeButton.classList.remove('hidden');
        } else {
            analyzeButton.classList.add('hidden');
        }
    });
}

// Start Analysis Button
if (startAnalysis) {
    startAnalysis.addEventListener('click', async () => {
        console.log('🚀 분석 시작!');
        
        fileUploadSection.classList.add('hidden');
        urlUploadSection.classList.add('hidden');
        analyzeButton.classList.add('hidden');
        if (uploadToggle) uploadToggle.style.display = 'none';
        
        processingSection.classList.remove('hidden');
        
        try {
            let result;
            
            // 프로세싱 애니메이션 시작
            simulateProcessing();
            
            if (currentMethod === 'file' && selectedFile) {
                console.log('📤 파일 업로드:', selectedFile.name);
                
                // ✅ 실제 백엔드 API 호출
                result = await analyzeVideo(selectedFile);
                console.log('✅ 백엔드 응답:', result);
                
            } else if (currentMethod === 'url' && videoUrl.value.trim()) {
                console.log('📤 URL 업로드:', videoUrl.value.trim());
                
                // ✅ 실제 백엔드 API 호출
                result = await analyzeVideoUrl(videoUrl.value.trim());
                console.log('✅ 백엔드 응답:', result);
            } else {
                throw new Error('파일 또는 URL을 선택해주세요.');
            }
            
            // 결과를 sessionStorage에 저장
            sessionStorage.setItem('analysisResult', JSON.stringify(result));
            
            // 대시보드로 이동
            console.log('📊 대시보드로 이동');
            window.location.href = 'dashboard.html';
            
        } catch (error) {
            console.error('❌ 에러 발생:', error);
            alert('분석 중 오류가 발생했습니다: ' + error.message);
            window.location.reload();
        }
    });
}

// Simulate Processing
async function simulateProcessing() {
    const steps = [0, 1, 2, 3];
    
    for (let i = 0; i < steps.length; i++) {
        const stepLabels = document.querySelectorAll('.step-label');
        const stepStatuses = document.querySelectorAll('.step-status');
        const progressFills = document.querySelectorAll('.progress-fill');
        
        stepLabels[i].classList.add('active');
        stepStatuses[i].classList.add('processing');
        progressFills[i].classList.add('half');
        
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        stepStatuses[i].classList.remove('processing');
        stepStatuses[i].classList.add('complete');
        progressFills[i].classList.remove('half');
        progressFills[i].classList.add('complete');
    }
}

// Display Results (사용 안 함 - 대시보드로 이동)
function displayResults(data) {
    // 대시보드로 이동하므로 불필요
}

// New Analysis Button
if (newAnalysis) {
    newAnalysis.addEventListener('click', () => {
        window.location.reload();
    });
}

// ==================== Backend API Functions ====================

async function analyzeVideo(file) {
    console.log('🔄 analyzeVideo 함수 호출');
    console.log('   파일명:', file.name);
    console.log('   파일 크기:', (file.size / 1024 / 1024).toFixed(2), 'MB');
    
    const formData = new FormData();
    formData.append('video', file);
    
    console.log('📡 백엔드로 전송 중...');
    
    try {
        const response = await fetch('http://localhost:5000/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        console.log('📥 응답 상태:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ 서버 에러:', errorText);
            throw new Error(`분석 실패: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ 분석 완료:', data);
        return data;
        
    } catch (error) {
        console.error('❌ API 호출 실패:', error);
        throw error;
    }
}

async function analyzeVideoUrl(url) {
    console.log('🔄 analyzeVideoUrl 함수 호출');
    console.log('   URL:', url);
    
    try {
        const response = await fetch('http://localhost:5000/api/analyze-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });
        
        console.log('📥 응답 상태:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ 서버 에러:', errorText);
            throw new Error(`분석 실패: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ 분석 완료:', data);
        return data;
        
    } catch (error) {
        console.error('❌ API 호출 실패:', error);
        throw error;
    }
}