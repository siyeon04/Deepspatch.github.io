// 결과 데이터 가져오기 (sessionStorage에서)
const resultData = JSON.parse(sessionStorage.getItem('analysisResult') || '{}');

console.log('분석 결과:', resultData);

// 점수 카드 업데이트
function updateScoreCards() {
    const videoScore = resultData.video_deepfake || 0;
    const audioScore = resultData.audio_deepfake || 0;
    const overallScore = resultData.overall_score || 0;
    
    document.getElementById('videoScore').textContent = videoScore.toFixed(1) + '%';
    document.getElementById('audioScore').textContent = audioScore.toFixed(1) + '%';
    document.getElementById('overallScore').textContent = overallScore.toFixed(1) + '%';
    
    // 색상 적용
    updateScoreCardColor('videoScore', videoScore);
    updateScoreCardColor('audioScore', audioScore);
    updateScoreCardColor('overallScore', overallScore);
}

function updateScoreCardColor(elementId, score) {
    const element = document.getElementById(elementId);
    if (score < 30) {
        element.style.color = '#10b981'; // 녹색
    } else if (score < 70) {
        element.style.color = '#f59e0b'; // 주황색
    } else {
        element.style.color = '#ef4444'; // 빨간색
    }
}

// 타임라인 그래프
let timelineChart = null;

function createTimelineChart() {
    const timeline = resultData.timeline || [];
    const suspiciousTime = resultData.most_suspicious_time || 0;
    
    // 가장 의심스러운 시점 표시
    document.getElementById('suspiciousTime').textContent = 
        `가장 의심스러운 시점: ${suspiciousTime}초 (${timeline.find(t => t.timestamp === suspiciousTime)?.score.toFixed(1) || '--'}%)`;
    
    if (timeline.length === 0) {
        console.warn('타임라인 데이터가 없습니다');
        return;
    }
    
    const ctx = document.getElementById('timelineChart').getContext('2d');
    
    // 데이터 준비
    const labels = timeline.map(t => t.timestamp + '초');
    const scores = timeline.map(t => t.score);
    
    // 색상 배열 (가장 의심스러운 시점은 빨간색)
    const colors = timeline.map(t => 
        t.timestamp === suspiciousTime ? '#ef4444' : '#60a5fa'
    );
    
    timelineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: '딥페이크 의심도 (%)',
                data: scores,
                borderColor: '#60a5fa',
                backgroundColor: 'rgba(96, 165, 250, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: colors,
                pointBorderColor: colors,
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#cbd5e1',
                        font: {
                            size: 14
                        }
                    }
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#fff',
                    bodyColor: '#cbd5e1',
                    borderColor: '#475569',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return `딥페이크 의심도: ${context.parsed.y.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        color: '#334155'
                    }
                },
                x: {
                    ticks: {
                        color: '#94a3b8',
                        maxRotation: 45,
                        minRotation: 45
                    },
                    grid: {
                        color: '#334155'
                    }
                }
            }
        }
    });
}

// 얼굴 부위별 분석 업데이트
function updateFacialAnalysis() {
    const facialRegions = resultData.facial_regions;
    
    if (!facialRegions) {
        document.getElementById('facialSection').style.display = 'none';
        return;
    }
    
    // 눈
    const eyeScore = facialRegions['눈'] || 0;
    document.getElementById('eyeScore').textContent = eyeScore.toFixed(1) + '%';
    document.getElementById('eyeDetail').textContent = eyeScore.toFixed(1) + '%';
    document.getElementById('eyeBar').style.width = eyeScore + '%';
    
    // 코
    const noseScore = facialRegions['코'] || 0;
    document.getElementById('noseScore').textContent = noseScore.toFixed(1) + '%';
    document.getElementById('noseDetail').textContent = noseScore.toFixed(1) + '%';
    document.getElementById('noseBar').style.width = noseScore + '%';
    
    // 입
    const mouthScore = facialRegions['입'] || 0;
    document.getElementById('mouthScore').textContent = mouthScore.toFixed(1) + '%';
    document.getElementById('mouthDetail').textContent = mouthScore.toFixed(1) + '%';
    document.getElementById('mouthBar').style.width = mouthScore + '%';
    
    // 얼굴 윤곽
    const contourScore = facialRegions['얼굴윤곽'] || 0;
    document.getElementById('contourScore').textContent = contourScore.toFixed(1) + '%';
    document.getElementById('contourDetail').textContent = contourScore.toFixed(1) + '%';
    document.getElementById('contourBar').style.width = contourScore + '%';
}

// 페이지 로드 시 실행
window.addEventListener('DOMContentLoaded', () => {
    // 결과 데이터가 없으면 업로드 페이지로 리다이렉트
    if (!resultData || Object.keys(resultData).length === 0) {
        alert('분석 결과가 없습니다. 먼저 비디오를 분석해주세요.');
        window.location.href = 'upload.html';
        return;
    }
    
    // 데이터 표시
    updateScoreCards();
    createTimelineChart();
    updateFacialAnalysis();
});

// 페이지 떠날 때 결과 데이터 유지
window.addEventListener('beforeunload', () => {
    // sessionStorage는 탭이 닫히면 자동 삭제됨
    // localStorage로 변경하면 영구 저장
});