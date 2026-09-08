/**
 * Phase 4: React Dashboard Frontend
 * Deepfake Detection Analysis UI
 * 
 * Run: npm install && npm start
 */

import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

function VerdictBadge({ verdict, risk }) {
  const getColor = (v) => {
    if (v === 'DEEPFAKE') return '#dc3545';
    if (v === 'INCONCLUSIVE') return '#ffc107';
    return '#28a745';
  };

  const getEmoji = (v) => {
    if (v === 'DEEPFAKE') return '🔴';
    if (v === 'INCONCLUSIVE') return '🟡';
    return '🟢';
  };

  return (
    <div style={{
      padding: '20px',
      borderRadius: '8px',
      backgroundColor: getColor(verdict),
      color: 'white',
      textAlign: 'center',
      marginBottom: '20px'
    }}>
      <div style={{ fontSize: '32px', marginBottom: '10px' }}>{getEmoji(verdict)}</div>
      <div style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '5px' }}>
        {verdict}
      </div>
      <div style={{ fontSize: '18px' }}>
        Risk Score: {risk.toFixed(1)}%
      </div>
    </div>
  );
}

function ScoreMeter({ label, score, icon }) {
  const normalizedScore = Math.max(0, Math.min(1, Number(score) || 0));
  return (
    <div style={{ marginBottom: '15px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
        <span>{icon} {label}</span>
        <span style={{ fontWeight: 'bold' }}>{(normalizedScore * 100).toFixed(0)}%</span>
      </div>
      <div style={{
        width: '100%',
        height: '8px',
        backgroundColor: '#e0e0e0',
        borderRadius: '4px',
        overflow: 'hidden'
      }}>
        <div style={{
          width: `${normalizedScore * 100}%`,
          height: '100%',
          backgroundColor: normalizedScore > 0.66 ? '#dc3545' : normalizedScore > 0.31 ? '#ffc107' : '#28a745',
          transition: 'width 0.3s ease'
        }} />
      </div>
    </div>
  );
}

function KeyFinding({ finding, index }) {
  const severityColor = {
    'high': '#dc3545',
    'medium': '#ffc107',
    'low': '#28a745'
  };

  return (
    <div style={{
      padding: '12px',
      marginBottom: '10px',
      borderLeft: `4px solid ${severityColor[finding.severity]}`,
      backgroundColor: '#f8f9fa',
      borderRadius: '4px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <strong>{index}. {finding.category}</strong>
        <span style={{
          padding: '2px 8px',
          borderRadius: '4px',
          fontSize: '12px',
          backgroundColor: severityColor[finding.severity],
          color: 'white'
        }}>
          {finding.severity.toUpperCase()}
        </span>
      </div>
      <div style={{ marginTop: '5px', color: '#666' }}>
        Confidence: {(finding.score * 100).toFixed(0)}%
      </div>
    </div>
  );
}

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [videoId, setVideoId] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [notice, setNotice] = useState(null);
  const fileInputRef = useRef(null);

  const showNotice = (message, type = 'error') => {
    setNotice({ message, type });
  };

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.size > 500 * 1024 * 1024) {
        showNotice('File too large (max 500 MB)');
        return;
      }
      const allowed = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'];
      if (!allowed.includes(file.name.slice(file.name.lastIndexOf('.')).toLowerCase())) {
        showNotice('Unsupported video format. Use MP4, AVI, MOV, MKV, FLV, or WEBM.');
        return;
      }
      setNotice(null);
      setVideoFile(file);
    }
  };

  const handleUpload = async () => {
    if (!videoFile) {
      showNotice('Please select a video first');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('async_mode', 'false');

    try {
      const response = await axios.post(`${API_BASE}/api/v1/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (response.data.analysis_id) {
        setVideoId(response.data.analysis_id);
        setResult(response.data);
      } else if (response.data.video_id) {
        setVideoId(response.data.video_id);
        pollResults(response.data.video_id);
      }
    } catch (error) {
      showNotice(`Upload failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setUploading(false);
    }
  };

  const pollResults = async (id) => {
    setLoading(true);
    setProgress(0);
    const interval = setInterval(async () => {
      try {
        const statusRes = await axios.get(`${API_BASE}/api/v1/status/${id}`);
        setProgress(statusRes.data.progress || 0);

        if (statusRes.data.status === 'completed') {
          clearInterval(interval);
          const resultRes = await axios.get(`${API_BASE}/api/v1/results/${id}`);
          setResult(resultRes.data);
          setLoading(false);
        } else if (statusRes.data.status === 'failed') {
          clearInterval(interval);
          setLoading(false);
          showNotice(`Analysis failed: ${statusRes.data.error || 'Unknown error'}`);
        }
      } catch (error) {
        clearInterval(interval);
        setLoading(false);
        showNotice(`Could not retrieve analysis status: ${error.message}`);
      }
    }, 2000);

    setTimeout(() => {
      clearInterval(interval);
      setLoading(false);
    }, 10 * 60 * 1000);
  };

  const downloadJSON = () => {
    if (!result) return;
    const dataStr = JSON.stringify(result, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    const exportFileDefaultName = `deepfake_result_${videoId}.json`;
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px', fontFamily: 'Segoe UI, sans-serif' }}>
      {notice && (
        <div
          role="alert"
          style={{
            padding: '12px 16px',
            marginBottom: '20px',
            borderRadius: '4px',
            backgroundColor: notice.type === 'error' ? '#f8d7da' : '#d1e7dd',
            color: notice.type === 'error' ? '#842029' : '#0f5132',
          }}
        >
          {notice.message}
          <button
            type="button"
            onClick={() => setNotice(null)}
            aria-label="Dismiss notification"
            style={{ float: 'right', border: 0, background: 'transparent', cursor: 'pointer' }}
          >
            ×
          </button>
        </div>
      )}
      <header style={{ textAlign: 'center', marginBottom: '40px', borderBottom: '2px solid #007bff', paddingBottom: '20px' }}>
        <h1 style={{ margin: '0 0 10px 0', color: '#333' }}>🔍 Deepfake Detection Suite</h1>
        <p style={{ margin: '0', color: '#666' }}>Language-Agnostic Analysis | WhatsApp Compression Ready</p>
      </header>

      <div className="dashboard-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '40px' }}>
        <div style={{ padding: '20px', border: '2px dashed #007bff', borderRadius: '8px', textAlign: 'center' }}>
          <h3>📹 Upload Video</h3>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{
              padding: '10px 20px',
              backgroundColor: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              marginBottom: '10px'
            }}
          >
            Choose Video
          </button>
          {videoFile && (
            <div>
              <p>Selected: {videoFile.name}</p>
              <p style={{ fontSize: '12px', color: '#666' }}>
                ({(videoFile.size / 1024 / 1024).toFixed(2)} MB)
              </p>
            </div>
          )}
          <button
            onClick={handleUpload}
            disabled={!videoFile || uploading}
            style={{
              padding: '10px 30px',
              backgroundColor: uploading ? '#6c757d' : '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: uploading ? 'default' : 'pointer',
              marginTop: '10px'
            }}
          >
            {uploading ? 'Uploading...' : 'Analyze'}
          </button>
        </div>

        <div style={{ padding: '20px', backgroundColor: '#f0f4ff', borderRadius: '8px' }}>
          <h3>ℹ️ Instructions</h3>
          <ul style={{ marginTop: '10px', marginBottom: '10px', paddingLeft: '20px' }}>
            <li>Upload MP4, AVI, MOV, or MKV video (max 500 MB)</li>
            <li>Supports any language (auto-detected)</li>
            <li>Analysis takes 1-5 minutes</li>
            <li>Results include risk score and key findings</li>
          </ul>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '15px' }}>
            <strong>Verdict Scale:</strong><br/>
            🟢 Authentic: 0-30% | 🟡 Inconclusive: 31-65% | 🔴 Deepfake: 66-100%
          </div>
        </div>
      </div>

      {loading && (
        <div style={{ marginBottom: '30px', padding: '20px', backgroundColor: '#e7f3ff', borderRadius: '8px' }}>
          <h4>Processing... {progress}%</h4>
          <div style={{
            width: '100%',
            height: '10px',
            backgroundColor: '#e0e0e0',
            borderRadius: '5px',
            overflow: 'hidden'
          }}>
            <div style={{
              width: `${progress}%`,
              height: '100%',
              backgroundColor: '#007bff',
              transition: 'width 0.3s'
            }} />
          </div>
        </div>
      )}

      {result && (
        <div style={{ marginTop: '40px', padding: '30px', backgroundColor: '#f8f9fa', borderRadius: '8px' }}>
          <h2 style={{ marginTop: '0' }}>📊 Analysis Results</h2>
          {result.disclaimer && (
            <div role="note" style={{ padding: '12px', marginBottom: '20px', backgroundColor: '#fff3cd', color: '#664d03', borderRadius: '4px' }}>
              ⚠️ {result.disclaimer}
            </div>
          )}

          <VerdictBadge
            verdict={result.verdict || result.final_result?.verdict}
            risk={result.risk_score || result.final_result?.risk_score || 0}
          />

          <div style={{ marginBottom: '30px' }}>
            <h4 style={{ marginBottom: '15px' }}>Component Analysis</h4>
            <ScoreMeter
              label="Vision AI"
              score={result.component_scores?.vision ?? result.final_result?.vision_score ?? 0.5}
              icon="👁️"
            />
            <ScoreMeter
              label="Audio Analysis"
              score={result.component_scores?.audio ?? result.final_result?.audio_score ?? 0.5}
              icon="🎤"
            />
            <ScoreMeter
              label="Lip-Sync Detection"
              score={result.component_scores?.sync ?? result.final_result?.sync_score ?? 0.5}
              icon="👄"
            />
          </div>

          {result.key_findings && result.key_findings.length > 0 && (
            <div style={{ marginBottom: '30px' }}>
              <h4 style={{ marginBottom: '15px' }}>🔎 Key Findings</h4>
              {result.key_findings.map((finding, idx) => (
                <KeyFinding key={idx} finding={finding} index={idx + 1} />
              ))}
            </div>
          )}

          <div style={{ padding: '15px', backgroundColor: 'white', borderRadius: '4px', marginBottom: '20px' }}>
            <h4 style={{ marginBottom: '10px' }}>📋 Metadata</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '14px' }}>
              <div><strong>Video ID:</strong> {videoId}</div>
              <div><strong>Confidence:</strong> {result.confidence || result.final_result?.confidence_level}</div>
              <div><strong>Language:</strong> {result.detected_language || 'Auto-detected'}</div>
              <div><strong>Processed:</strong> {new Date().toLocaleString()}</div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
            <button
              onClick={downloadJSON}
              style={{
                padding: '10px 20px',
                backgroundColor: '#6c757d',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              📥 Export JSON
            </button>
            <button
              onClick={() => {
                setResult(null);
                setVideoFile(null);
                setVideoId(null);
              }}
              style={{
                padding: '10px 20px',
                backgroundColor: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              ➕ Analyze Another
            </button>
          </div>
        </div>
      )}

      <footer style={{ marginTop: '60px', paddingTop: '20px', borderTop: '1px solid #ddd', textAlign: 'center', color: '#666', fontSize: '12px' }}>
        <p>🇮🇳 Built for election integrity in India | Language-Agnostic Deepfake Detection</p>
        <p>© 2026 SOUL CODERS</p>
      </footer>
    </div>
  );
}

export default App;
