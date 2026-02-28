import React, { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Dimensions,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import axios from 'axios';
import { API_BASE_URL } from './config';
const CAPTURE_INTERVAL = 1500;
const MAX_HISTORY = 5;

export default function RealtimeCamera({ onClose }) {
  const cameraRef = useRef(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [averageHemoglobin, setAverageHemoglobin] = useState(null);
  const [frameCount, setFrameCount] = useState(0);
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [statusColor, setStatusColor] = useState('#FFC107');
  const captureIntervalRef = useRef(null);

  const [permission, requestPermission] = useCameraPermissions();

  useEffect(() => {
    if (!permission) {
      requestPermission();
    }
  }, [permission, requestPermission]);

  useEffect(() => {
    checkAPIConnection();
  }, []);

  useEffect(() => {
    if (isCameraReady && permission?.granted && connectionStatus === 'connected') {
      captureIntervalRef.current = setInterval(captureFrame, CAPTURE_INTERVAL);
      return () => clearInterval(captureIntervalRef.current);
    }
  }, [isCameraReady, permission, connectionStatus]);

  useEffect(() => {
    if (predictions.length > 0) {
      const average =
        predictions.reduce((sum, p) => sum + p.hemoglobin, 0) / predictions.length;
      setAverageHemoglobin(average);
    }
  }, [predictions]);

  const checkAPIConnection = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`, { timeout: 5000 });
      if (response.data.status === 'healthy') {
        setConnectionStatus('connected');
        setStatusColor('#4CAF50');
      }
    } catch (error) {
      setConnectionStatus('error');
      setStatusColor('#FF5252');
    }
  };

  const captureFrame = async () => {
    if (!cameraRef.current || isProcessing) return;

    try {
      setIsProcessing(true);
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.7,
        skipProcessing: true,
      });

      if (photo) {
        setFrameCount((prev) => prev + 1);
        await sendFrameForPrediction(photo);
      }
    } catch (error) {
      console.log('Capture error:', error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const sendFrameForPrediction = async (photo) => {
    try {
      const formData = new FormData();
      formData.append('file', {
        uri: photo.uri,
        type: 'image/jpeg',
        name: `frame_${frameCount}.jpg`,
      });

      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000,
      });

      if (response.data.status === 'no_eyes_detected') {
        console.log('No eyes detected:', response.data.message);
        return;
      }

      if (response.data && response.data.hemoglobin_estimate !== undefined) {
        const newPrediction = {
          hemoglobin: response.data.hemoglobin_estimate,
          timestamp: new Date().toLocaleTimeString(),
          health_status: response.data.health_status,
          health_message: response.data.health_message,
          health_color: response.data.health_color,
        };

        setPredictions((prev) => {
          const updated = [newPrediction, ...prev];
          return updated.slice(0, MAX_HISTORY);
        });
      }
    } catch (error) {
      console.log('Prediction error:', error.message);
    }
  };

  const handleStop = () => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
    }
    onClose();
  };

  if (!permission) {
    return (
      <View style={styles.container}>
        <Text style={styles.centerText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.centerText}>Camera permission denied</Text>
        <Text style={styles.smallText}>
          Please enable camera access in your device settings
        </Text>
        <TouchableOpacity
          style={styles.closeBtn}
          onPress={handleStop}
        >
          <Text style={styles.closeBtnText}>Close</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        onCameraReady={() => setIsCameraReady(true)}
        facing="front"
      />

      <View style={styles.overlay} pointerEvents="box-none">
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Live detection</Text>
          <Text style={styles.headerSubtitle}>Position eye in frame</Text>
          <View style={[styles.statusPill, { backgroundColor: connectionStatus === 'connected' ? 'rgba(5, 150, 105, 0.25)' : 'rgba(220, 38, 38, 0.25)' }]}>
            <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
            <Text style={styles.statusPillText}>{connectionStatus === 'connected' ? 'Connected' : 'Offline'}</Text>
          </View>
        </View>

        <View style={styles.guidanceWrap}>
          <View style={styles.guidanceRing} />
          <Text style={styles.guidanceLabel}>Palpebral conjunctiva</Text>
        </View>

        <View style={styles.bottomSection}>
          <View style={[styles.resultCard, predictions.length > 0 && predictions[0].health_color && { backgroundColor: predictions[0].health_color, opacity: 0.95 }]}>
            {averageHemoglobin !== null ? (
              <>
                <Text style={styles.resultLabel}>Hemoglobin</Text>
                <Text style={styles.resultValue}>{averageHemoglobin.toFixed(1)} <Text style={styles.resultUnit}>g/dL</Text></Text>
                {predictions.length > 0 && (
                  <>
                    <View style={styles.levelWrap}>
                      <Text style={styles.healthStatus}>{predictions[0].health_status}</Text>
                    </View>
                    <Text style={styles.healthMessage}>{predictions[0].health_message}</Text>
                  </>
                )}
                <Text style={styles.resultMeta}>{predictions.length} readings</Text>
              </>
            ) : (
              <>
                <Text style={styles.resultLabel}>Initializing</Text>
                <ActivityIndicator size="large" color="#0D9488" style={{ marginVertical: 8 }} />
              </>
            )}
          </View>
          {isProcessing && (
            <View style={styles.processingPill}>
              <ActivityIndicator size="small" color="#FFF" />
              <Text style={styles.processingText}>Analyzing…</Text>
            </View>
          )}
          <Text style={styles.frameCount}>{frameCount} frames</Text>

          {predictions.length > 0 && (
            <View style={styles.recentPanel}>
              <Text style={styles.recentTitle}>Recent</Text>
              {predictions.slice(0, 3).map((pred, idx) => (
                <View key={idx} style={styles.recentRow}>
                  <Text style={styles.recentTime}>{pred.timestamp}</Text>
                  <Text style={styles.recentValue}>{pred.hemoglobin.toFixed(1)} g/dL</Text>
                </View>
              ))}
            </View>
          )}
        </View>
      </View>

      <View style={styles.controlsContainer}>
        <TouchableOpacity style={styles.closeBtn} onPress={handleStop} activeOpacity={0.85}>
          <Text style={styles.closeBtnText}>Stop</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'space-between',
    paddingTop: 56,
    paddingBottom: 28,
    paddingHorizontal: 24,
  },
  header: {
    backgroundColor: 'rgba(15, 23, 42, 0.85)',
    paddingVertical: 14,
    paddingHorizontal: 18,
    borderRadius: 16,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#FFF',
    marginBottom: 2,
    letterSpacing: -0.3,
  },
  headerSubtitle: {
    fontSize: 13,
    color: 'rgba(255,255,255,0.7)',
    marginBottom: 10,
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderRadius: 12,
    gap: 6,
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  statusPillText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#FFF',
  },
  guidanceWrap: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  guidanceRing: {
    width: 140,
    height: 140,
    borderRadius: 70,
    borderWidth: 2,
    borderColor: 'rgba(13, 148, 136, 0.6)',
    marginBottom: 10,
  },
  guidanceLabel: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.8)',
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 8,
    overflow: 'hidden',
  },
  bottomSection: {
    alignItems: 'center',
  },
  resultCard: {
    backgroundColor: 'rgba(13, 148, 136, 0.92)',
    paddingVertical: 20,
    paddingHorizontal: 24,
    borderRadius: 16,
    minWidth: '88%',
    alignItems: 'center',
    marginBottom: 10,
  },
  resultLabel: {
    fontSize: 11,
    fontWeight: '600',
    color: 'rgba(255,255,255,0.85)',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
    marginBottom: 4,
  },
  resultValue: {
    fontSize: 28,
    fontWeight: '700',
    color: '#FFF',
    letterSpacing: -0.5,
  },
  resultUnit: {
    fontSize: 14,
    fontWeight: '600',
    opacity: 0.9,
  },
  levelWrap: {
    marginTop: 8,
  },
  healthStatus: {
    fontSize: 14,
    fontWeight: '700',
    color: '#FFF',
  },
  healthMessage: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.9)',
    marginTop: 4,
    textAlign: 'center',
  },
  resultMeta: {
    fontSize: 11,
    color: 'rgba(255,255,255,0.7)',
    marginTop: 8,
  },
  processingPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(13, 148, 136, 0.9)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 12,
    marginBottom: 8,
    gap: 8,
  },
  processingText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#FFF',
  },
  frameCount: {
    fontSize: 11,
    color: 'rgba(255,255,255,0.6)',
    marginBottom: 8,
  },
  recentPanel: {
    backgroundColor: 'rgba(15, 23, 42, 0.8)',
    borderRadius: 12,
    padding: 12,
    width: '100%',
    maxWidth: 320,
  },
  recentTitle: {
    fontSize: 11,
    fontWeight: '700',
    color: 'rgba(255,255,255,0.7)',
    letterSpacing: 0.3,
    marginBottom: 8,
    textTransform: 'uppercase',
  },
  recentRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.08)',
  },
  recentTime: {
    fontSize: 11,
    color: 'rgba(255,255,255,0.6)',
  },
  recentValue: {
    fontSize: 12,
    fontWeight: '600',
    color: '#0D9488',
  },
  controlsContainer: {
    position: 'absolute',
    bottom: 28,
    left: 24,
    right: 24,
    zIndex: 10,
    alignItems: 'center',
  },
  closeBtn: {
    backgroundColor: 'rgba(220, 38, 38, 0.95)',
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 12,
    minWidth: 120,
    alignItems: 'center',
  },
  closeBtnText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#FFF',
  },
  centerText: {
    fontSize: 16,
    color: '#FFF',
    textAlign: 'center',
  },
  smallText: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.7)',
    textAlign: 'center',
    marginTop: 10,
    marginBottom: 30,
  },
});
