import React, { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { Image } from 'expo-image';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import RealtimeCamera from './RealtimeCamera';
import { API_BASE_URL } from './config';

function getLevelLabel(status) {
  const labels = { LOW: 'Low', BORDERLINE: 'Borderline', SAFE: 'Normal', HIGH: 'High' };
  return labels[status] || status;
}

function getInsightsForStatus(status, value) {
  const insights = {
    LOW: [
      'Possible anemia. Consider a blood test to confirm.',
      'Eat iron-rich foods: leafy greens, beans, fortified cereals, lean meat.',
      'Vitamin C helps iron absorption — pair with citrus or peppers.',
      'Consult a doctor for diagnosis and treatment plan.',
    ],
    BORDERLINE: [
      'Level is below optimal. Good time to improve diet and habits.',
      'Include iron-rich foods and avoid excess tea/coffee with meals.',
      'Get a lab test if you have fatigue, weakness, or dizziness.',
      'Monitor with follow-up checks in a few weeks.',
    ],
    SAFE: [
      'Your level is in the healthy range (WHO: 13.5–17.5 g/dL for adults).',
      'Keep a balanced diet with iron, folate, and B12.',
      'Stay hydrated and maintain regular health check-ups.',
    ],
    HIGH: [
      'High hemoglobin can be due to dehydration, smoking, or other conditions.',
      'Stay well hydrated and avoid smoking.',
      'Consult a doctor to rule out polycythemia or other causes.',
    ],
  };
  return insights[status] || [];
}

export default function App() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('unknown');
  const [useRealtimeMode, setUseRealtimeMode] = useState(false);

  React.useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`, {
        timeout: 10000,
      });
      setApiStatus(response.data.status === 'healthy' ? 'connected' : 'error');
    } catch (error) {
      setApiStatus('disconnected');
      console.log('API Health Check:', error.message);
    }
  };

  const pickImageFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: false,
        aspect: [4, 3],
        quality: 0.8,
        base64: true,
      });

      if (!result.canceled) {
        setResult(null);
        setImage(result.assets[0]);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick image: ' + error.message);
    }
  };

  const pickImageFromCamera = async () => {
    try {
      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: false,
        aspect: [4, 3],
        quality: 0.8,
        base64: true,
      });

      if (!result.canceled) {
        setResult(null);
        setImage(result.assets[0]);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to capture image: ' + error.message);
    }
  };

  const predictHemoglobin = async () => {
    if (!image) {
      Alert.alert('Error', 'Please select an image first');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', {
        uri: image.uri,
        type: 'image/jpeg',
        name: image.filename || 'photo.jpg',
      });

      const response = await axios.post(
        `${API_BASE_URL}/predict`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: 60000,
        }
      );

      if (response.data.status === 'no_eyes_detected') {
        Alert.alert(
          'No eyes detected',
          response.data.message || 'Please provide a clear image of your eye.'
        );
        return;
      }

      setResult({
        hemoglobin: response.data.hemoglobin_estimate,
        unit: response.data.unit || 'g/dL',
        status: response.data.status,
        healthStatus: response.data.health_status,
        healthMessage: response.data.health_message,
        healthColor: response.data.health_color,
        processingTime: response.data.processing_time_ms,
      });

      setApiStatus('connected');
    } catch (error) {
      setApiStatus('error');
      console.error('Prediction error:', error);
      Alert.alert(
        'Prediction Error',
        `Failed to get prediction: ${error.message || 'Unknown error'}`
      );
    } finally {
      setLoading(false);
    }
  };

  if (useRealtimeMode) {
    return (
      <RealtimeCamera onClose={() => setUseRealtimeMode(false)} />
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <View style={styles.header}>
        <View style={styles.headerTop}>
          <View style={styles.logoWrap}>
            <View style={styles.logoIcon} />
            <Text style={styles.title}>HemoLens</Text>
          </View>
          <View style={[styles.statusPill, { backgroundColor: apiStatus === 'connected' ? '#D1FAE5' : apiStatus === 'disconnected' ? '#FEE2E2' : '#FEF3C7' }]}>
            <View style={[styles.statusDot, { backgroundColor: apiStatus === 'connected' ? '#059669' : apiStatus === 'disconnected' ? '#DC2626' : '#D97706' }]} />
            <Text style={[styles.statusText, { color: apiStatus === 'connected' ? '#047857' : apiStatus === 'disconnected' ? '#B91C1C' : '#B45309' }]}>
              {apiStatus === 'connected' ? 'Connected' : apiStatus === 'disconnected' ? 'Offline' : 'Checking…'}
            </Text>
          </View>
        </View>
        <Text style={styles.subtitle}>Non-invasive hemoglobin estimate from eye images</Text>
      </View>

      <View style={styles.section}>
        {image ? (
          <View style={styles.imageCard}>
            <Image
              source={
                image.base64
                  ? { uri: `data:image/jpeg;base64,${image.base64}` }
                  : { uri: image.uri }
              }
              style={styles.image}
              contentFit="cover"
            />
            <Text style={styles.imageMeta}>{Math.round((image.fileSize || 0) / 1024)} KB</Text>
          </View>
        ) : (
          <View style={styles.placeholderCard}>
            <View style={styles.placeholderIcon} />
            <Text style={styles.placeholderTitle}>Add eye image</Text>
            <Text style={styles.placeholderSub}>Camera or gallery — palpebral conjunctiva works best</Text>
          </View>
        )}

        <View style={styles.actionsRow}>
          <TouchableOpacity style={[styles.actionBtn, styles.actionCamera]} onPress={pickImageFromCamera} activeOpacity={0.8}>
            <Text style={styles.actionBtnText}>Camera</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.actionBtn, styles.actionGallery]} onPress={pickImageFromGallery} activeOpacity={0.8}>
            <Text style={styles.actionBtnText}>Gallery</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={[styles.actionBtn, styles.actionRealtime]} onPress={() => setUseRealtimeMode(true)} activeOpacity={0.8}>
          <Text style={styles.actionBtnTextOutlined}>Live detection</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.primaryBtn, (!image || loading) && styles.primaryBtnDisabled]}
          onPress={predictHemoglobin}
          disabled={!image || loading}
          activeOpacity={0.85}
        >
          {loading ? (
            <ActivityIndicator color="#FFF" size="small" />
          ) : (
            <Text style={styles.primaryBtnText}>Analyze hemoglobin</Text>
          )}
        </TouchableOpacity>
      </View>

      {result && (
        <View style={[styles.resultCard, result.healthColor && { borderLeftWidth: 4, borderLeftColor: result.healthColor }]}>
          <View style={styles.resultHead}>
            <Text style={styles.resultHeadTitle}>Result</Text>
            {result.processingTime != null && (
              <Text style={styles.resultMeta}>{result.processingTime} ms</Text>
            )}
          </View>
          <View style={styles.resultMain}>
            <Text style={styles.resultValue}>{result.hemoglobin.toFixed(1)} <Text style={styles.resultUnit}>{result.unit}</Text></Text>
            {result.healthStatus && (
              <View style={[styles.levelBadge, result.healthColor && { backgroundColor: result.healthColor }]}>
                <Text style={styles.levelBadgeText}>{getLevelLabel(result.healthStatus)}</Text>
              </View>
            )}
          </View>
          {result.healthMessage && (
            <Text style={styles.healthMessage}>{result.healthMessage}</Text>
          )}
          {getInsightsForStatus(result.healthStatus, result.hemoglobin).length > 0 && (
            <View style={styles.insightsBlock}>
              <Text style={styles.insightsTitle}>Insights</Text>
              {getInsightsForStatus(result.healthStatus, result.hemoglobin).map((line, idx) => (
                <View key={idx} style={styles.insightRow}>
                  <Text style={styles.insightBullet}>•</Text>
                  <Text style={styles.insightText}>{line}</Text>
                </View>
              ))}
            </View>
          )}
          <Text style={styles.whoRef}>WHO: Low &lt;12 · Normal 12–17.5 · High &gt;17.5 g/dL</Text>
        </View>
      )}

      <View style={styles.footer}>
        <Text style={styles.footerText}>HemoLens</Text>
        <Text style={styles.footerSub}>Estimate only · not a diagnosis</Text>
      </View>
    </ScrollView>
  );
}

const colors = {
  primary: '#0D9488',
  primaryLight: '#CCFBF1',
  surface: '#FFFFFF',
  background: '#F8FAFC',
  text: '#0F172A',
  textSecondary: '#475569',
  textMuted: '#94A3B8',
  border: '#E2E8F0',
  success: '#059669',
  warning: '#D97706',
  error: '#DC2626',
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    paddingBottom: 40,
  },
  header: {
    backgroundColor: colors.surface,
    paddingTop: 48,
    paddingBottom: 24,
    paddingHorizontal: 24,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  logoWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  logoIcon: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: colors.primary,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: colors.text,
    letterSpacing: -0.5,
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 20,
    gap: 6,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600',
  },
  subtitle: {
    fontSize: 14,
    color: colors.textSecondary,
    lineHeight: 20,
  },
  section: {
    paddingHorizontal: 24,
    paddingTop: 24,
  },
  imageCard: {
    backgroundColor: colors.surface,
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 3,
  },
  image: {
    width: '100%',
    height: 260,
    backgroundColor: colors.border,
  },
  imageMeta: {
    padding: 12,
    fontSize: 12,
    color: colors.textMuted,
    textAlign: 'center',
  },
  placeholderCard: {
    backgroundColor: colors.surface,
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
    marginBottom: 16,
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: colors.border,
  },
  placeholderIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.primaryLight,
    marginBottom: 12,
  },
  placeholderTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text,
    marginBottom: 4,
  },
  placeholderSub: {
    fontSize: 13,
    color: colors.textMuted,
    textAlign: 'center',
  },
  actionsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
  },
  actionBtn: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionCamera: {
    backgroundColor: colors.primary,
  },
  actionGallery: {
    backgroundColor: colors.textSecondary,
  },
  actionRealtime: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    marginBottom: 20,
  },
  actionBtnText: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.surface,
  },
  actionRealtime: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    marginBottom: 20,
  },
  actionBtnTextOutlined: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.text,
  },
  primaryBtn: {
    backgroundColor: colors.primary,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 52,
  },
  primaryBtnDisabled: {
    backgroundColor: colors.textMuted,
    opacity: 0.7,
  },
  primaryBtnText: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.surface,
  },
  resultCard: {
    marginHorizontal: 24,
    marginTop: 28,
    backgroundColor: colors.surface,
    borderRadius: 16,
    padding: 20,
    borderLeftWidth: 4,
    borderLeftColor: colors.success,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 3,
  },
  resultHead: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  resultHeadTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.textMuted,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  resultMeta: {
    fontSize: 12,
    color: colors.textMuted,
  },
  resultMain: {
    flexDirection: 'row',
    alignItems: 'baseline',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 12,
  },
  resultValue: {
    fontSize: 32,
    fontWeight: '700',
    color: colors.text,
    letterSpacing: -0.5,
  },
  resultUnit: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.textSecondary,
  },
  levelBadge: {
    backgroundColor: colors.success,
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
  },
  levelBadgeText: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.surface,
  },
  healthMessage: {
    fontSize: 14,
    color: colors.textSecondary,
    lineHeight: 22,
    marginBottom: 16,
  },
  insightsBlock: {
    backgroundColor: colors.background,
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
  },
  insightsTitle: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.textMuted,
    letterSpacing: 0.3,
    marginBottom: 10,
    textTransform: 'uppercase',
  },
  insightRow: {
    flexDirection: 'row',
    marginBottom: 8,
    alignItems: 'flex-start',
  },
  insightBullet: {
    fontSize: 12,
    color: colors.primary,
    marginRight: 8,
    marginTop: 1,
  },
  insightText: {
    flex: 1,
    fontSize: 13,
    color: colors.textSecondary,
    lineHeight: 20,
  },
  whoRef: {
    fontSize: 11,
    color: colors.textMuted,
    fontStyle: 'italic',
  },
  footer: {
    marginTop: 40,
    paddingVertical: 24,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textMuted,
  },
  footerSub: {
    fontSize: 11,
    color: colors.textMuted,
    marginTop: 4,
  },
});
