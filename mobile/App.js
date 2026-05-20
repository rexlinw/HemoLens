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

const MODALITIES = [
  { key: 'eye', label: 'Eye', hint: 'Palpebral conjunctiva, well lit' },
  { key: 'nail', label: 'Nail', hint: 'Fingernail bed, clear focus' },
  { key: 'palm', label: 'Palm', hint: 'Open palm, fingers spread' },
];

function getLevelLabel(status) {
  const labels = { LOW: 'Low', BORDERLINE: 'Borderline', SAFE: 'Normal', HIGH: 'High' };
  return labels[status] || status;
}

function getInsightsForStatus(status) {
  const insights = {
    LOW: [
      'Possible anemia. Consider a blood test to confirm.',
      'Eat iron-rich foods and pair with vitamin C for absorption.',
      'Consult a doctor for diagnosis and treatment.',
    ],
    BORDERLINE: [
      'Below optimal range. Improve diet and monitor symptoms.',
      'Avoid tea/coffee with iron-rich meals.',
      'Recheck in a few weeks or get a lab test.',
    ],
    SAFE: [
      'Within healthy range (WHO: 13.5–17.5 g/dL for adults).',
      'Maintain balanced diet with iron, folate, and B12.',
    ],
    HIGH: [
      'High hemoglobin may relate to dehydration or other conditions.',
      'Stay hydrated and consult a doctor if concerned.',
    ],
  };
  return insights[status] || [];
}

function appendImageToFormData(formData, fieldName, asset) {
  formData.append(fieldName, {
    uri: asset.uri,
    type: 'image/jpeg',
    name: asset.fileName || `${fieldName}.jpg`,
  });
}

export default function App() {
  const [images, setImages] = useState({ eye: null, nail: null, palm: null });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('unknown');
  const [useRealtimeMode, setUseRealtimeMode] = useState(false);

  React.useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`, { timeout: 10000 });
      setApiStatus(response.data.status === 'healthy' ? 'connected' : 'error');
    } catch (error) {
      setApiStatus('disconnected');
    }
  };

  const pickImage = async (modality, useCamera) => {
    try {
      const launcher = useCamera
        ? ImagePicker.launchCameraAsync
        : ImagePicker.launchImageLibraryAsync;
      const picked = await launcher({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: false,
        quality: 0.8,
      });
      if (!picked.canceled) {
        setResult(null);
        setImages((prev) => ({ ...prev, [modality]: picked.assets[0] }));
      }
    } catch (error) {
      Alert.alert('Error', error.message);
    }
  };

  const clearModality = (modality) => {
    setImages((prev) => ({ ...prev, [modality]: null }));
    setResult(null);
  };

  const hasAnyImage = MODALITIES.some((m) => images[m.key]);

  const predictEyeOnly = async () => {
    const formData = new FormData();
    formData.append('file', {
      uri: images.eye.uri,
      type: 'image/jpeg',
      name: 'eye.jpg',
    });
    const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000,
    });
    return response;
  };

  const predictMultimodal = async () => {
    if (!hasAnyImage) {
      Alert.alert('Add images', 'Add at least one of: eye, nail, or palm.');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      let healthData = {};
      try {
        const healthRes = await axios.get(`${API_BASE_URL}/health`, { timeout: 10000 });
        healthData = healthRes.data || {};
      } catch (_) {
        /* use multimodal endpoint and fall back if needed */
      }

      const multimodalAvailable = healthData.multimodal_loaded === true;
      const hasNailOrPalm = Boolean(images.nail || images.palm);
      let response;

      const tryMultimodalPost = async () => {
        const formData = new FormData();
        if (images.eye) appendImageToFormData(formData, 'eye_file', images.eye);
        if (images.nail) appendImageToFormData(formData, 'nail_file', images.nail);
        if (images.palm) appendImageToFormData(formData, 'palm_file', images.palm);

        const urls = [
          `${API_BASE_URL}/predict/multimodal`,
          `${API_BASE_URL}/predict-multimodal`,
        ];
        let lastError;
        for (const url of urls) {
          try {
            return await axios.post(url, formData, {
              headers: { 'Content-Type': 'multipart/form-data' },
              timeout: 120000,
            });
          } catch (err) {
            lastError = err;
          }
        }
        throw lastError;
      };

      const isLegacyApiError = (err) => {
        const status = err?.response?.status;
        const detail = err?.response?.data?.detail;
        return (
          status === 405 ||
          status === 404 ||
          (typeof detail === 'string' && detail.toLowerCase().includes('method not allowed'))
        );
      };

      if (multimodalAvailable || hasNailOrPalm) {
        try {
          response = await tryMultimodalPost();
        } catch (multimodalError) {
          if (isLegacyApiError(multimodalError) && images.eye) {
            Alert.alert(
              'Eye-only mode',
              'Server is still on API v1. Using eye image only. Redeploy Render from latest main for nail+palm.'
            );
            response = await predictEyeOnly();
          } else {
            throw multimodalError;
          }
        }
      } else if (images.eye) {
        response = await predictEyeOnly();
      } else {
        Alert.alert(
          'Server update required',
          'Redeploy the backend on Render from the latest main branch to enable nail and palm analysis.'
        );
        return;
      }

      if (response.data.status === 'no_eyes_detected') {
        Alert.alert('Eye not detected', response.data.message || 'Try a clearer eye image or add nail/palm.');
        return;
      }

      setResult({
        hemoglobin: response.data.hemoglobin_estimate,
        unit: response.data.unit || 'g/dL',
        healthStatus: response.data.health_status,
        healthMessage: response.data.health_message,
        healthColor: response.data.health_color,
        modalitiesUsed: response.data.modalities_used || (images.eye ? ['eye'] : []),
        processingTime: response.data.processing_time_ms,
      });
      setApiStatus('connected');
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      Alert.alert('Prediction error', String(detail));
      setApiStatus('error');
    } finally {
      setLoading(false);
    }
  };

  if (useRealtimeMode) {
    return <RealtimeCamera onClose={() => setUseRealtimeMode(false)} />;
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <View style={styles.headerTop}>
          <View style={styles.logoWrap}>
            <View style={styles.logoIcon} />
            <Text style={styles.title}>HemoLens</Text>
          </View>
          <View
            style={[
              styles.statusPill,
              {
                backgroundColor:
                  apiStatus === 'connected' ? '#D1FAE5' : apiStatus === 'disconnected' ? '#FEE2E2' : '#FEF3C7',
              },
            ]}
          >
            <View
              style={[
                styles.statusDot,
                {
                  backgroundColor:
                    apiStatus === 'connected' ? '#059669' : apiStatus === 'disconnected' ? '#DC2626' : '#D97706',
                },
              ]}
            />
            <Text style={styles.statusText}>
              {apiStatus === 'connected' ? 'Connected' : apiStatus === 'disconnected' ? 'Offline' : 'Checking…'}
            </Text>
          </View>
        </View>
        <Text style={styles.subtitle}>Multimodal anemia screening — eye, nail & palm</Text>
      </View>

      <View style={styles.section}>
        {MODALITIES.map((mod) => {
          const asset = images[mod.key];
          return (
            <View key={mod.key} style={styles.modalityCard}>
              <View style={styles.modalityHead}>
                <Text style={styles.modalityLabel}>{mod.label}</Text>
                {asset && (
                  <TouchableOpacity onPress={() => clearModality(mod.key)}>
                    <Text style={styles.clearText}>Clear</Text>
                  </TouchableOpacity>
                )}
              </View>
              {asset ? (
                <Image source={{ uri: asset.uri }} style={styles.modalityImage} contentFit="cover" />
              ) : (
                <View style={styles.modalityPlaceholder}>
                  <Text style={styles.modalityHint}>{mod.hint}</Text>
                </View>
              )}
              <View style={styles.modalityActions}>
                <TouchableOpacity
                  style={styles.modalityBtn}
                  onPress={() => pickImage(mod.key, true)}
                >
                  <Text style={styles.modalityBtnText}>Camera</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.modalityBtn, styles.modalityBtnAlt]}
                  onPress={() => pickImage(mod.key, false)}
                >
                  <Text style={styles.modalityBtnTextAlt}>Gallery</Text>
                </TouchableOpacity>
              </View>
            </View>
          );
        })}

        <TouchableOpacity
          style={styles.liveBtn}
          onPress={() => setUseRealtimeMode(true)}
          activeOpacity={0.85}
        >
          <Text style={styles.liveBtnText}>Live eye detection</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.primaryBtn, (!hasAnyImage || loading) && styles.primaryBtnDisabled]}
          onPress={predictMultimodal}
          disabled={!hasAnyImage || loading}
          activeOpacity={0.85}
        >
          {loading ? (
            <ActivityIndicator color="#FFF" size="small" />
          ) : (
            <Text style={styles.primaryBtnText}>Analyze (multimodal)</Text>
          )}
        </TouchableOpacity>
        <Text style={styles.helpText}>Add one or more images for best accuracy with all three.</Text>
      </View>

      {result && (
        <View style={[styles.resultCard, result.healthColor && { borderLeftColor: result.healthColor }]}>
          <View style={styles.resultHead}>
            <Text style={styles.resultHeadTitle}>Result</Text>
            {result.processingTime != null && (
              <Text style={styles.resultMeta}>{result.processingTime} ms</Text>
            )}
          </View>
          <View style={styles.resultMain}>
            <Text style={styles.resultValue}>
              {result.hemoglobin.toFixed(1)} <Text style={styles.resultUnit}>{result.unit}</Text>
            </Text>
            {result.healthStatus && (
              <View style={[styles.levelBadge, { backgroundColor: result.healthColor || colors.success }]}>
                <Text style={styles.levelBadgeText}>{getLevelLabel(result.healthStatus)}</Text>
              </View>
            )}
          </View>
          {result.modalitiesUsed?.length > 0 && (
            <Text style={styles.modalitiesUsed}>
              Used: {result.modalitiesUsed.join(' · ')}
            </Text>
          )}
          {result.healthMessage && <Text style={styles.healthMessage}>{result.healthMessage}</Text>}
          {getInsightsForStatus(result.healthStatus).length > 0 && (
            <View style={styles.insightsBlock}>
              <Text style={styles.insightsTitle}>Insights</Text>
              {getInsightsForStatus(result.healthStatus).map((line, idx) => (
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
        <Text style={styles.footerSub}>Screening estimate only · not a diagnosis</Text>
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
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { paddingBottom: 40 },
  header: {
    backgroundColor: colors.surface,
    paddingTop: 48,
    paddingBottom: 20,
    paddingHorizontal: 24,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  headerTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  logoWrap: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  logoIcon: { width: 36, height: 36, borderRadius: 10, backgroundColor: colors.primary },
  title: { fontSize: 24, fontWeight: '700', color: colors.text },
  statusPill: { flexDirection: 'row', alignItems: 'center', paddingVertical: 6, paddingHorizontal: 12, borderRadius: 20, gap: 6 },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
  statusText: { fontSize: 12, fontWeight: '600' },
  subtitle: { fontSize: 14, color: colors.textSecondary, lineHeight: 20 },
  section: { paddingHorizontal: 24, paddingTop: 20 },
  modalityCard: {
    backgroundColor: colors.surface,
    borderRadius: 14,
    padding: 12,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: colors.border,
  },
  modalityHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  modalityLabel: { fontSize: 16, fontWeight: '700', color: colors.text },
  clearText: { fontSize: 13, color: colors.primary, fontWeight: '600' },
  modalityImage: { width: '100%', height: 140, borderRadius: 10, backgroundColor: colors.border },
  modalityPlaceholder: {
    height: 100,
    borderRadius: 10,
    backgroundColor: colors.primaryLight,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 12,
  },
  modalityHint: { fontSize: 12, color: colors.textSecondary, textAlign: 'center' },
  modalityActions: { flexDirection: 'row', gap: 10, marginTop: 10 },
  modalityBtn: {
    flex: 1,
    backgroundColor: colors.primary,
    paddingVertical: 10,
    borderRadius: 10,
    alignItems: 'center',
  },
  modalityBtnAlt: { backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.border },
  modalityBtnText: { color: '#FFF', fontWeight: '600', fontSize: 14 },
  modalityBtnTextAlt: { color: colors.text, fontWeight: '600', fontSize: 14 },
  liveBtn: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: 'center',
    marginBottom: 16,
    backgroundColor: colors.surface,
  },
  liveBtnText: { fontSize: 15, fontWeight: '600', color: colors.text },
  primaryBtn: {
    backgroundColor: colors.primary,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    minHeight: 52,
    justifyContent: 'center',
  },
  primaryBtnDisabled: { backgroundColor: colors.textMuted, opacity: 0.7 },
  primaryBtnText: { fontSize: 16, fontWeight: '600', color: '#FFF' },
  helpText: { fontSize: 12, color: colors.textMuted, textAlign: 'center', marginTop: 10 },
  resultCard: {
    marginHorizontal: 24,
    marginTop: 8,
    backgroundColor: colors.surface,
    borderRadius: 16,
    padding: 20,
    borderLeftWidth: 4,
    borderLeftColor: colors.success,
  },
  resultHead: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 12 },
  resultHeadTitle: { fontSize: 13, fontWeight: '700', color: colors.textMuted, textTransform: 'uppercase' },
  resultMeta: { fontSize: 12, color: colors.textMuted },
  resultMain: { flexDirection: 'row', alignItems: 'baseline', flexWrap: 'wrap', gap: 10, marginBottom: 8 },
  resultValue: { fontSize: 32, fontWeight: '700', color: colors.text },
  resultUnit: { fontSize: 16, fontWeight: '600', color: colors.textSecondary },
  levelBadge: { paddingVertical: 6, paddingHorizontal: 12, borderRadius: 8 },
  levelBadgeText: { fontSize: 13, fontWeight: '700', color: '#FFF' },
  modalitiesUsed: { fontSize: 13, color: colors.primary, fontWeight: '600', marginBottom: 8 },
  healthMessage: { fontSize: 14, color: colors.textSecondary, lineHeight: 22, marginBottom: 12 },
  insightsBlock: { backgroundColor: colors.background, borderRadius: 12, padding: 14, marginBottom: 12 },
  insightsTitle: { fontSize: 12, fontWeight: '700', color: colors.textMuted, marginBottom: 8, textTransform: 'uppercase' },
  insightRow: { flexDirection: 'row', marginBottom: 6 },
  insightBullet: { color: colors.primary, marginRight: 8 },
  insightText: { flex: 1, fontSize: 13, color: colors.textSecondary, lineHeight: 20 },
  whoRef: { fontSize: 11, color: colors.textMuted, fontStyle: 'italic' },
  footer: { marginTop: 32, paddingVertical: 24, alignItems: 'center' },
  footerText: { fontSize: 13, fontWeight: '600', color: colors.textMuted },
  footerSub: { fontSize: 11, color: colors.textMuted, marginTop: 4 },
});
