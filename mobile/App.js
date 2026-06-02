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
import * as FileSystem from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import { Share } from 'react-native';
import RealtimeCamera from './RealtimeCamera';
import { API_BASE_URL } from './config';

const HISTORY_FILE = `${FileSystem.documentDirectory}hemolens_history.json`;
const MAX_HISTORY = 12;
const MIN_API_MAJOR_VERSION = 3;

const MODALITIES = [
  { key: 'eye', label: 'Eye', hint: 'Palpebral conjunctiva, well lit' },
  { key: 'nail', label: 'Finger/Nail', hint: 'Fingernail bed, clear focus' },
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

function formatHistoryDate(isoString) {
  try {
    return new Date(isoString).toLocaleString();
  } catch (_) {
    return isoString;
  }
}

function buildHistorySummary(entries) {
  if (entries.length < 2) {
    return null;
  }

  const latest = entries[0].hemoglobin;
  const oldest = entries[Math.min(entries.length - 1, 4)].hemoglobin;
  const delta = latest - oldest;
  const label = delta > 0.2 ? 'Up' : delta < -0.2 ? 'Down' : 'Stable';

  return {
    label,
    delta,
  };
}

function isSupportedApiVersion(version) {
  const major = Number(String(version || '').split('.')[0]);
  return Number.isFinite(major) && major >= MIN_API_MAJOR_VERSION;
}

async function ensureSupportedBackend() {
  try {
    const rootRes = await axios.get(`${API_BASE_URL}/`, { timeout: 10000 });
    const rootData = rootRes.data || {};
    if (!isSupportedApiVersion(rootData.version)) {
      return { ok: false, reason: 'stale_version' };
    }
    return { ok: true, rootData };
  } catch (_) {
    return { ok: false, reason: 'unreachable' };
  }
}

function CaptureGuide({ modality }) {
  if (modality === 'eye') {
    return (
      <View style={styles.captureGuide}>
        <View style={styles.eyeGuideFrame}>
          <View style={styles.eyeGuideShape}>
            <View style={styles.eyeGuidePupil} />
          </View>
        </View>
        <Text style={styles.captureGuideLabel}>Center the lower eyelid</Text>
      </View>
    );
  }

  if (modality === 'nail') {
    return (
      <View style={styles.captureGuide}>
        <View style={styles.nailGuideFrame}>
          <View style={styles.nailGuidePlate} />
          <View style={styles.nailGuideCuticle} />
        </View>
        <Text style={styles.captureGuideLabel}>Fill the frame with one clean nail bed</Text>
      </View>
    );
  }

  return (
    <View style={styles.captureGuide}>
      <View style={styles.palmGuideFrame}>
        <View style={styles.palmGuideThumb} />
        <View style={styles.palmGuideFingerRow}>
          <View style={styles.palmGuideFinger} />
          <View style={styles.palmGuideFinger} />
          <View style={styles.palmGuideFinger} />
          <View style={styles.palmGuideFinger} />
        </View>
        <View style={styles.palmGuideBase} />
      </View>
      <Text style={styles.captureGuideLabel}>Show an open palm with fingers spread</Text>
    </View>
  );
}

export default function App() {
  const [images, setImages] = useState({ eye: null, nail: null, palm: null });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('unknown');
  const [useRealtimeMode, setUseRealtimeMode] = useState(false);
  const selectedModalities = MODALITIES.filter((m) => images[m.key]).map((m) => m.label);
  const allModalitiesSelected = selectedModalities.length === MODALITIES.length;
  const [retakeNotice, setRetakeNotice] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(true);

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
    setRetakeNotice(null);
  };

  const saveHistory = async (entries) => {
    const trimmed = entries.slice(0, MAX_HISTORY);
    setHistory(trimmed);
    try {
      await FileSystem.writeAsStringAsync(HISTORY_FILE, JSON.stringify(trimmed, null, 2));
    } catch (error) {
      console.log('History save error:', error.message);
    }
  };

  const addHistoryEntry = async (entry) => {
    const next = [entry, ...history].slice(0, MAX_HISTORY);
    await saveHistory(next);
  };

  const exportHistory = async () => {
    if (history.length === 0) {
      Alert.alert('No history', 'There are no saved readings to export yet.');
      return;
    }

    const header = [
      'Date',
      'Hemoglobin (g/dL)',
      'Status',
      'Modalities',
      'Processing Time (ms)',
    ];
    const rows = history.map((entry) => [
      entry.date,
      entry.hemoglobin.toFixed(1),
      entry.healthStatus || '',
      (entry.modalitiesUsed || []).join(' / '),
      entry.processingTime ?? '',
    ]);
    const csv = [header, ...rows]
      .map((row) => row.map((value) => `"${String(value).replace(/"/g, '""')}"`).join(','))
      .join('\n');

    const fileUri = `${FileSystem.documentDirectory}hemolens_history_export.csv`;
    await FileSystem.writeAsStringAsync(fileUri, csv);

    await Share.share({
      url: fileUri,
      title: 'HemoLens history export',
      message: 'HemoLens screening history export for clinician review. Screening estimate only, not a diagnosis.',
    });
  };

function getRetakeAdvice(modality, message) {
  const lower = (message || '').toLowerCase();
  const title = modality.charAt(0).toUpperCase() + modality.slice(1);

  if (lower.includes('dark')) {
    return `${title}: too dark. Move closer to a brighter light source and avoid shadows.`;
  }
  if (lower.includes('overexposed')) {
    return `${title}: too bright. Step away from direct flash or harsh light.`;
  }
  if (lower.includes('blurry') || lower.includes('focus')) {
    return `${title}: blurry. Hold the camera steady and make sure the target is sharply focused.`;
  }
  if (lower.includes('eye')) {
    return 'Eye: retake with the lower eyelid or conjunctiva centered in the frame.';
  }
  if (lower.includes('nail')) {
    return 'Nail: retake with one clean nail bed filling most of the frame.';
  }
  if (lower.includes('palm')) {
    return 'Palm: retake with an open palm, fingers spread, and the full hand visible.';
  }
  return `${title}: retake this image with a clearer, better centered photo.`;
}

function buildRetakeNotice(validation, fallbackMessage) {
  if (!validation) {
    return null;
  }

  const failed = Object.entries(validation)
    .filter(([, result]) => result && result.valid === false)
    .map(([modality, result]) => getRetakeAdvice(modality, result.message));

  if (failed.length === 0) {
    return {
      title: 'Retake needed',
      items: [fallbackMessage || 'One or more images need a clearer retake.'],
    };
  }

  return {
    title: failed.length === 1 ? 'Retake this image' : 'Retake these images',
    items: failed,
  };
}

  const hasAnyImage = MODALITIES.some((m) => images[m.key]);

  const predictEyeOnly = async () => {
    const backendCheck = await ensureSupportedBackend();
    if (!backendCheck.ok) {
      Alert.alert(
        'Backend update required',
        'This server is still on an older API. Please redeploy the latest backend before running predictions.'
      );
      setApiStatus('error');
      return null;
    }

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
      Alert.alert('Add images', 'Add at least one of: eye, finger/nail, or palm.');
      return;
    }

    setLoading(true);
    setResult(null);
    setRetakeNotice(null);

    try {
      let rootData = {};
      let healthData = {};
      const backendCheck = await ensureSupportedBackend();
      if (!backendCheck.ok) {
        Alert.alert(
          'Backend update required',
          'This server is still on an older API. Please redeploy the latest backend before running predictions.'
        );
        setApiStatus('error');
        return;
      }
      rootData = backendCheck.rootData || {};
      try {
        const healthRes = await axios.get(`${API_BASE_URL}/health`, { timeout: 10000 });
        healthData = healthRes.data || {};
      } catch (_) {
        /* use multimodal endpoint and fall back if needed */
      }

      if (!isSupportedApiVersion(rootData.version)) {
        Alert.alert(
          'Backend update required',
          'This server is still on an older API. Please redeploy the latest backend before running predictions.'
        );
        setApiStatus('error');
        return;
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
            response = await predictEyeOnly();
            if (!response) {
              return;
            }
          } else {
            throw multimodalError;
          }
        }
      } else if (images.eye) {
        response = await predictEyeOnly();
        if (!response) {
          return;
        }
      } else {
        Alert.alert(
          'Server update required',
          'Redeploy the backend on Render from the latest main branch to enable nail and palm analysis.'
        );
        return;
      }

      if (response.data.status === 'invalid_image' || response.data.status === 'no_eyes_detected') {
        const notice = buildRetakeNotice(response.data.validation, response.data.message);
        setRetakeNotice(notice);
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
      addHistoryEntry({
        date: new Date().toISOString(),
        hemoglobin: Number(response.data.hemoglobin_estimate),
        healthStatus: response.data.health_status,
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
        <Text style={styles.subtitle}>Multimodal anemia screening — eye, finger/nail & palm</Text>
        <View style={styles.modelBanner}>
          <View style={styles.modelBannerTop}>
            <Text style={styles.modelBannerTitle}>Trained on eye, finger/nail, and palm</Text>
            <Text style={styles.modelBannerMeta}>
              {selectedModalities.length}/3 selected
            </Text>
          </View>
          <View style={styles.modelChips}>
            {MODALITIES.map((mod) => {
              const active = Boolean(images[mod.key]);
              return (
                <View key={mod.key} style={[styles.modelChip, active && styles.modelChipActive]}>
                  <Text style={[styles.modelChipText, active && styles.modelChipTextActive]}>{mod.label}</Text>
                </View>
              );
            })}
          </View>
          <Text style={styles.modelBannerCopy}>
            Best accuracy comes from eye + finger/nail + palm together. You can still analyze with any subset.
          </Text>
        </View>
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
                  <CaptureGuide modality={mod.key} />
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
            <Text style={styles.primaryBtnText}>
              {allModalitiesSelected ? 'Analyze eye + finger/nail + palm' : 'Analyze selected images'}
            </Text>
          )}
        </TouchableOpacity>
        <Text style={styles.helpText}>Use all three captures for the trained multimodal model, or start with one and add more later.</Text>
      </View>

      {retakeNotice && (
        <View style={styles.retakeCard}>
          <Text style={styles.retakeTitle}>{retakeNotice.title}</Text>
          {retakeNotice.items.map((item, idx) => (
            <View key={idx} style={styles.retakeRow}>
              <Text style={styles.retakeBullet}>•</Text>
              <Text style={styles.retakeText}>{item}</Text>
            </View>
          ))}
          <Text style={styles.retakeFooter}>Retake the flagged image(s), then run Analyze again.</Text>
        </View>
      )}

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
          {result.modalitiesUsed?.length > 1 && (
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

      <View style={styles.historyCard}>
        <View style={styles.historyHeader}>
          <View>
            <Text style={styles.historyTitle}>History & trends</Text>
            <Text style={styles.historySubtitle}>Stored locally on this device</Text>
          </View>
          <TouchableOpacity style={styles.exportBtn} onPress={exportHistory} activeOpacity={0.85}>
            <Text style={styles.exportBtnText}>Export CSV</Text>
          </TouchableOpacity>
        </View>

        {historyLoading ? (
          <Text style={styles.historyEmpty}>Loading readings…</Text>
        ) : history.length === 0 ? (
          <Text style={styles.historyEmpty}>Your recent readings will appear here after each successful analyze.</Text>
        ) : (
          <>
            {buildHistorySummary(history) && (
              <View style={styles.trendCard}>
                <Text style={styles.trendLabel}>Trend</Text>
                <Text style={styles.trendValue}>
                  {buildHistorySummary(history).label} {Math.abs(buildHistorySummary(history).delta).toFixed(1)} g/dL
                </Text>
                <Text style={styles.trendText}>Compared with the last few local readings.</Text>
              </View>
            )}

            {history.slice(0, 5).map((entry, idx) => (
              <View key={`${entry.date}-${idx}`} style={styles.historyRow}>
                <View style={styles.historyRowMain}>
                  <Text style={styles.historyDate}>{formatHistoryDate(entry.date)}</Text>
                  <Text style={styles.historyMeta}>
                    {entry.modalitiesUsed?.join(' · ') || 'eye'}
                  </Text>
                </View>
                <View style={styles.historyValueWrap}>
                  <Text style={styles.historyValue}>{entry.hemoglobin.toFixed(1)}</Text>
                  <Text style={styles.historyUnit}>g/dL</Text>
                </View>
              </View>
            ))}
          </>
        )}
        <Text style={styles.historyNote}>Use export for clinician review. These readings are screening estimates only.</Text>
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
  modelBanner: {
    marginTop: 16,
    backgroundColor: colors.primaryLight,
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: '#A7F3D0',
  },
  modelBannerTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 12 },
  modelBannerTitle: { fontSize: 14, fontWeight: '800', color: colors.text },
  modelBannerMeta: { fontSize: 12, color: colors.textSecondary, fontWeight: '600' },
  modelChips: { flexDirection: 'row', gap: 8, marginTop: 10, flexWrap: 'wrap' },
  modelChip: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#B7E4DA',
  },
  modelChipActive: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  modelChipText: { fontSize: 12, fontWeight: '700', color: colors.textSecondary },
  modelChipTextActive: { color: '#FFFFFF' },
  modelBannerCopy: { marginTop: 10, fontSize: 12, color: colors.textSecondary, lineHeight: 18 },
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
  captureGuide: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  captureGuideLabel: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.textSecondary,
    textAlign: 'center',
    marginTop: 8,
  },
  eyeGuideFrame: {
    width: 128,
    height: 38,
    borderRadius: 24,
    borderWidth: 1.5,
    borderColor: '#7DD3FC',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.55)',
  },
  eyeGuideShape: {
    width: 84,
    height: 24,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#FFF',
  },
  eyeGuidePupil: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: colors.primary,
  },
  nailGuideFrame: {
    width: 98,
    height: 56,
    borderRadius: 16,
    borderWidth: 1.5,
    borderColor: '#FDBA74',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.55)',
  },
  nailGuidePlate: {
    width: 42,
    height: 26,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#EA580C',
    backgroundColor: '#FFF7ED',
  },
  nailGuideCuticle: {
    width: 28,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#FDBA74',
    marginTop: 4,
  },
  palmGuideFrame: {
    width: 108,
    height: 68,
    borderRadius: 18,
    borderWidth: 1.5,
    borderColor: '#86EFAC',
    backgroundColor: 'rgba(255,255,255,0.55)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  palmGuideThumb: {
    position: 'absolute',
    left: 18,
    top: 20,
    width: 14,
    height: 24,
    borderRadius: 8,
    backgroundColor: '#BBF7D0',
    transform: [{ rotate: '-22deg' }],
  },
  palmGuideFingerRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 5,
    marginBottom: 4,
  },
  palmGuideFinger: {
    width: 10,
    height: 28,
    borderRadius: 5,
    backgroundColor: '#22C55E',
  },
  palmGuideBase: {
    width: 42,
    height: 18,
    borderRadius: 10,
    backgroundColor: '#86EFAC',
  },
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
  retakeCard: {
    marginHorizontal: 24,
    marginTop: 8,
    borderRadius: 16,
    padding: 16,
    backgroundColor: '#FEF2F2',
    borderWidth: 1,
    borderColor: '#FCA5A5',
  },
  retakeTitle: { fontSize: 14, fontWeight: '800', color: '#991B1B', marginBottom: 8 },
  retakeRow: { flexDirection: 'row', marginBottom: 6 },
  retakeBullet: { color: '#DC2626', marginRight: 8, fontSize: 16, lineHeight: 20 },
  retakeText: { flex: 1, fontSize: 13, color: '#7F1D1D', lineHeight: 19 },
  retakeFooter: { marginTop: 6, fontSize: 12, color: '#991B1B', fontWeight: '600' },
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
  historyCard: {
    marginHorizontal: 24,
    marginTop: 16,
    marginBottom: 24,
    backgroundColor: colors.surface,
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: colors.border,
  },
  historyHeader: { flexDirection: 'row', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 12 },
  historyTitle: { fontSize: 16, fontWeight: '800', color: colors.text },
  historySubtitle: { fontSize: 12, color: colors.textMuted, marginTop: 2 },
  exportBtn: {
    backgroundColor: colors.primary,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 10,
    minHeight: 44,
    justifyContent: 'center',
  },
  exportBtnText: { color: '#FFF', fontSize: 13, fontWeight: '700' },
  historyEmpty: { fontSize: 13, color: colors.textSecondary, lineHeight: 20 },
  trendCard: {
    backgroundColor: colors.background,
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
  },
  trendLabel: { fontSize: 12, fontWeight: '700', color: colors.textMuted, textTransform: 'uppercase' },
  trendValue: { fontSize: 18, fontWeight: '800', color: colors.text, marginTop: 4 },
  trendText: { marginTop: 4, fontSize: 12, color: colors.textSecondary, lineHeight: 18 },
  historyRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  historyRowMain: { flex: 1, paddingRight: 12 },
  historyDate: { fontSize: 13, fontWeight: '700', color: colors.text },
  historyMeta: { fontSize: 11, color: colors.textMuted, marginTop: 3 },
  historyValueWrap: { alignItems: 'flex-end' },
  historyValue: { fontSize: 20, fontWeight: '800', color: colors.primary },
  historyUnit: { fontSize: 11, color: colors.textMuted, marginTop: 1 },
  historyNote: { marginTop: 12, fontSize: 11, color: colors.textMuted, lineHeight: 16 },
});
