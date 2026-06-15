import { StatusBar } from "expo-status-bar";
import * as ImagePicker from "expo-image-picker";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Image,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  useWindowDimensions,
} from "react-native";

const DEFAULT_API_URL = "http://localhost:5000";

const pathologyLabels = {
  Infiltration: "Infiltration",
  Effusion: "Effusion",
  Atelectasis: "Atelectasis",
  Pneumothorax: "Pneumothorax",
};

export default function App() {
  const { width, height } = useWindowDimensions();
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [health, setHealth] = useState(null);
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const detectedText = useMemo(() => {
    if (!result) return "Aguardando imagem";
    if (!result.detected_labels?.length) return "Nenhuma patologia acima do threshold";
    return result.detected_labels.join(", ");
  }, [result]);

  const uploadHeight = useMemo(() => {
    if (width >= 900) {
      return Math.min(420, Math.max(280, height * 0.36));
    }
    return Math.min(320, Math.max(220, height * 0.32));
  }, [height, width]);

  async function checkBackend() {
    try {
      setError("");
      const response = await fetch(`${apiUrl}/health`);
      const data = await response.json();
      setHealth(data);
    } catch (err) {
      setHealth(null);
      setError("Nao foi possivel conectar ao backend.");
    }
  }

  async function loadMetrics() {
    try {
      const response = await fetch(`${apiUrl}/metrics`);
      const data = await response.json();
      setMetrics(data);
    } catch {
      setMetrics(null);
    }
  }

  async function pickImage() {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError("Permissao de galeria negada.");
      return;
    }

    const selected = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9,
    });

    if (!selected.canceled) {
      setImage(selected.assets[0]);
      setResult(null);
      setError("");
    }
  }

  async function sendImage() {
    if (!image) {
      setError("Selecione uma imagem antes de classificar.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      const fileName = image.fileName || "xray.png";
      const mimeType = image.mimeType || "image/jpeg";

      if (Platform.OS === "web") {
        const imageResponse = await fetch(image.uri);
        const imageBlob = await imageResponse.blob();
        formData.append("image", new File([imageBlob], fileName, { type: mimeType }));
      } else {
        formData.append("image", {
          uri: image.uri,
          name: fileName,
          type: mimeType,
        });
      }

      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Falha na classificacao.");
      }
      setResult(data);
    } catch (err) {
      setError(err.message || "Erro ao enviar imagem.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    checkBackend();
    loadMetrics();
  }, []);

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.header}>
          <View>
            <Text style={styles.kicker}>CardioIA</Text>
            <Text style={styles.title}>Triagem de raio-X toracico</Text>
          </View>
          <View style={[styles.statusDot, health?.model_exists ? styles.online : styles.offline]} />
        </View>

        <View style={styles.backendPanel}>
          <Text style={styles.label}>Backend Flask</Text>
          <View style={styles.urlRow}>
            <TextInput
              value={apiUrl}
              onChangeText={setApiUrl}
              autoCapitalize="none"
              autoCorrect={false}
              style={styles.input}
            />
            <Pressable style={styles.iconButton} onPress={checkBackend}>
              <MaterialCommunityIcons name="refresh" size={22} color="#123c3a" />
            </Pressable>
          </View>
          <Text style={styles.backendText}>
            {health?.model_exists
              ? `Online em ${health.device}. Threshold ${health.threshold}.`
              : "Aguardando backend e checkpoint do modelo."}
          </Text>
        </View>

        <View style={[styles.uploadArea, { height: uploadHeight }]}>
          {image ? (
            <Image source={{ uri: image.uri }} style={styles.preview} />
          ) : (
            <View style={styles.emptyPreview}>
              <MaterialCommunityIcons name="image-search-outline" size={54} color="#4a6f6b" />
              <Text style={styles.emptyText}>Selecione uma imagem de raio-X</Text>
            </View>
          )}
        </View>

        <View style={styles.actions}>
          <Pressable style={styles.secondaryButton} onPress={pickImage}>
            <MaterialCommunityIcons name="image-plus" size={21} color="#123c3a" />
            <Text style={styles.secondaryText}>Escolher imagem</Text>
          </Pressable>
          <Pressable style={styles.primaryButton} onPress={sendImage} disabled={loading}>
            {loading ? (
              <ActivityIndicator color="#ffffff" />
            ) : (
              <MaterialCommunityIcons name="brain" size={21} color="#ffffff" />
            )}
            <Text style={styles.primaryText}>Classificar</Text>
          </Pressable>
        </View>

        {!!error && <Text style={styles.error}>{error}</Text>}

        <View style={styles.resultPanel}>
          <Text style={styles.sectionTitle}>Resultado</Text>
          <Text style={styles.detected}>{detectedText}</Text>
          {result?.predictions?.map((item) => (
            <View key={item.label} style={styles.resultRow}>
              <View style={styles.resultLabel}>
                <MaterialCommunityIcons
                  name={item.detected ? "alert-circle" : "check-circle-outline"}
                  size={20}
                  color={item.detected ? "#b9502c" : "#2f7d5b"}
                />
                <Text style={styles.pathology}>{pathologyLabels[item.label] || item.label}</Text>
              </View>
              <View style={styles.barTrack}>
                <View style={[styles.barFill, { width: `${Math.round(item.probability * 100)}%` }]} />
              </View>
              <Text style={styles.percent}>{Math.round(item.probability * 100)}%</Text>
            </View>
          ))}
        </View>

        <View style={styles.metricsPanel}>
          <Text style={styles.sectionTitle}>Modelo final</Text>
          <Text style={styles.metricLine}>Vision Transformer ViT</Text>
          <View style={styles.metricGrid}>
            <Metric label="F1 macro" value={metrics?.final_model?.f1_macro} />
            <Metric label="Recall macro" value={metrics?.final_model?.recall_macro} />
            <Metric label="AUC macro" value={metrics?.final_model?.roc_auc_macro} />
            <Metric label="Hamming loss" value={metrics?.final_model?.hamming_loss} />
          </View>
        </View>

      </ScrollView>
    </SafeAreaView>
  );
}

function Metric({ label, value }) {
  const formatted = typeof value === "number" ? value.toFixed(3) : "--";
  return (
    <View style={styles.metricBox}>
      <Text style={styles.metricValue}>{formatted}</Text>
      <Text style={styles.metricLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: "#f7faf9",
  },
  container: {
    padding: 20,
    gap: 16,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginTop: 8,
  },
  kicker: {
    color: "#2d6f67",
    fontSize: 13,
    fontWeight: "700",
    textTransform: "uppercase",
  },
  title: {
    color: "#102221",
    fontSize: 28,
    fontWeight: "800",
    marginTop: 4,
  },
  statusDot: {
    width: 14,
    height: 14,
    borderRadius: 7,
  },
  online: {
    backgroundColor: "#29a36a",
  },
  offline: {
    backgroundColor: "#d35f45",
  },
  backendPanel: {
    backgroundColor: "#ffffff",
    borderRadius: 8,
    padding: 14,
    borderWidth: 1,
    borderColor: "#dce7e4",
  },
  label: {
    color: "#526966",
    fontSize: 12,
    fontWeight: "700",
    marginBottom: 8,
  },
  urlRow: {
    flexDirection: "row",
    gap: 8,
  },
  input: {
    flex: 1,
    minHeight: 44,
    borderWidth: 1,
    borderColor: "#cbdad6",
    borderRadius: 6,
    paddingHorizontal: 12,
    color: "#102221",
    backgroundColor: "#fbfdfc",
  },
  iconButton: {
    width: 44,
    height: 44,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 6,
    backgroundColor: "#e7f0ed",
  },
  backendText: {
    marginTop: 8,
    color: "#526966",
    fontSize: 13,
  },
  uploadArea: {
    borderRadius: 8,
    overflow: "hidden",
    backgroundColor: "#e5efec",
    borderWidth: 1,
    borderColor: "#cbdad6",
  },
  preview: {
    width: "100%",
    height: "100%",
    resizeMode: "contain",
    backgroundColor: "#101817",
  },
  emptyPreview: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  emptyText: {
    color: "#4a6f6b",
    fontSize: 16,
    fontWeight: "700",
  },
  actions: {
    flexDirection: "row",
    gap: 10,
  },
  secondaryButton: {
    flex: 1,
    minHeight: 48,
    borderRadius: 6,
    backgroundColor: "#e7f0ed",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8,
  },
  secondaryText: {
    color: "#123c3a",
    fontWeight: "800",
  },
  primaryButton: {
    flex: 1,
    minHeight: 48,
    borderRadius: 6,
    backgroundColor: "#165d57",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8,
  },
  primaryText: {
    color: "#ffffff",
    fontWeight: "800",
  },
  error: {
    color: "#b23b2d",
    fontWeight: "700",
  },
  resultPanel: {
    backgroundColor: "#ffffff",
    borderRadius: 8,
    padding: 14,
    borderWidth: 1,
    borderColor: "#dce7e4",
    gap: 12,
  },
  sectionTitle: {
    color: "#102221",
    fontSize: 18,
    fontWeight: "800",
  },
  detected: {
    color: "#2d6f67",
    fontSize: 16,
    fontWeight: "800",
  },
  resultRow: {
    gap: 6,
  },
  resultLabel: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  pathology: {
    color: "#102221",
    fontSize: 15,
    fontWeight: "700",
  },
  barTrack: {
    height: 9,
    borderRadius: 5,
    backgroundColor: "#e7eeeb",
    overflow: "hidden",
  },
  barFill: {
    height: "100%",
    minWidth: 2,
    borderRadius: 5,
    backgroundColor: "#47a385",
  },
  percent: {
    alignSelf: "flex-end",
    color: "#526966",
    fontSize: 12,
    fontWeight: "700",
  },
  metricsPanel: {
    backgroundColor: "#ffffff",
    borderRadius: 8,
    padding: 14,
    borderWidth: 1,
    borderColor: "#dce7e4",
    gap: 12,
  },
  metricLine: {
    color: "#526966",
    fontWeight: "700",
  },
  metricGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  metricBox: {
    width: "47%",
    padding: 12,
    borderRadius: 6,
    backgroundColor: "#f0f6f4",
  },
  metricValue: {
    color: "#102221",
    fontSize: 20,
    fontWeight: "800",
  },
  metricLabel: {
    color: "#526966",
    marginTop: 3,
    fontSize: 12,
    fontWeight: "700",
  },
  disclaimer: {
    color: "#607773",
    fontSize: 12,
    lineHeight: 18,
    marginBottom: 20,
  },
});
