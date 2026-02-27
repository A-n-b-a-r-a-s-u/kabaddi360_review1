import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  StatusBar,
  FlatList,
  Platform,
  AppState,
} from "react-native";
import * as Notifications from "expo-notifications";
import * as Haptics from "expo-haptics";
import { Audio } from "expo-av";
import * as Device from "expo-device";

// ─────────────────────────────────────────────
// CONFIG — Change IP to your laptop's hotspot IP
// ─────────────────────────────────────────────
const WS_URL = "ws://10.7.105.108:8000/ws"; // ← Replace X.X
const RECONNECT_INTERVAL_MS = 3000;
const MAX_LOG_ENTRIES = 50;

// ─────────────────────────────────────────────
// Notification handler (foreground)
// ─────────────────────────────────────────────
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

// ─────────────────────────────────────────────
// Helper: Request notification permissions
// ─────────────────────────────────────────────
async function registerForPushNotifications() {
  if (!Device.isDevice) {
    console.warn("Push notifications only work on physical devices.");
    return false;
  }
  const { status: existingStatus } = await Notifications.getPermissionsAsync();
  let finalStatus = existingStatus;
  if (existingStatus !== "granted") {
    const { status } = await Notifications.requestPermissionsAsync();
    finalStatus = status;
  }
  if (finalStatus !== "granted") {
    console.warn("Notification permission not granted.");
    return false;
  }

  if (Platform.OS === "android") {
    await Notifications.setNotificationChannelAsync("kabaddi-alerts", {
      name: "Kabaddi Alerts",
      importance: Notifications.AndroidImportance.MAX,
      vibrationPattern: [0, 250, 250, 250],
      lightColor: "#FF0000",
      sound: "default",
    });
  }
  return true;
}

// ─────────────────────────────────────────────
// Helper: Send local notification
// ─────────────────────────────────────────────
async function sendNotification(title, body, channelId = "kabaddi-alerts") {
  await Notifications.scheduleNotificationAsync({
    content: {
      title,
      body,
      sound: "default",
      priority: Notifications.AndroidNotificationPriority.MAX,
      ...(Platform.OS === "android" && { channelId }),
    },
    trigger: null, // fire immediately
  });
}

// ─────────────────────────────────────────────
// Helper: Play alert sound
// ─────────────────────────────────────────────
async function playAlertSound() {
  try {
    await Audio.setAudioModeAsync({ playsInSilentModeIOS: true });
    const { sound } = await Audio.Sound.createAsync(
      { uri: "https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3" },
      { shouldPlay: true, volume: 1.0 }
    );
    sound.setOnPlaybackStatusUpdate((status) => {
      if (status.didJustFinish) sound.unloadAsync();
    });
  } catch (e) {
    console.warn("Sound error:", e.message);
  }
}

// ─────────────────────────────────────────────
// Helper: Haptic patterns
// ─────────────────────────────────────────────
async function lightVibration() {
  await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
}

async function mediumVibration() {
  await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
}

async function strongVibration() {
  await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
  setTimeout(() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy), 300);
  setTimeout(() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy), 600);
}

// ─────────────────────────────────────────────
// Helper: Process incoming WebSocket message
// ─────────────────────────────────────────────
async function processMessage(data) {
  let parsed;
  try {
    parsed = typeof data === "string" ? JSON.parse(data) : data;
  } catch {
    console.warn("Invalid JSON from WS:", data);
    return null;
  }

  console.log("📨 Event received from server:", parsed); // Debug log

  const { type, data: eventData } = parsed;

  // ✅ FIXED: Handle actual server event types
  switch (type) {
    case "RAIDER_IDENTIFIED":
      await sendNotification(
        "🏃 Raider Identified",
        `Raider #${eventData?.raider_id} entered at frame ${eventData?.frame}`
      );
      await lightVibration();
      return {
        type,
        label: `🏃 Raider #${eventData?.raider_id} Identified`,
        color: "#3B82F6",
      };

    case "COLLISION":
      const defenders = eventData?.defenders || [];
      await sendNotification(
        "⚠️ Collision Detected",
        `Raider hit by defenders: ${defenders.join(", ")}`
      );
      await mediumVibration();
      return {
        type,
        label: `⚠️ Collision - Defenders: [${defenders.join(", ")}]`,
        color: "#F59E0B",
      };

    case "FALL":
      const fallSeverity = eventData?.fall_severity || 0;
      if (fallSeverity > 60) {
        // Critical fall
        await sendNotification(
          "🚨 CRITICAL FALL DETECTED",
          `Severity: ${fallSeverity.toFixed(1)} - Medical attention needed!`
        );
        await strongVibration();
        await playAlertSound();
        return {
          type,
          label: `🚨 CRITICAL FALL (Severity: ${fallSeverity.toFixed(1)})`,
          color: "#DC2626",
        };
      } else {
        // Minor fall
        await sendNotification(
          "💥 Fall Detected",
          `Severity: ${fallSeverity.toFixed(1)}`
        );
        await mediumVibration();
        return {
          type,
          label: `💥 Fall (Severity: ${fallSeverity.toFixed(1)})`,
          color: "#EF4444",
        };
      }

    case "INJURY_RISK":
      const riskScore = eventData?.risk_score || 0;
      const riskLevel = eventData?.risk_level || "UNKNOWN";

      let riskColor = "#94A3B8"; // Default gray
      let riskEmoji = "📊";

      if (riskLevel === "HIGH") {
        riskColor = "#DC2626";
        riskEmoji = "🔴";
        await sendNotification(
          "🔴 HIGH INJURY RISK",
          `Risk Score: ${riskScore.toFixed(1)}/100`
        );
        await strongVibration();
      } else if (riskLevel === "MEDIUM") {
        riskColor = "#F59E0B";
        riskEmoji = "🟠";
        await sendNotification(
          "🟠 MEDIUM INJURY RISK",
          `Risk Score: ${riskScore.toFixed(1)}/100`
        );
        await lightVibration();
      } else if (riskLevel === "LOW") {
        riskColor = "#10B981";
        riskEmoji = "🟢";
      }

      return {
        type,
        label: `${riskEmoji} Injury Risk: ${riskScore.toFixed(1)}/100 (${riskLevel})`,
        color: riskColor,
      };

    case "CONNECTED":
      // Server connection message
      return {
        type,
        label: "✅ Connected to Server",
        color: "#10B981",
      };

    case "PING":
      // Keep-alive ping from server
      console.log("Ping from server");
      return null; // Don't log pings

    default:
      console.warn("Unknown message type:", type);
      return {
        type,
        label: `Event: ${type}`,
        color: "#6B7280",
      };
  }
}

// ─────────────────────────────────────────────
// Custom hook: WebSocket with auto-reconnect
// ─────────────────────────────────────────────
function useKabaddiWebSocket(url, onMessage) {
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const isMounted = useRef(true);
  const [status, setStatus] = useState("DISCONNECTED");

  const connect = useCallback(() => {
    if (!isMounted.current) return;
    setStatus("CONNECTING");

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!isMounted.current) return;
      console.log("✅ WebSocket connected");
      setStatus("CONNECTED");
      clearTimeout(reconnectTimer.current);
    };

    ws.onmessage = (event) => {
      if (!isMounted.current) return;
      console.log("📨 Raw message received:", event.data);
      onMessage(event.data);
    };

    ws.onerror = (e) => {
      console.warn("❌ WebSocket error:", e.message);
    };

    ws.onclose = () => {
      if (!isMounted.current) return;
      console.log("WebSocket closed — reconnecting...");
      setStatus("DISCONNECTED");
      reconnectTimer.current = setTimeout(connect, RECONNECT_INTERVAL_MS);
    };
  }, [url, onMessage]);

  useEffect(() => {
    registerForPushNotifications();
    connect();

    const appStateSub = AppState.addEventListener("change", (state) => {
      if (state === "active" && wsRef.current?.readyState !== WebSocket.OPEN) {
        clearTimeout(reconnectTimer.current);
        connect();
      }
    });

    return () => {
      isMounted.current = false;
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
      appStateSub.remove();
    };
  }, [connect]);

  return status;
}

// ─────────────────────────────────────────────
// UI Component: Status Badge
// ─────────────────────────────────────────────
function StatusBadge({ status }) {
  const config = {
    CONNECTED: { color: "#22C55E", bg: "#DCFCE7", label: "● Connected" },
    DISCONNECTED: { color: "#EF4444", bg: "#FEE2E2", label: "● Disconnected" },
    CONNECTING: { color: "#F59E0B", bg: "#FEF9C3", label: "◌ Connecting..." },
  }[status] ?? { color: "#6B7280", bg: "#F3F4F6", label: status };

  return (
    <View style={[styles.badge, { backgroundColor: config.bg }]}>
      <Text style={[styles.badgeText, { color: config.color }]}>{config.label}</Text>
    </View>
  );
}

// ─────────────────────────────────────────────
// UI Component: Event Log Item
// ─────────────────────────────────────────────
function LogItem({ item }) {
  return (
    <View style={[styles.logItem, { borderLeftColor: item.color }]}>
      <Text style={styles.logLabel}>{item.label}</Text>
      <Text style={styles.logTime}>{item.time}</Text>
    </View>
  );
}

// ─────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────
export default function App() {
  const [eventLog, setEventLog] = useState([]);
  const [isConnected, setIsConnected] = useState(false);

  const handleMessage = useCallback(async (raw) => {
    const result = await processMessage(raw);
    if (!result) return; // Don't log pings or parsing errors

    const entry = {
      ...result,
      id: Date.now().toString(),
      time: new Date().toLocaleTimeString(),
    };

    console.log("📝 Adding to log:", entry);
    setEventLog((prev) => [entry, ...prev].slice(0, MAX_LOG_ENTRIES));
  }, []);

  const wsStatus = useKabaddiWebSocket(WS_URL, handleMessage);

  // Update connection status
  useEffect(() => {
    setIsConnected(wsStatus === "CONNECTED");
  }, [wsStatus]);

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#F8FAFC" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>🏉 Kabaddi Alert System</Text>
        <Text style={styles.subtitle}>{WS_URL}</Text>
        <StatusBadge status={wsStatus} />
      </View>

      {/* Divider */}
      <View style={styles.divider} />

      {/* Event Log Section */}
      <Text style={styles.sectionTitle}>Live Event Log</Text>

      {/* Waiting State - Only show if connected but no events yet */}
      {isConnected && eventLog.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={styles.waitingEmoji}>⏳</Text>
          <Text style={styles.emptyText}>Waiting for Kabaddi Events...</Text>
          <Text style={styles.emptySubText}>
            Processing video in progress. Events will appear here in real-time when detected:
          </Text>
          <View style={styles.eventTypesList}>
            <Text style={styles.eventItem}>🏃 Raider Identification</Text>
            <Text style={styles.eventItem}>⚠️ Collision Detection</Text>
            <Text style={styles.eventItem}>💥 Fall Detection</Text>
            <Text style={styles.eventItem}>📊 Injury Risk Updates</Text>
          </View>
        </View>
      ) : !isConnected ? (
        // Disconnected State
        <View style={styles.emptyState}>
          <Text style={styles.connectingEmoji}>🔌</Text>
          <Text style={styles.emptyText}>Connecting to Server...</Text>
          <Text style={styles.emptySubText}>
            Make sure the Streamlit dashboard is running with a video being processed.
          </Text>
          <Text style={styles.serverUrl}>{WS_URL}</Text>
        </View>
      ) : (
        // Events List
        <FlatList
          data={eventLog}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => <LogItem item={item} />}
          contentContainerStyle={styles.logList}
          scrollEnabled={true}
        />
      )}
    </View>
  );
}

// ─────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F8FAFC",
    paddingTop: Platform.OS === "android" ? 40 : 60,
  },
  header: {
    paddingHorizontal: 20,
    paddingBottom: 16,
    gap: 6,
  },
  title: {
    fontSize: 24,
    fontWeight: "800",
    color: "#0F172A",
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 12,
    color: "#94A3B8",
    fontFamily: Platform.OS === "ios" ? "Courier" : "monospace",
  },
  badge: {
    alignSelf: "flex-start",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    marginTop: 4,
  },
  badgeText: {
    fontSize: 13,
    fontWeight: "700",
    letterSpacing: 0.3,
  },
  divider: {
    height: 1,
    backgroundColor: "#E2E8F0",
    marginHorizontal: 20,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: "#64748B",
    letterSpacing: 1,
    textTransform: "uppercase",
    paddingHorizontal: 20,
    marginBottom: 10,
  },
  logList: {
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  logItem: {
    backgroundColor: "#FFFFFF",
    borderRadius: 10,
    padding: 14,
    marginBottom: 10,
    borderLeftWidth: 4,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
  },
  logLabel: {
    fontSize: 15,
    fontWeight: "600",
    color: "#1E293B",
  },
  logTime: {
    fontSize: 12,
    color: "#94A3B8",
    marginTop: 4,
  },
  emptyState: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 40,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: "600",
    color: "#CBD5E1",
    marginBottom: 8,
  },
  emptySubText: {
    fontSize: 14,
    color: "#CBD5E1",
    textAlign: "center",
    lineHeight: 20,
  },
});
