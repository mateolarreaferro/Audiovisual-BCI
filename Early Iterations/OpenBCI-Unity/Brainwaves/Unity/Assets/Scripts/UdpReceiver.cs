using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Non-blocking UDP receiver that listens for EEG JSON packets
/// Uses background thread with lockless queue for main thread consumption
/// </summary>
public class UdpReceiver : MonoBehaviour
{
    [Header("Network Settings")]
    [SerializeField] private string listenIP = "0.0.0.0";
    [SerializeField] private int listenPort = 7777;
    
    [Header("Status")]
    [SerializeField] private bool isReceiving;
    [SerializeField] private float lastPacketTime;
    [SerializeField] private int packetsReceived;
    [SerializeField] private float latencyMs;
    
    // Thread-safe queue for packets
    private ConcurrentQueue<EegPacket> packetQueue = new ConcurrentQueue<EegPacket>();
    private Thread receiveThread;
    private UdpClient udpClient;
    private bool shouldStop;
    
    // Events for packet processing
    public event Action<EegPacket> OnPacketReceived;
    
    // Packet structure matching JSON schema
    [Serializable]
    public class EegPacket
    {
        public float ts;
        public float alpha;
        public float beta;
        public float theta;
        public float delta;
        public float arousal;
        public float left;
        public float right;
        public float front;
        public float back;
        public int calm;
        public int engaged;
        public float[] imu;
        public float receiveTime; // Unity time when received
    }
    
    void Start()
    {
        StartReceiving();
    }
    
    void OnDestroy()
    {
        StopReceiving();
    }
    
    void OnApplicationPause(bool pauseStatus)
    {
        if (pauseStatus) StopReceiving();
        else StartReceiving();
    }
    
    void OnApplicationFocus(bool hasFocus)
    {
        if (!hasFocus) StopReceiving();
        else StartReceiving();
    }
    
    public void StartReceiving()
    {
        if (isReceiving) return;
        
        shouldStop = false;
        receiveThread = new Thread(ReceiveThreadFunction)
        {
            IsBackground = true,
            Name = "EEG UDP Receiver"
        };
        receiveThread.Start();
        isReceiving = true;
        Debug.Log($"[UdpReceiver] Started listening on {listenIP}:{listenPort}");
    }
    
    public void StopReceiving()
    {
        if (!isReceiving) return;
        
        shouldStop = true;
        
        // Close UDP client to unblock receive
        try
        {
            udpClient?.Close();
            udpClient?.Dispose();
        }
        catch { }
        
        // Wait for thread with timeout
        if (receiveThread != null && receiveThread.IsAlive)
        {
            if (!receiveThread.Join(500))
            {
                Debug.LogWarning("[UdpReceiver] Thread didn't stop gracefully, aborting");
                receiveThread.Abort();
            }
        }
        
        isReceiving = false;
        Debug.Log("[UdpReceiver] Stopped");
    }
    
    private void ReceiveThreadFunction()
    {
        try
        {
            // Create UDP client
            IPEndPoint endPoint = new IPEndPoint(IPAddress.Any, listenPort);
            udpClient = new UdpClient(endPoint);
            udpClient.Client.ReceiveTimeout = 100; // 100ms timeout for periodic checks
            
            byte[] buffer;
            IPEndPoint remoteEP = new IPEndPoint(IPAddress.Any, 0);
            
            while (!shouldStop)
            {
                try
                {
                    // Receive data
                    buffer = udpClient.Receive(ref remoteEP);
                    
                    if (buffer != null && buffer.Length > 0)
                    {
                        string json = Encoding.UTF8.GetString(buffer);
                        ProcessPacket(json);
                    }
                }
                catch (SocketException se)
                {
                    // Timeout is normal, continue
                    if (se.SocketErrorCode != SocketError.TimedOut)
                    {
                        Debug.LogError($"[UdpReceiver] Socket error: {se.Message}");
                        Thread.Sleep(10);
                    }
                }
                catch (Exception e)
                {
                    if (!shouldStop)
                    {
                        Debug.LogError($"[UdpReceiver] Receive error: {e.Message}");
                        Thread.Sleep(10);
                    }
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[UdpReceiver] Thread error: {e.Message}");
        }
        finally
        {
            try
            {
                udpClient?.Close();
                udpClient?.Dispose();
            }
            catch { }
        }
    }
    
    private void ProcessPacket(string json)
    {
        try
        {
            // Parse JSON manually for performance (avoid heavy JSON libraries)
            EegPacket packet = ParseSimpleJson(json);
            packet.receiveTime = Time.realtimeSinceStartup;
            
            // Add to queue for main thread
            packetQueue.Enqueue(packet);
            
            // Keep queue size reasonable
            while (packetQueue.Count > 100)
            {
                packetQueue.TryDequeue(out _);
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[UdpReceiver] Failed to parse packet: {e.Message}");
        }
    }
    
    private EegPacket ParseSimpleJson(string json)
    {
        var packet = new EegPacket();
        
        // Simple regex-based parsing for performance
        packet.alpha = ExtractFloat(json, "\"alpha\":", 0f);
        packet.beta = ExtractFloat(json, "\"beta\":", 0f);
        packet.theta = ExtractFloat(json, "\"theta\":", 0f);
        packet.delta = ExtractFloat(json, "\"delta\":", 0f);
        packet.arousal = ExtractFloat(json, "\"arousal\":", 0.5f);
        packet.left = ExtractFloat(json, "\"left\":", 0.5f);
        packet.right = ExtractFloat(json, "\"right\":", 0.5f);
        packet.front = ExtractFloat(json, "\"front\":", 0.5f);
        packet.back = ExtractFloat(json, "\"back\":", 0.5f);
        packet.calm = ExtractInt(json, "\"calm\":", 0);
        packet.engaged = ExtractInt(json, "\"engaged\":", 0);
        packet.ts = ExtractFloat(json, "\"ts\":", 0f);
        
        // Parse IMU array if present
        int imuStart = json.IndexOf("\"imu\":");
        if (imuStart >= 0)
        {
            int arrayStart = json.IndexOf('[', imuStart);
            int arrayEnd = json.IndexOf(']', arrayStart);
            if (arrayStart >= 0 && arrayEnd > arrayStart)
            {
                string imuStr = json.Substring(arrayStart + 1, arrayEnd - arrayStart - 1);
                string[] parts = imuStr.Split(',');
                if (parts.Length >= 3)
                {
                    packet.imu = new float[3];
                    for (int i = 0; i < 3; i++)
                    {
                        float.TryParse(parts[i].Trim(), out packet.imu[i]);
                    }
                }
            }
        }
        
        return packet;
    }
    
    private float ExtractFloat(string json, string key, float defaultValue)
    {
        int idx = json.IndexOf(key);
        if (idx < 0) return defaultValue;
        
        int start = idx + key.Length;
        int end = json.IndexOfAny(new[] { ',', '}' }, start);
        if (end < 0) end = json.Length;
        
        string valueStr = json.Substring(start, end - start).Trim();
        return float.TryParse(valueStr, out float value) ? value : defaultValue;
    }
    
    private int ExtractInt(string json, string key, int defaultValue)
    {
        int idx = json.IndexOf(key);
        if (idx < 0) return defaultValue;
        
        int start = idx + key.Length;
        int end = json.IndexOfAny(new[] { ',', '}' }, start);
        if (end < 0) end = json.Length;
        
        string valueStr = json.Substring(start, end - start).Trim();
        return int.TryParse(valueStr, out int value) ? value : defaultValue;
    }
    
    void Update()
    {
        // Process packets on main thread
        int processed = 0;
        while (packetQueue.TryDequeue(out EegPacket packet) && processed < 10)
        {
            lastPacketTime = Time.time;
            packetsReceived++;
            
            // Calculate latency if timestamp present
            if (packet.ts > 0)
            {
                float currentTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000f;
                latencyMs = (currentTime - packet.ts) * 1000f;
            }
            
            // Notify listeners
            OnPacketReceived?.Invoke(packet);
            processed++;
        }
    }
    
    // Public API
    public float GetPacketRate()
    {
        return packetsReceived / Time.time;
    }
    
    public float GetLatencyMs()
    {
        return latencyMs;
    }
    
    public bool IsConnected()
    {
        return isReceiving && (Time.time - lastPacketTime) < 2f;
    }
}