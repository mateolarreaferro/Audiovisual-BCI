using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class SimpleTerrainVisualizer : MonoBehaviour
{
    [Header("Terrain Settings")]
    public int gridSize = 32;
    public float terrainScale = 10f;
    public float heightScale = 3f;
    
    [Header("Network")]
    public int udpPort = 12345;
    
    [Header("Visualization")]
    public Material terrainMaterial;
    
    private Mesh terrainMesh;
    private MeshFilter meshFilter;
    private MeshRenderer meshRenderer;
    private Vector3[] vertices;
    private int[] triangles;
    private Color[] colors;
    
    private UdpClient udpClient;
    private Thread udpThread;
    private bool isRunning = false;
    
    private float currentBandPower = 0f;
    private string currentBand = "alpha";
    
    void Start()
    {
        SetupTerrain();
        StartUDPListener();
    }
    
    void SetupTerrain()
    {
        meshFilter = GetComponent<MeshFilter>();
        meshRenderer = GetComponent<MeshRenderer>();
        
        if (meshFilter == null) meshFilter = gameObject.AddComponent<MeshFilter>();
        if (meshRenderer == null) meshRenderer = gameObject.AddComponent<MeshRenderer>();
        
        // Create default material if none provided
        if (terrainMaterial != null)
        {
            meshRenderer.material = terrainMaterial;
        }
        else
        {
            // Create a simple material that supports vertex colors
            Material defaultMat = new Material(Shader.Find("Sprites/Default"));
            defaultMat.color = Color.white;
            meshRenderer.material = defaultMat;
            Debug.Log("Created default material for terrain");
        }
        
        CreateTerrainMesh();
    }
    
    void CreateTerrainMesh()
    {
        terrainMesh = new Mesh();
        terrainMesh.name = "Simple Terrain";
        
        int vertexCount = (gridSize + 1) * (gridSize + 1);
        vertices = new Vector3[vertexCount];
        colors = new Color[vertexCount];
        
        // Create vertices
        for (int z = 0; z <= gridSize; z++)
        {
            for (int x = 0; x <= gridSize; x++)
            {
                int index = z * (gridSize + 1) + x;
                float xPos = (x - gridSize * 0.5f) * terrainScale / gridSize;
                float zPos = (z - gridSize * 0.5f) * terrainScale / gridSize;
                
                vertices[index] = new Vector3(xPos, 0, zPos);
                colors[index] = Color.blue;
            }
        }
        
        // Create triangles
        int triangleCount = gridSize * gridSize * 6;
        triangles = new int[triangleCount];
        int triIndex = 0;
        
        for (int z = 0; z < gridSize; z++)
        {
            for (int x = 0; x < gridSize; x++)
            {
                int bottomLeft = z * (gridSize + 1) + x;
                int bottomRight = bottomLeft + 1;
                int topLeft = bottomLeft + gridSize + 1;
                int topRight = topLeft + 1;
                
                // First triangle
                triangles[triIndex] = bottomLeft;
                triangles[triIndex + 1] = topLeft;
                triangles[triIndex + 2] = bottomRight;
                
                // Second triangle
                triangles[triIndex + 3] = bottomRight;
                triangles[triIndex + 4] = topLeft;
                triangles[triIndex + 5] = topRight;
                
                triIndex += 6;
            }
        }
        
        terrainMesh.vertices = vertices;
        terrainMesh.triangles = triangles;
        terrainMesh.colors = colors;
        terrainMesh.RecalculateNormals();
        
        meshFilter.mesh = terrainMesh;
    }
    
    void StartUDPListener()
    {
        try
        {
            udpClient = new UdpClient(udpPort);
            isRunning = true;
            udpThread = new Thread(new ThreadStart(UDPListener));
            udpThread.Start();
            Debug.Log($"✅ UDP Listener started on port {udpPort}");
            Debug.Log($"Waiting for data from Python server...");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"❌ Failed to start UDP listener: {e.Message}");
        }
    }
    
    void UDPListener()
    {
        IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
        
        while (isRunning)
        {
            try
            {
                byte[] data = udpClient.Receive(ref remoteEndPoint);
                string message = Encoding.UTF8.GetString(data);
                
                Debug.Log($"Received UDP: {message}");
                
                // Parse simple format: "band:value" e.g., "alpha:0.75"
                string[] parts = message.Split(':');
                if (parts.Length == 2)
                {
                    currentBand = parts[0].Trim().ToLower();
                    if (float.TryParse(parts[1], out float value))
                    {
                        currentBandPower = Mathf.Clamp01(value);
                        Debug.Log($"Updated: Band={currentBand}, Power={currentBandPower}");
                    }
                }
            }
            catch (System.Exception e)
            {
                if (isRunning)
                    Debug.LogError($"UDP receive error: {e.Message}");
            }
        }
    }
    
    void Update()
    {
        UpdateTerrain();
    }
    
    void UpdateTerrain()
    {
        if (vertices == null || terrainMesh == null) return;
        
        float time = Time.time;
        
        for (int z = 0; z <= gridSize; z++)
        {
            for (int x = 0; x <= gridSize; x++)
            {
                int index = z * (gridSize + 1) + x;
                
                float xPos = (x - gridSize * 0.5f) / (float)gridSize;
                float zPos = (z - gridSize * 0.5f) / (float)gridSize;
                
                // Create wave based on band power (add base animation even with no data)
                float distance = Mathf.Sqrt(xPos * xPos + zPos * zPos);
                float baseWave = Mathf.Sin(distance * 8f - time * 3f) * 0.1f; // Small base animation
                float powerWave = Mathf.Sin(distance * 15f - time * 8f) * currentBandPower;
                
                vertices[index].y = (baseWave + powerWave) * heightScale;
                
                // Color based on height and band
                Color bandColor = GetBandColor(currentBand);
                float intensity = Mathf.Max(0.1f, currentBandPower); // Minimum visibility
                colors[index] = Color.Lerp(Color.gray, bandColor, intensity);
            }
        }
        
        terrainMesh.vertices = vertices;
        terrainMesh.colors = colors;
        terrainMesh.RecalculateNormals();
        terrainMesh.RecalculateBounds();
    }
    
    Color GetBandColor(string band)
    {
        switch (band)
        {
            case "delta": return new Color(0.5f, 0, 1f);  // Purple
            case "theta": return new Color(0, 0.5f, 1f);  // Blue
            case "alpha": return new Color(0, 1f, 0.5f);  // Green
            case "beta": return new Color(1f, 0.5f, 0);   // Orange
            case "gamma": return new Color(1f, 0, 0.5f);  // Pink
            default: return Color.white;
        }
    }
    
    void OnDestroy()
    {
        isRunning = false;
        if (udpThread != null && udpThread.IsAlive)
        {
            udpThread.Join(1000);
        }
        if (udpClient != null)
        {
            udpClient.Close();
        }
    }
    
    void OnGUI()
    {
        GUI.Label(new Rect(10, 10, 300, 20), $"Band: {currentBand.ToUpper()}");
        GUI.Label(new Rect(10, 30, 300, 20), $"Power: {currentBandPower:F2}");
        GUI.Label(new Rect(10, 50, 300, 20), $"UDP Port: {udpPort}");
    }
}