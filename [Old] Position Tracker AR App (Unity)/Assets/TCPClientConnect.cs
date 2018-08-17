using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class TCPClientConnect : MonoBehaviour
{

    public string serverIP = "10.50.34.104";
    public int port = 80;
    public float recordingInterval = 0.1f;

    private bool connected = false;


    #region private members 	
    private TcpClient socketConnection;
    private Thread clientReceiveThread;
    #endregion
    // Use this for initialization 	
    void Start()
    {
        ConnectToTcpServer();
        InvokeRepeating("SendMessage", recordingInterval, recordingInterval);

    }
    // Update is called once per frame
    void Update()
    {
        if (!connected)
        {
            ConnectToTcpServer();
        }
    }
    /// <summary> 	
    /// Setup socket connection. 	
    /// </summary> 	
    private void ConnectToTcpServer()
    {
        Debug.Log("Connecting " + connected);
        try
        {
            clientReceiveThread = new Thread(new ThreadStart(ListenForData));
            clientReceiveThread.IsBackground = true;
            clientReceiveThread.Start();
        }
        catch (Exception e)
        {
            Debug.Log("On client connect exception " + e);
        }
    }
    /// <summary> 	
    /// Runs in background clientReceiveThread; Listens for incomming data. 	
    /// </summary>     
    private void ListenForData()
    {
        try
        {
            socketConnection = new TcpClient(serverIP, port);
            connected = true;

            Byte[] bytes = new Byte[1024];
            while (true)
            {
                // Get a stream object for reading 				
                using (NetworkStream stream = socketConnection.GetStream())
                {
                    int length;
                    // Read incomming stream into byte arrary. 					
                    while ((length = stream.Read(bytes, 0, bytes.Length)) != 0)
                    {
                        var incommingData = new byte[length];
                        Array.Copy(bytes, 0, incommingData, 0, length);
                        // Convert byte array to string message. 						
                        string serverMessage = Encoding.ASCII.GetString(incommingData);
                        Debug.Log("server message received as: " + serverMessage);
                    }
                }
            }
        }
        catch (SocketException socketException)
        {
            Debug.Log("Socket exception: " + socketException);
        }
    }
    /// <summary> 	
    /// Send message to server using socket connection. 	
    /// </summary> 	
    private void SendMessage()
    {
        Debug.Log("Send Message");
        if (socketConnection == null)
        {
            Debug.Log("No Socket");
            return;
        }
        try
        {
            // Get a stream object for writing. 			
            NetworkStream stream = socketConnection.GetStream();
            if (stream.CanWrite)
            {
                string clientMessage = GetPosition();
                // Convert string message to byte array.                 
                byte[] clientMessageAsByteArray = Encoding.ASCII.GetBytes(clientMessage);
                // Write byte array to socketConnection stream.                 
                stream.Write(clientMessageAsByteArray, 0, clientMessageAsByteArray.Length);
                Debug.Log("Client sent his message - should be received by server");
            }
        }
        catch (SocketException socketException)
        {
            Debug.Log("Socket exception: " + socketException);
        }
    }

    private Vector3 initialPosition;
    private Vector3 initialRotation;
    public Vector3 RotatePointAroundPivot(Vector3 point, Vector3 pivot, Vector3 angles)
    {
        return Quaternion.Euler(angles) * (point - pivot) + pivot;
    }
    private string GetPosition()
    {
        //Record the position data
        Vector3 pos = transform.position - initialPosition;
        Vector3 editedPos = RotatePointAroundPivot(pos, new Vector3(0, 0, 0), new Vector3(0, -initialRotation.y, 0));
        string p = editedPos.x + "," + editedPos.y + "," + editedPos.z;
        return p;
    }
    public void ResetOrigin()
    {
        initialPosition = transform.position;
        initialRotation = transform.eulerAngles;
    }
}