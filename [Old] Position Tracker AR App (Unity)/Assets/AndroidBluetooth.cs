using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class AndroidBluetooth : MonoBehaviour {

    private Vector3 initialPosition;
    private Vector3 initialRotation;
    private const float recordingInterval = 0.05f;
    private bool connected = false;
    private AndroidJavaClass bluetoothClass;
    public GameObject text;
    public bool lostSurface = false;

    public float rangeFromOrigin;
    private bool leftOrigin = false;
    private Vector3 lastPos = new Vector3(0, 0, 0);
	private bool crossedStartLine = false;

    private void Start()
    {
        AndroidJavaClass bluetoothClass2 = new AndroidJavaClass("com.example.justinvanheek.bluetoothsimple.Bluetooth");
        bluetoothClass = bluetoothClass2;
        InvokeRepeating("SendPosition", recordingInterval, recordingInterval);
        
    }
    public Vector3 RotatePointAroundPivot(Vector3 point, Vector3 pivot, Vector3 angles)
    {
        return Quaternion.Euler(angles) * (point - pivot) + pivot;
    }
    private Vector3 GetPosition()
    {
        //Record the position data
        Vector3 pos = transform.position - initialPosition;
        Vector3 editedPos = RotatePointAroundPivot(pos, new Vector3(0, 0, 0), new Vector3(0, -initialRotation.y, 0));
        return editedPos;
    }
    private string GetPos()
    {
        Vector3 editedPos = GetPosition();
        string p = editedPos.x + "," + editedPos.y + "," + editedPos.z;
        return p;
    }
    public void ResetOrigin()
    {
        initialPosition = transform.position;
        initialRotation = transform.eulerAngles;
    }

    public void Connect()
    {
        text.GetComponent<Text>().text = "Connecting";

        connected = true;
        try
        {
            bluetoothClass.CallStatic("connect");
            text.GetComponent<Text>().text = "Connected";

        }
        catch (System.Exception e)
        {
            string error = e.Message;
            text.GetComponent<Text>().text = error;
        }
    }

    public void SendPosition()
    {
        if (connected)
        {
            try
            {
                bool newBeat = false;
				if (!crossedStartLine && transform.position.y > initialPosition.y)
                {
					crossedStartLine = true;
				}
				else if (crossedStartLine && transform.position.y < initialPosition.y) {
					newBeat = true;
					crossedStartLine = false;
				}

                string[] pos = { GetPos()+ "END" };
                if (newBeat)
                {
                    pos[0] = pos[0] + "BEAT";
                    newBeat = false;
                    ResetOrigin();
                }
                if (lostSurface)
                {
                    pos[0] = "END";
                }
            //text.GetComponent<Text>().text = "Sending Pos: " + pos[0];

            string result = bluetoothClass.CallStatic<string>("send", pos);
				text.GetComponent<Text>().text = " Distance = " + (initialPosition.y - transform.position.y) + " -" + result + "-   Sent " + pos[0];

            }
            catch (System.Exception e)
            {
                string error = e.Message;
                text.GetComponent<Text>().text = error;
            }
        }
    }
}
