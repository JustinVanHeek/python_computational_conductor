using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEngine.UI;

public class RecordPosition : MonoBehaviour {

	public bool record = false;
	public Button recordButton;
    public InputField fileNameField;
    public Text storagetext;
	public GameObject trail;
    public float recordingInterval = 0.1f;

	private TrailRenderer trailRenderer;

	private ArrayList data = new ArrayList();
    private Vector3 initialPosition;
    private Vector3 initialRotation;

	private string storageDir;
    private bool delayTrail = false;
    public bool lostSurface = false;

    // Use this for initialization
    void Start () {
		Debug.Log ("Loading Record Script");
		storageDir = Application.persistentDataPath;
		Debug.Log ("File Storage Dir: " + storageDir);
		trailRenderer = trail.GetComponent < TrailRenderer> ();
        //button.onClick.AddListener (Record);
        recordButton.GetComponent<Image>().color = Color.green;
    }

    // Update is called once per frame
    void Update () {
		if (record) {

            if(delayTrail)
            {
                delayTrail = false;
            }
            else
            {
                trailRenderer.time = 10000;
            }
        }
	}

    bool badRecording = false;

    void RecordPos()
    {
        if (lostSurface)
        {
            badRecording = true;
        }
        //Record the position data
        Vector3 pos = transform.position - initialPosition;
        data.Add(RotatePointAroundPivot(pos, new Vector3(0, 0, 0), new Vector3(0, -initialRotation.y, 0)));
    }

	public void Record() {
		Debug.Log ("Record Button Clicked!");
		if (record) {
            recordButton.GetComponent<Image>().color = Color.green;
            recordButton.GetComponentInChildren<Text>().text = "Record";
            trail.transform.parent = null;
			Debug.Log ("Stop Recording");
            CancelInvoke();
            record = false;
			SaveData (fileNameField.text);
		} else {
            recordButton.GetComponent<Image>().color = Color.red;
            recordButton.GetComponentInChildren<Text>().text = "Stop";
			trail.transform.parent = transform;
            trail.transform.localPosition = new Vector3(0, 0, 0);
            trailRenderer.time = 0;
            delayTrail = true;
            Debug.Log ("Start Recording");
            InvokeRepeating("RecordPos", recordingInterval, recordingInterval);
            record = true;
            data.Clear();
            initialPosition = transform.position;
            initialRotation = transform.rotation.eulerAngles;

        }
	}

	public void SaveData(string fileName) {
        if (!badRecording)
        {
            if (fileName == null || fileName == "")
            {
                fileName = "positionData";
            }
            Debug.Log("Saving data to " + storageDir + "/" + fileName + ".csv");
            string[] lines = new string[data.Count];

            for (int i = 0; i < lines.Length; i++)
            {
                Vector3 pos = (Vector3)data[i];
                lines[i] = pos.x + "," + pos.y + "," + pos.z;
            }

            System.IO.File.WriteAllLines(storageDir + "/" + fileName + ".csv", lines);

            Debug.Log("Data Saved!");
        }
	}

    public void ClearData()
    {
        System.IO.DirectoryInfo di = new DirectoryInfo(storageDir);

        foreach (FileInfo file in di.GetFiles())
        {
            file.Delete();
        }
    }

    public Vector3 RotatePointAroundPivot(Vector3 point, Vector3 pivot, Vector3 angles)
    {
        return Quaternion.Euler(angles) * (point - pivot) + pivot;
    }
}
