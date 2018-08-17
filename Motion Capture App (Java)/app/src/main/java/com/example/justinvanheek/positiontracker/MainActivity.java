package com.example.justinvanheek.positiontracker;

import android.Manifest;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.camera2.params.BlackLevelPattern;
import android.os.Environment;
import android.os.Handler;
import android.os.SystemClock;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.ToggleButton;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private SensorManager mSensorManager;
    private Sensor accelerationSensor;
    private Sensor rotationSensor;
    private Sensor quaternionSensor;

    private boolean record = false;
    private double[] rotation;
    private double[] acceleration;
    private double[] quaternion;
    private double[] initialOrientation;
    private boolean initialOrientationRecorded = false;

    private boolean newBeat = false;


    Handler bluetoothSenderHandler = new Handler();
    int bluetoothMessageFrequency = 25; //milliseconds

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //SENSORS

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        Log.d("MOTION CAPTURE", "Sensor Manager initialized :" + (mSensorManager!=null));

        accelerationSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        Log.d("MOTION CAPTURE", "Acceleration Sensor initialized :" + (accelerationSensor!=null));
        if(accelerationSensor==null)Log.e("MOTION CAPTURE","No Linear Acceleration Sensor");

        rotationSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        Log.d("MOTION CAPTURE", "Gyroscope Sensor initialized :" + (rotationSensor!=null));
        if(rotationSensor==null)Log.e("MOTION CAPTURE","No Gyroscope Sensor");

        quaternionSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
        Log.d("MOTION CAPTURE", "Rotation Sensor initialized :" + (quaternionSensor!=null));
        if(quaternionSensor==null)Log.e("MOTION CAPTURE","No Rotation Vector Sensor");

        //BUTTONS

        final ToggleButton toggleRecord = (ToggleButton) findViewById(R.id.buttonRecord);
        toggleRecord.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                toggleRecording();
            }
        });

        Button beatButton = (Button) findViewById(R.id.buttonBeat);
        beatButton.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View view) {
                newBeat = true;
            }
        });

        ToggleButton connectButton = (ToggleButton) findViewById(R.id.buttonConnect);
        connectButton.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                toggleConnection();
            }
        });

    }

    boolean connected = false;
    private void toggleConnection() {
        if (connected) {
            Bluetooth.disconnect();
            connected = false;
        }
        else {
            Bluetooth.connect();
            connected = true;
        }

    }

    private void toggleRecording() {
        if(record) {
            //Stop recording
            mSensorManager.unregisterListener(this);
            record = false;
        } else {
            //Reset values
            rotation = new double[]{0,0,0};
            acceleration = new double[]{0,0,0};
            quaternion = new double[]{0,0,0,0};
            initialOrientation = new double[]{0,0,0,0};
            initialOrientationRecorded = false;
            //Start recording
            mSensorManager.registerListener(this, accelerationSensor, SensorManager.SENSOR_DELAY_FASTEST);
            mSensorManager.registerListener(this, rotationSensor, SensorManager.SENSOR_DELAY_FASTEST);
            mSensorManager.registerListener(this, quaternionSensor, SensorManager.SENSOR_DELAY_FASTEST);
            record = true;



            bluetoothSenderHandler.postDelayed(new Runnable(){
                public void run(){
                    if (record) {
                        double[] rotVector = quaternion_to_euler_angle(quaternion[3], quaternion[0], quaternion[1], quaternion[2]);
                        rotVector[2] = rotVector[2]-initialOrientation[2];
                        if(rotVector[2] > 180) rotVector[2] = rotVector[2]-360;
                        if(rotVector[2] < -180) rotVector[2] = rotVector[2]+360;
                        if (newBeat) {
                            newBeat = false;
                            Bluetooth.send(acceleration[0] + "," + acceleration[1] + "," + acceleration[2] + "," + rotation[0] + "," + rotation[1] + "," + rotation[2] + "," + rotVector[0] + "," + rotVector[1] + "," + rotVector[2] + ",ENDBEAT");
                        } else {
                            Bluetooth.send(acceleration[0] + "," + acceleration[1] + "," + acceleration[2] + "," + rotation[0] + "," + rotation[1] + "," + rotation[2] + "," + rotVector[0] + "," + rotVector[1] + "," + rotVector[2] + ",END");
                        }
                        bluetoothSenderHandler.postDelayed(this, bluetoothMessageFrequency);
                    }
                }
            }, bluetoothMessageFrequency);

        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if(record) {
            if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
                acceleration = new double[]{event.values[0],event.values[1],event.values[2]};
            } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
                rotation = new double[]{event.values[0],event.values[1],event.values[2]};
            } else if (event.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR) {
                quaternion = new double[]{event.values[0], event.values[1],event.values[2],event.values[3]};
                if(!initialOrientationRecorded) {
                    initialOrientation = quaternion_to_euler_angle(event.values[3], event.values[0],event.values[1],event.values[2]);
                    initialOrientationRecorded = true;
                }
            }

            double[] rotVector = quaternion_to_euler_angle(quaternion[3],quaternion[0],quaternion[1],quaternion[2]);
            rotVector[2] = rotVector[2]-initialOrientation[2];
            if(rotVector[2] > 180) rotVector[2] = rotVector[2]-360;
            if(rotVector[2] < -180) rotVector[2] = rotVector[2]+360;

            if(rotation != null && acceleration != null && quaternion != null) {

                Log.d("MOTION DATA", "Acceleration = " + acceleration[0] + ", " + acceleration[1] + ", " + acceleration[2] + "  " + "Rotation " + rotation[0] + ", " + rotation[1] + ", " + rotation[2]);
                Log.d("QUAT DATA", quaternion[0] + "  " + quaternion[1] + "  " + quaternion[2] + "  " + quaternion[3]);
                Log.d("ROTV DATA", rotVector[0] + "  " + rotVector[1] + "  " + rotVector[2]);

            }

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    public static double[] quaternion_to_euler_angle(double w, double x, double y, double z) {
        double ysqr = y * y;

        double t0 = +2.0 * (w * x + y * z);
        double t1 = +1.0 - 2.0 * (x * x + ysqr);
        double X = Math.toDegrees(Math.atan2(t0, t1));

        double t2 = +2.0 * (w * y - z * x);
        if (t2 > +1.0) {
            t2 = +1.0;
        }
        if (t2 < -1.0) {
            t2 = -1.0;
        }
        double Y = Math.toDegrees(Math.asin(t2));

        double t3 = +2.0 * (w * z + x * y);
        double t4 = +1.0 - 2.0 * (ysqr + z * z);
        double Z = Math.toDegrees(Math.atan2(t3, t4));

        return new double[]{X,Y,Z};
    }

}

