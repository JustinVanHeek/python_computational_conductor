package com.example.justinvanheek.positiontracker;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.os.SystemClock;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Set;
import java.util.UUID;

/**
 * Created by Justin.vanHeek on 7/10/2018.
 */

public class Bluetooth {

    private static BluetoothAdapter mBTAdapter = BluetoothAdapter.getDefaultAdapter();
    private static ConnectedThread mConnectedThread; // bluetooth background worker thread to send and receive data
    private static BluetoothSocket mBTSocket = null; // bi-directional client-to-client data path
    private static final UUID BTMODULEUUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB"); // "random" unique identifier
    private static Set<BluetoothDevice> mPairedDevices;

    // #defines for identifying shared types between calling functions
    private final static int REQUEST_ENABLE_BT = 1; // used to identify adding bluetooth names
    private final static int MESSAGE_READ = 2; // used in bluetooth handler to identify message update
    private final static int CONNECTING_STATUS = 3; // used in bluetooth handler to identify message status

    public static void connect() {

        new Thread()
        {
            public void run() {
                boolean fail = false;

                BluetoothDevice device = mBTAdapter.getBondedDevices().iterator().next();

                try {
                    mBTSocket = createBluetoothSocket(device);
                } catch (IOException e) {
                    fail = true;
                }
                // Establish the Bluetooth socket connection.
                try {
                    mBTSocket.connect();
                } catch (IOException e) {
                    try {
                        fail = true;
                        mBTSocket.close();
                    } catch (IOException e2) {
                    }
                }
                if(fail == false) {
                    mConnectedThread = new ConnectedThread(mBTSocket);
                    mConnectedThread.start();

                }
            }
        }.start();
    }

    public static void disconnect() {
        mConnectedThread.cancel();
    }


    public static String send(String message) {
        try {
            String s = "start ";
            if (mConnectedThread != null) {
                s = s + " connect";
                mConnectedThread.write(message);
                s = s + " write " + message + " ";
            } else {
                s = s + " not connect";
            }
            s = s + " done";
            return s;
        }
        catch(Exception e) {
            return "Nope " + e.getMessage();
        }
    }



    private static BluetoothSocket createBluetoothSocket(BluetoothDevice device) throws IOException {
        return  device.createRfcommSocketToServiceRecord(BTMODULEUUID);
        //creates secure outgoing connection with BT device using UUID
    }

    private static class ConnectedThread extends Thread {
        private final BluetoothSocket mmSocket;
        private final InputStream mmInStream;
        private final OutputStream mmOutStream;

        public ConnectedThread(BluetoothSocket socket) {
            mmSocket = socket;
            InputStream tmpIn = null;
            OutputStream tmpOut = null;

            // Get the input and output streams, using temp objects because
            // member streams are final
            try {
                tmpIn = socket.getInputStream();
                tmpOut = socket.getOutputStream();
            } catch (IOException e) { }

            mmInStream = tmpIn;
            mmOutStream = tmpOut;
        }

        public void run() {
            byte[] buffer = new byte[1024];  // buffer store for the stream
            int bytes; // bytes returned from read()

            // Keep listening to the InputStream until an exception occurs
            while (true) {
                try {
                    // Read from the InputStream
                    bytes = mmInStream.read(buffer);
                    if(bytes != 0) {
                        SystemClock.sleep(100);
                        mmInStream.read(buffer);
                    }
                    // Send the obtained bytes to the UI activity

                } catch (IOException e) {
                    break;
                }
            }
        }

        /* Call this from the main activity to send data to the remote device */
        public void write(String input) {
            byte[] bytes = input.getBytes();           //converts entered String into bytes
            try {
                mmOutStream.write(bytes);
            } catch (IOException e) { }
        }

        /* Call this from the main activity to shutdown the connection */
        public void cancel() {
            try {
                mmSocket.close();
            } catch (IOException e) { }
        }
    }
}
