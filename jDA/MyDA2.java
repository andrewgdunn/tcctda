/*
  DXC Framework (http://dx-competition.org/)
  Copyright 2008 David Garcia
  
  The DXC Framework is free software: you can redistribute it and/or modify it 
  under the terms of the GNU Lesser General Public License as published by the 
  Free Software Foundation, either version 3 of the License, or (at your 
  option) any later version. 

  This program is distributed in the hope that it will be useful, but WITHOUT 
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
  FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more 
  details. 

  You should have received a copy of the GNU Lesser General Public License 
  along with this program. If not, see <http://www.gnu.org/licenses/> 
 */

import org.dxc.api.DxcCallback;
import org.dxc.api.connection.ConnectorFactory;
import org.dxc.api.connection.DxcConnector;
import org.dxc.api.datatypes.*;
import java.util.Iterator;
import java.util.Map;
import java.util.HashMap;

/**
 * ExampleJavaDA demonstrates how a diagnosis algorithm should send and receive
 * scenario data.
 */
public class MyDA2 {

    private static boolean run = true;
	private static long startTime = 0;
	
	private static DxcConnector mainConnector;
		
	private static RecoveryData oracleResponse;
	
	// all sensor data
	private static Map<String, Sensor> allSensors = new HashMap();
	
	@SuppressWarnings("unused")
    
	private static final DxcCallback exampleCallback = new DxcCallback() {
        public void processData(DxcData d) {
            if(d instanceof RecoveryData) {
            	// Oracle response, determine here what response is best
				synchronized (this) {
                    oracleResponse = (RecoveryData)d;
					String output = "Received from Oracle:\n";
					for(Command com : oracleResponse.getCommands()) {
						output += "   " + com.getCommandID() + ": " + com.getValue();
					}
					System.out.println(output + "  (cost: " + oracleResponse.getCost() + ")\n");
					
					notifyAll();
                }
			} else if (d instanceof SensorData) {
				if(startTime == 0) {
					startTime = d.getTimeStamp();
				}

                SensorData sd = (SensorData) d;
                Map<String, Value> sensors = sd.getSensorValueMap();

                Iterator<String> i = sensors.keySet().iterator();

                while (i.hasNext()) {

                    String sensorID = i.next();
                    Value value = sensors.get(sensorID);
					
					// add sensor to vector
					if(!allSensors.containsKey(sensorID))
						allSensors.put(sensorID, new Sensor(sensorID));
					
					// record sensor reading
					allSensors.get(sensorID).addData(value, d.getTimeStamp());
                }
			} else if (d instanceof CommandData) {
                // nothing to do here for ADAPT-Lite
            } else if (d instanceof ScenarioStatusData) {
            	// all we care about is if it is time to stop
                ScenarioStatusData stat = (ScenarioStatusData) d;
                if (stat.getStatus().equals(ScenarioStatusData.ENDED)) {
                    run = false;
				}
            } else if (d instanceof ErrorData) {
                System.out.print("JavaDA received Error: ");
                System.out.print(((ErrorData) d).getError() + "\n");
            }
        }
    };
    
    //---------------------------------------------------------------------------------------------
	
    
    public static void main(String[] args) {
        try {
        	mainConnector = ConnectorFactory.getDAConnector(exampleCallback);

        	mainConnector.sendMessage(new ScenarioStatusData(ScenarioStatusData.DA_READY));

            while (run) {
				// do something ?
                Thread.sleep(1000);
            }
        } catch (Exception ex) {
            System.out.append(ex.toString() + " " + ex.getMessage());
        }
        
        // look for errors; for now just print them out
        for(Object s : allSensors.keySet()) {
        	Sensor o = (Sensor)allSensors.get(s);
        	if(ErrorFinder.timeOfError(o) != -1)
        		System.out.println(o.id + ": " + (ErrorFinder.timeOfError(o)-startTime));
        }
		
        System.exit(0);
    }

}
