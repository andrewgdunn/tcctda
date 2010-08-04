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
import java.util.Vector;
import java.util.Map;
import java.util.HashMap;

/**
 * ExampleJavaDA demonstrates how a diagnosis algorithm should send and receive
 * scenario data.
 */
public class DiagnosticAlgorithm {

    private static boolean run = true;
	private static long startTime = 0;
	
	private static DxcConnector mainConnector;

	// current lowest-cost action
	private static CommandSet recommendedAction = null;
	private static double recommendedActionCost = 9999999;
	
	// all sensor data
	private static Map<String, Sensor> allSensors = new HashMap();
	
	@SuppressWarnings("unused")
    
	private static final DxcCallback exampleCallback = new DxcCallback() {
        public void processData(DxcData d) {
            if(d instanceof RecoveryData) {
            	// Oracle response, determine here what response is best
				synchronized (this) {
                    RecoveryData oracleResponse = (RecoveryData)d;
					
                    String output = "Received from Oracle:\n";
					for(Command com : oracleResponse.getCommands()) {
						output += "   " + com.getCommandID() + ": " + com.getValue();
					}
					System.out.println(output + "  (cost: " + oracleResponse.getCost() + ")\n");
					
					if(oracleResponse.getCost() < recommendedActionCost) {
						recommendedAction = oracleResponse.getCommands();
						recommendedActionCost = oracleResponse.getCost();
					}
					
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
        Vector<Map<String, Value>> errorSensors = new Vector();
        for(Object s : allSensors.keySet()) {
        	Sensor o = (Sensor)allSensors.get(s);
        	Map<String, Value> map = ErrorFinder.errorParams(o);
        	if(map.containsKey("faultIndex")) {
        		int sensorFrequency = (int)(1000 / ( (o.timestamps.elementAt(o.timestamps.size()-1)-o.timestamps.elementAt(0)) / o.timestamps.size() ));
        		//map.put("FaultTime", map.get("faultIndex")/(double)sensorFrequency);
        		map.put("FaultTime", Value.v(o.timestamps.elementAt( ((IntegerValue)(map.get("faultIndex"))).get() ) -
        									 o.timestamps.elementAt(0) ) );
        		map.remove("faultIndex");
        		System.out.println(o.id + ":");
        		
        		map.put("sensorId", Value.v(o.id));
        		errorSensors.add(map);
        	}        		
        	printMap(map);
        }
        
        // based on which sensors found faults, determine which component is problematic
        if(errorSensors.size()==1)
        	reportError(errorSensors.elementAt(0));
        
        // wait for the Oracle response ...
        try {
        	Thread.sleep(1000);
        } catch (Exception ex) {
            System.out.append(ex.toString() + " " + ex.getMessage());
        }
        
        // ... and then choose the lowest-cost action we have
        if(recommendedAction != null) {
        	mainConnector.sendMessage(new CommandData(recommendedAction));
        	System.out.println("DA recommendation: " + ((Command)(recommendedAction.toArray()[0])).getValue() + "\n" );
        }
		
        System.exit(0);
    }

  //---------------------------------------------------------------------------------------------
    
    public static void printMap(Map<String, Value> map) {
    	for(String s:map.keySet()) {
    		System.out.println("   " + s + ": " + map.get(s));
    	}
    }
    
  //---------------------------------------------------------------------------------------------
    
    private static void reportError(Map<String, Value> errorValues) {
    	CandidateSet candidateSet = new CandidateSet();
		Candidate candidate = new Candidate();
		
		StringValueMap faultValues = new StringValueMap();
		for(String s:errorValues.keySet())
			if(Character.isUpperCase(s.charAt(0)))
				faultValues.put(s, errorValues.get(s));
		
		candidate.getFaultSet().add(new Fault( ((StringValue)(errorValues.get("sensorId"))).get() , 
											   ((StringValue)(errorValues.get("faultType"))).get(), 
											   faultValues));
		
		candidate.setWeight(1);
		candidateSet.add(candidate);
		
		// send diagnosis to Oracle
		mainConnector.sendMessage(new DiagnosisData(true, true, candidateSet, "notes"));

		// query Oracle for cost of both ABORT and NOP, choose lower-cost action
		CommandSet commands = new CommandSet();
		commands.add(new Command("systemAction", Value.v("NOP")));
		RecoveryData rd = new RecoveryData(candidate.getFaultSet(), commands);
		
		CommandSet commands2 = new CommandSet();
		commands2.add(new Command("systemAction", Value.v("ABORT")));
		RecoveryData rd2 = new RecoveryData(candidate.getFaultSet(), commands2);
		
		try {
			mainConnector.sendMessage(rd);
			mainConnector.sendMessage(rd2);
		} catch(Exception ex) {
			System.out.println(ex.toString() + " " + ex.getMessage());
		}
    }
}
