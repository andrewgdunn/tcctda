package gov.dod.army.rdecom.tardec.tcctda;
import org.dxc.api.DxcCallback;
import org.dxc.api.connection.ConnectorFactory;
import org.dxc.api.connection.DxcConnector;
import org.dxc.api.datatypes.*;

import java.util.Iterator;
import java.util.Vector;
import java.util.Map;
import java.util.HashMap;

/**
 * 
 * @author Jeremy Mange, Michael Duffy, Andrew Dunn 
 * @see License and contact information in project root
 * @version 0.0.1
 * 
 * Code adaptation from the supplied example during the PHM DXC'10 competition.
 * Developed for participation in PHM DXC'10 while the authors were employed at 
 * US Army TARDEC (Tank Automotive Research Development Engineering Command)
 * 
 * The code and comments contained in all files do not directly represent the
 * intentions of the authors organization. 
 */
public class DiagnosticAlgorithm {
	private static boolean printDebug = true;
	// Keep the loop alive until scenario is complete
	private static boolean isRun = true;
	private static int threadSleep = 60;
	// Connector to communicate with DXC framework
	private static DxcConnector mainConnector;				
	// Need to refer to this statically, therefore requires it be declared null here
	private static CommandSet recommendedAction = null;
	// Instantiate as a high value, Oracle will certainly return something lower than this.
	private static double recommendedActionCost = 9999999;
	// all sensor data
	private static Map<String, Sensor> allSensors = new HashMap<String, Sensor>();
	
	/**
	 * Main routine will instantiate a call back to the DXC framework, the function 'processData' is required
	 * and will be called by the framework. It is passed a DxcData object which can be casted into many other
	 * object types.
	 * 
	 * Program flow:
	 * 
	 * |---------------------------ConnectAndGetData---------------------------| |---------------------ProcessRecievedData------------------|
	 * instantiate callback -> connect callback -> scenario runs and finishes -> process collected data -> send framework recommended action 
	 * 
	 */
    public static void main(String[] args) {
    	ConnectAndGetData();
    	ProcessRecievedData();		
        System.exit(0);
    }

    private static void ConnectAndGetData() {
		/** Instantiate our connection to the framework, the processData function is the required
		 *  hook that the framework will execute as a call back...
		 **/
    	DxcCallback dxcFrameworkCallBack = new DxcCallback() {
	        public void processData(DxcData dxcData) {
	            if(dxcData instanceof RecoveryData) {
	            	RecoveryData(this, dxcData);
				} 
	            else if (dxcData instanceof SensorData) {
	            	SensorData(this, dxcData);
				} 
	            else if (dxcData instanceof CommandData) {
	            	CommandData(this, dxcData);
	            } 
	            else if (dxcData instanceof ScenarioStatusData) {
	            	ScenarioStatusData(this, dxcData);
	            } 
	            else if (dxcData instanceof ErrorData) {
	            	ErrorData(this, dxcData);
	            }
	        }
	    };
		
	    // with the callback function instantiated, lets make a connection and wait for activity
		try {
        	mainConnector = ConnectorFactory.getDAConnector(dxcFrameworkCallBack);
        	mainConnector.sendMessage(new ScenarioStatusData(ScenarioStatusData.DA_READY));
            while (isRun) {
                Thread.sleep(threadSleep);
            }
        } 
		catch (Exception ex) {
            System.out.append(ex.toString() + " " + ex.getMessage());
        }
    }

    /**
     * function handles when the oracle responds with recovery information. 
     * builds up a string to output the commands and values. will compare the
     * supplied action cost with current recommended cost and select the lesser
     * @param callback - reference to DxcCallback object
     * @param daters - Generic object that can be cast, contains all data
     */
    private static void RecoveryData(DxcCallback callback, DxcData daters) {
    	// Oracle response, determine here what response is best
    	// Using threading someplace?
    	synchronized (callback) {
			// Cast to RecoveryData
            RecoveryData oracleResponse = (RecoveryData) daters;
			
            String output = "Received from Oracle:\n";
			for(Command command : oracleResponse.getCommands()) {
				output += "   " + command.getCommandID() + ": " + command.getValue();
			}
			//System.out.println(output + "  (cost: " + oracleResponse.getCost() + ")\n");
			
			if(oracleResponse.getCost() < recommendedActionCost) {
				recommendedAction = oracleResponse.getCommands();
				recommendedActionCost = oracleResponse.getCost();
			}
			// Using threading someplace?
			callback.notifyAll();
        }
    }
    
    /**
     * function handles scenario where sensor data comes back from callback
     * iterates over the sensorValueMap and pushes new sensors into the static
     * allSensors variable. Adds any new sensor values to the particular sensor
     * in the allSensors variable.
     * @param callback - reference to DxcCallback object
     * @param daters - Generic object that can be cast, contains all data
     */
	private static void SensorData(DxcCallback callback, DxcData daters) {
			// Cast to SensorData 
	        SensorData sensorData = (SensorData) daters;
	        // get value map for keys
	        Map<String, Value> sensors = sensorData.getSensorValueMap();
	        // build iterator
	        Iterator<String> sensorIterator = sensors.keySet().iterator();
	
	        while (sensorIterator.hasNext()) {	
	            String sensorID = sensorIterator.next();
	            Value value = sensors.get(sensorID);
				
				// add sensor to vector, if it doesnt already exist
				if(!allSensors.containsKey(sensorID))
					allSensors.put(sensorID, new Sensor(sensorID));
				
				// record sensor reading
				allSensors.get(sensorID).addData(value, daters.getTimeStamp());
	        }
	    }
    
	/**
     * currently we are only handling the ADAPT-Lite system, if we were to handle
     * another scenario that required us to understand commandData from the oracle
     * we have this stubbed.
     * @param callback - reference to DxcCallback object
     * @param daters - Generic object that can be cast, contains all data
     */
    private static void CommandData(DxcCallback callback, DxcData daters) {
    	//Do Nothing, We are only handling the ADAPT-Lite system
    }
    
    /**
     * handles changing our looping static variable 'isRun' to false when the
     * scenario officially ends, good place to put a dump of variables since we
     * currently don't have much in the way of debugging in real time. 
     * @param callback - reference to DxcCallback object
     * @param daters - Generic object that can be cast, contains all data
     */
    private static void ScenarioStatusData(DxcCallback callback, DxcData daters) {
    	// all we care about is if it is time to stop
        ScenarioStatusData scenarioStatus = (ScenarioStatusData) daters;
        if (scenarioStatus.getStatus().equals(ScenarioStatusData.ENDED)) {
            isRun = false;
		}
    }
    
    /**
     * purely prints out errors that the callback supplies
     * @param callback - reference to DxcCallback object
     * @param daters - Generic object that can be cast, contains all data
     */
    private static void ErrorData(DxcCallback callback, DxcData daters) {    
        System.out.print(callback.getClass().getName() + " received Error: ");
        System.out.print(((ErrorData) daters).getError() + "\n");
    }
    
    private static void ProcessRecievedData() {
        Vector<Map<String, Value>> errorSensors = new Vector<Map<String, Value>>();
        
        for(Object keySet : allSensors.keySet()) {
        	Sensor individualSensor = (Sensor)allSensors.get(keySet);
        	individualSensor.removeOutliers();
        	
        	//Send the individual sensor to our filters, if there is a detected error we will make sure to set the falutIndex. 
        	Map<String, Value> filterSensor = ErrorFinder.errorParams(individualSensor);
        	
        	if(filterSensor.containsKey("faultIndex")) {
        		filterSensor.put("sensorId", Value.v(individualSensor.id));
        		errorSensors.add(filterSensor);
        	}        		
        	if(printDebug)
        		printMap(filterSensor);
        }
        
        // based on which sensors found faults, determine which component is problematic
        Map<String, Value> finalError = ComponentError.finalError(errorSensors, allSensors);
        //printMap(finalError);
        if(finalError.size() >0)
        	reportError(finalError);
        
        // wait for the Oracle response ...
        try {
        	Thread.sleep(threadSleep);
        } catch (Exception e) {
            System.out.append(e.toString() + " " + e.getMessage());
        }
        
        // ... and then choose the lowest-cost action we have
        if(recommendedAction != null) {
        	mainConnector.sendMessage(new CommandData(recommendedAction));
        	//System.out.println("DA recommendation: " + ((Command)(recommendedAction.toArray()[0])).getValue() + "\n" );
        }
    }
        
    /**
     * Just prints out the map of information
     * @param map -- map to print out
     */
    public static void printMap(Map<String, Value> map) {
    	for(String s:map.keySet()) {
    		System.out.println("   " + s + ": " + map.get(s));
    	}
    	System.out.println();
    }
    
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
