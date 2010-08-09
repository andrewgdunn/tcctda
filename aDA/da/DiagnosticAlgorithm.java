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
 * @author Jeremy Mange, Andrew Dunn, Michael Duffy
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
	 * Main routine will instantiate a callback to the DXC framework, the function 'processData' is required
	 * and will be called by the framework. It is passed a DxcData object which can be casted into many other
	 * object types.
	 */
    public static void main(String[] args) {
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
		
		try {
        	mainConnector = ConnectorFactory.getDAConnector(dxcFrameworkCallBack);
        	mainConnector.sendMessage(new ScenarioStatusData(ScenarioStatusData.DA_READY));
            while (isRun) {
                Thread.sleep(threadSleep);
            }
        } catch (Exception ex) {
            System.out.append(ex.toString() + " " + ex.getMessage());
        }
        
        // look for errors; for now just print them out
        Vector<Map<String, Value>> errorSensors = new Vector<Map<String, Value>>();
        
        for(Object keySet : allSensors.keySet()) {
        	Sensor individualSensor = (Sensor)allSensors.get(keySet);
        	Map<String, Value> map = ErrorFinder.errorParams(individualSensor);
        	if(map.containsKey("faultIndex")) {
        		map.put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(map.get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
        		map.remove("faultIndex");
        		System.out.println(individualSensor.id + ":");
        		map.put("sensorId", Value.v(individualSensor.id));
        		errorSensors.add(map);
        	}        		
        	printMap(map);
        }
        
        // based on which sensors found faults, determine which component is problematic
        if(errorSensors.size()==1)
        	reportError(errorSensors.elementAt(0));
        
        // wait for the Oracle response ...
        try {
        	Thread.sleep(threadSleep);
        } catch (Exception e) {
            System.out.append(e.toString() + " " + e.getMessage());
        }
        
        // ... and then choose the lowest-cost action we have
        if(recommendedAction != null) {
        	mainConnector.sendMessage(new CommandData(recommendedAction));
        	System.out.println("DA recommendation: " + ((Command)(recommendedAction.toArray()[0])).getValue() + "\n" );
        }
		
        System.exit(0);
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
			System.out.println(output + "  (cost: " + oracleResponse.getCost() + ")\n");
			
			if(oracleResponse.getCost() < recommendedActionCost) {
				recommendedAction = oracleResponse.getCommands();
				recommendedActionCost = oracleResponse.getCost();
			}
			// Using threading someplace?
			callback.notifyAll();
        }
    }
    
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
    
    private static void CommandData(DxcCallback callback, DxcData daters) {
    	//Do Nothing, We are only handling the ADAPT-Lite system
    }
    
    private static void ScenarioStatusData(DxcCallback callback, DxcData daters) {
    	// all we care about is if it is time to stop
        ScenarioStatusData scenarioStatus = (ScenarioStatusData) daters;
        if (scenarioStatus.getStatus().equals(ScenarioStatusData.ENDED)) {
            isRun = false;
		}
    }
    
    private static void ErrorData(DxcCallback callback, DxcData daters) {
        System.out.print(callback.getClass().getName() + " received Error: ");
        System.out.print(((ErrorData) daters).getError() + "\n");
    }
    
    
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
