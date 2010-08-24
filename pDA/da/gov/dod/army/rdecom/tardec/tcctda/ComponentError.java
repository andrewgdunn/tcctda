package gov.dod.army.rdecom.tardec.tcctda;

import org.dxc.api.datatypes.*;

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
public class ComponentError {
	public static Map<String, Value> finalError(Vector<Map<String, Value>> errorSensors, Map<String, Sensor> allSensors) {
		// returns the component error to report to the Oracle
		Map<String, Value> componentError = new HashMap<String, Value>();
		
		// if there is only one error, it should be a sensor error, we can just report it back directly
		if(errorSensors.size()==1) {
			Sensor individualSensor = allSensors.get( ((StringValue)errorSensors.elementAt(0).get("sensorId")).get() );
			errorSensors.elementAt(0).put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(errorSensors.elementAt(0).get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
			errorSensors.elementAt(0).remove("faultIndex");
    		return errorSensors.elementAt(0);
		}
		
		// next simplest -- if there is a boolean error
		for(int i=0; i<errorSensors.size(); i++) {
			String id = ((StringValue)errorSensors.elementAt(i).get("sensorId")).get(); 
			if(id.equals("ISH236")) {
				componentError.put("sensorId", Value.v("CB236"));
				componentError.put("faultType", Value.v("FailedOpen"));
				
				Sensor individualSensor = allSensors.get( "ISH236" );
				componentError.put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(errorSensors.elementAt(i).get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
				
				return componentError;
			} else if(id.equals("ESH244A")) {
				componentError.put("sensorId", Value.v("EY244"));
				componentError.put("faultType", Value.v("StuckOpen"));
				
				Sensor individualSensor = allSensors.get( "ESH244A" );
				componentError.put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(errorSensors.elementAt(i).get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
				
				return componentError;
			} 
		}
		
		// look for something failing utterly (CB, relay, inverter, ...)
		if(allSensors.get("E242").hitsZero) {
			componentError.put("sensorId", Value.v("EY260"));
			componentError.put("faultType", Value.v("StuckOpen"));
			
			Sensor individualSensor = allSensors.get( "E242" );
			Map<String, Value> errorS = null;
			for(int i=0; i<errorSensors.size(); i++)
				if(((StringValue)errorSensors.elementAt(i).get("sensorId")).get().equals("E242"))
					errorS = errorSensors.elementAt(i);
			componentError.put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(errorS.get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
			
			return componentError;	
		}
		if(allSensors.get("E265").hitsZero) {
			componentError.put("sensorId", Value.v("INV2"));
			componentError.put("faultType", Value.v("FailedOff"));
			
			Sensor individualSensor = allSensors.get( "E265" );
			Map<String, Value> errorS = null;
			for(int i=0; i<errorSensors.size(); i++)
				if(((StringValue)errorSensors.elementAt(i).get("sensorId")).get().equals("E265"))
					errorS = errorSensors.elementAt(i);
			componentError.put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(errorS.get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
			
			return componentError;	
		}
		if(allSensors.get("IT267").hitsZero) {
			// see whether E265 has an error
			boolean has265=false;
			for(int i=0; i<errorSensors.size(); i++)
				if(((StringValue)errorSensors.elementAt(i).get("sensorId")).get().equals("E265"))
					has265=true;
			
			if(has265) {
				componentError.put("sensorId", Value.v("CB262"));
				componentError.put("faultType", Value.v("FailedOpen"));
				
				Sensor individualSensor = allSensors.get( "IT267" );
				Map<String, Value> errorS = null;
				for(int i=0; i<errorSensors.size(); i++)
					if(((StringValue)errorSensors.elementAt(i).get("sensorId")).get().equals("IT267"))
						errorS = errorSensors.elementAt(i);
				componentError.put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(errorS.get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
				
				return componentError;	
			} 			
			componentError.put("sensorId", Value.v("CB266"));
			componentError.put("faultType", Value.v("FailedOpen"));
			
			Sensor individualSensor = allSensors.get( "IT267" );
			Map<String, Value> errorS = null;
			for(int i=0; i<errorSensors.size(); i++)
				if(((StringValue)errorSensors.elementAt(i).get("sensorId")).get().equals("IT267"))
					errorS = errorSensors.elementAt(i);
			componentError.put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(errorS.get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
			
			return componentError;	
		}
		if(allSensors.get("E281").hitsZero) {
			componentError.put("sensorId", Value.v("CB280"));
			componentError.put("faultType", Value.v("FailedOpen"));
			
			Sensor individualSensor = allSensors.get( "IT281" );
			Map<String, Value> errorS = null;
			for(int i=0; i<errorSensors.size(); i++)
				if(((StringValue)errorSensors.elementAt(i).get("sensorId")).get().equals("IT281"))
					errorS = errorSensors.elementAt(i);
			componentError.put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(errorS.get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
			
			return componentError;	
		}
		
		
		// FAN416 failed, over, under
		for(int i=0; i<errorSensors.size(); i++) {
			String id = ((StringValue)errorSensors.elementAt(i).get("sensorId")).get();
			if(id.equals("ST516")) {
				String faultType = ((StringValue)errorSensors.elementAt(i).get("faultType")).get();
				if( faultType == "Stuck" ) {
					componentError.put("sensorId", Value.v("FAN416"));
					componentError.put("faultType", Value.v("FailedOff"));
					
					Sensor individualSensor = allSensors.get( "ST516" );
					componentError.put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(errorSensors.elementAt(i).get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
					
					return componentError;
				} else {
					double val = ((RealValue)errorSensors.elementAt(i).get("Offset")).get();
					if(val > 0) {
						componentError.put("sensorId", Value.v("FAN416"));
						componentError.put("faultType", Value.v("OverSpeed"));
						
						Sensor individualSensor = allSensors.get( "ST516" );
						componentError.put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(errorSensors.elementAt(i).get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
						
						return componentError;
					} else {
						componentError.put("sensorId", Value.v("FAN416"));
						componentError.put("faultType", Value.v("UnderSpeed"));
						
						Sensor individualSensor = allSensors.get( "ST516" );
						componentError.put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(errorSensors.elementAt(i).get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
						
						return componentError;
					}
				}
			}
		}
		
		
		
		// next, resistance error
		if(errorSensors.size() <= 3) {
			Map<String, Value> s240=null;
			Map<String, Value> s281=null;
			Map<String, Value> s267=null;
			for(int i=0; i<errorSensors.size(); i++) { 
				String id = ((StringValue)errorSensors.elementAt(i).get("sensorId")).get(); 
				if(id.equals("IT240"))
					s240=errorSensors.elementAt(i);
				if(id.equals("IT281"))
					s281=errorSensors.elementAt(i);
				if(id.equals("IT267"))
					s267=errorSensors.elementAt(i);
			}
			if(s240!=null && s281!=null) {
				// this is a DC 485 error
				componentError.put("sensorId", Value.v("DC485"));
				
				if( ((StringValue)s281.get("faultType")).get().equals("Offset") ) {				
					double current = ((RealValue)s281.get("Offset")).get() + 24/9.5;
					double offset = (24 / current) - 9.5;
					if(offset > 100 || offset < -100) {
						componentError.put("faultType", Value.v("FailedOff"));
					} else {
						componentError.put("faultType", Value.v("ResistanceOffset"));
						componentError.put("Offset", Value.v(offset));
					}
				} else if( ((StringValue)s281.get("faultType")).get().equals("IntermittentOffset") ) {
					componentError.put("faultType", Value.v("IntermittentResistanceOffset"));
					
					double current = ((RealValue)s281.get("MeanOffset")).get() + 24/9.5;
					double offset = (24 / current) - 9.5;
					componentError.put("MeanOffset", Value.v(offset));
					
					componentError.put("MeanFaultDuration", s281.get("MeanFaultDuration"));
					componentError.put("MeanNominalDuration", s281.get("MeanNominalDuration"));
				} else {
					// drift
					componentError.put("faultType", Value.v("ResistanceDrift"));
					double current = ((RealValue)s281.get("Slope")).get() + 24/9.5;
					double offset = (24 / current) - 9.5;
					componentError.put("Slope", Value.v(offset));
				}
				
				Sensor individualSensor = allSensors.get( "IT281" );
				componentError.put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(s281.get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
			}
			if(s240!=null && s267!=null) {
				// this is a AC 483 error
				componentError.put("sensorId", Value.v("AC483"));
				
				if( ((StringValue)s267.get("faultType")).get().equals("Offset") ) {
					double current = ((RealValue)s267.get("Offset")).get() + 2.3743; 
					double totalResistence = 120.0 / current;
					double offset = 1/(1/totalResistence - 1/137.2492) - 80;
					if(offset > 100 || offset < -100) {
						componentError.put("faultType", Value.v("FailedOff"));
					} else {
						componentError.put("faultType", Value.v("ResistanceOffset"));
						componentError.put("Offset", Value.v(offset));
					}
				} else if( ((StringValue)s267.get("faultType")).get().equals("IntermittentOffset") ) {
					componentError.put("faultType", Value.v("IntermittentResistanceOffset"));
					
					double current = ((RealValue)s267.get("MeanOffset")).get() + 2.3743; 
					double totalResistence = 120.0 / current;
					double offset = 1/(1/totalResistence - 1/137.2492) - 80;
					componentError.put("MeanOffset", Value.v(offset));
					
					componentError.put("MeanFaultDuration", s267.get("MeanFaultDuration"));
					componentError.put("MeanNominalDuration", s267.get("MeanNominalDuration"));
				} else {
					// drift -- could be wrong?
					componentError.put("faultType", Value.v("ResistanceDrift"));
					
					double current = ((RealValue)s267.get("Slope")).get() + 2.3743; 
					double totalResistence = 120.0 / current;
					double offset = 1/(1/totalResistence - 1/137.2492) - 80;
					componentError.put("Slope", Value.v(offset));
				}
				
				Sensor individualSensor = allSensors.get( "IT267" );
				componentError.put("FaultTime", Value.v(individualSensor.timestamps.elementAt( ((IntegerValue)(s267.get("faultIndex"))).get() ) - individualSensor.timestamps.elementAt(0) ) );
			}
		}
		
		return componentError;
	}
}