import org.dxc.api.connection.ConnectorFactory;
import org.dxc.api.connection.DxcConnector;
import org.dxc.api.datatypes.*;
import java.util.Iterator;
import java.util.Vector;
import java.util.Map;
import java.util.HashMap;

public class ComponentError {
	public static Map<String, Value> finalError(Vector<Map<String, Value>> errorSensors) {
		// returns the component error to report to the Oracle
		Map<String, Value> componentError = new HashMap();
		
		// if there is only one error, it should be a sensor error, we can just report it back directly
		if(errorSensors.size()==1)
        	return errorSensors.elementAt(0);
		
		// next simplest -- if there is a boolean error
		for(int i=0; i<errorSensors.size(); i++) {
			String id = ((StringValue)errorSensors.elementAt(i).get("sensorId")).get(); 
			if(id.equals("ISH236")) {
				componentError.put("sensorId", Value.v("CB236"));
				componentError.put("faultType", Value.v("FailedOpen"));
				componentError.put("FaultTime", errorSensors.elementAt(i).get("FaultTime"));
				return componentError;
			} else if(id.equals("ESH244A")) {
				componentError.put("sensorId", Value.v("EY244"));
				componentError.put("faultType", Value.v("StuckOpen"));
				componentError.put("FaultTime", errorSensors.elementAt(i).get("FaultTime"));
				return componentError;
			}
		}
		
		// next, resistance error
		if(errorSensors.size() <= 3) {
			
		}
		
		return componentError();
	}
}