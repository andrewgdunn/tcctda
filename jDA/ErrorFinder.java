import org.dxc.api.connection.ConnectorFactory;
import org.dxc.api.connection.DxcConnector;
import org.dxc.api.datatypes.*;
import java.util.Iterator;
import java.util.Vector;

public class ErrorFinder {
	
	public static int START_CHECK = 20;		// when to start looking for errors (to allow for a buffer for valid initial stats)
	public static int MAX_IDENTICAL = 30;	// how many identical readings allowed before we assume a stuck error
	public static int STD_RANGE = 10;		// maximum number of standard deviations off a reading can be before we assume an error
	
	//---------------------------------------------------------------------------------------------
	
	public static long timeOfError(Sensor sensor) {
		// if this is a boolean sensor ... do nothing for now
		if(!(sensor.data.elementAt(0) instanceof RealValue))
			return -1;
		
		long returnValue = timeOfStuckError(sensor);
		if(returnValue != -1)
			return returnValue;
		
		returnValue = timeOfAbruptError(sensor);
		if(returnValue != -1)
			return returnValue;
		
		return -1;
	}
	
	//---------------------------------------------------------------------------------------------
	
	private static long timeOfStuckError(Sensor sensor) {
		int numIdenticalReadings = 0;
		double lastReading = ((RealValue)sensor.data.elementAt(0)).get();
		
		for(int i=1; i<sensor.data.size(); i++) {
			double val = ((RealValue)sensor.data.elementAt(i)).get();
			
			if(val==lastReading)
				numIdenticalReadings++;
			else
				numIdenticalReadings=0;
			
			lastReading = val;
			
			if(i>START_CHECK && numIdenticalReadings > MAX_IDENTICAL)
				return sensor.timestamps.elementAt(i-numIdenticalReadings+1);
		}
		return -1;
	}
	
	//---------------------------------------------------------------------------------------------
	
	private static long timeOfAbruptError(Sensor sensor) {
		for(int i=START_CHECK; i<sensor.data.size(); i++) {
			double val = ((RealValue)sensor.data.elementAt(i)).get();
			double std = sensor.stdThrough(i-1);
			double mean = sensor.meanThrough(i-1);
			
			if(Math.abs(mean-val) > STD_RANGE*std)
				return sensor.timestamps.elementAt(i);
		}
		return -1;
	}
}