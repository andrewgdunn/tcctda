package gov.dod.army.rdecom.tardec.tcctda;

import org.dxc.api.datatypes.*;

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
public class SensorError {
	
	private static int START_CHECK = 40;			// when to start looking for errors (to allow for a buffer for valid initial stats)
	private static int MAX_IDENTICAL = 80;			// how many identical readings allowed before we assume a stuck error
	private static int STANDARD_RANGE = 8;			// maximum number of standard deviations off a reading can be before we assume an error
	private static int IGNORE = 1;					// minimum number of data errors before we assume the sensor is in error
	private static int INTERMITTENT_OFFSET = 5;		// number of sensor readings after abrupt error before we look for intermittent error
	private static int INTERMITTENT_UPANDLOW = 50;	// number of upper and lower 
	private static int DRIFT_WINDOW_SIZE = 20;
	private static int largePositiveNumber = 99999999;
	private static int largeNegativeNumber = -99999999;
	
	/**
	 * Takes in an individual sensor and directs it through any one of our detection search algorithms
	 * @param sensor -- the specific sensor we will be examining
	 * @return map -- a map that will contain specific values marking the fault index and parameters of a sensor
	 */
	public static Map<String, Value> findError(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		// Boolean Error, need to catch it because it will trip up the others.
		// it wouldn't if we were not doing so much silly casting of generic objects around here...
		if(!(sensor.data.elementAt(0) instanceof RealValue)) {
			map = booleanError(sensor);
			if(map.size() > 0)
				return map;
		}
		// Not boolean, well lets pass it on to the other detections
		else {		
			// overriding case: stuck error
			map = stuckError(sensor);
			if(map.size() > 0)
				return map;
			
			// next look for intermittent
			map = intermittentError(sensor);
			if(map.size() > 0)
				return map;
					
			// then abrupt
			map = abruptError(sensor);
			if(map.size() > 0)
				return map;
			
			// finally drift
			map = driftError(sensor);
			if(map.size() > 0)
				return map;
		}
		// Hopefully it was populated, otherwise coming back with nada
		return map;
	}
	
	/**
	 * Looks through the entire data sequence to see if there is an error on a sensor that is realvalue.
	 * These sensors are boolean style sensors so it compares the initial value to every value following to see if there is a difference.
	 * @param sensor -- The sensor to check through
	 * @return map -- the map will have a faultIndex if there is a problem.
	 */
	private static Map<String, Value> booleanError(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		boolean firstVal = ((BoolValue)sensor.data.elementAt(0)).get();
		
		for(int index = 1; index < sensor.data.size()-1; index++) {
			boolean indexValue = ( (BoolValue) sensor.data.elementAt(index)).get();
			if(indexValue != firstVal) {
				map.put("booleanError", Value.v(indexValue));
				map.put("faultIndex", Value.v(index));
				break;
			}
		}	
		return map;
	}
	
	/**
	 * Looks through the entire data sequence to tally up the amount of sequential similarities, if it is above the maximum we allow then we set a fault 
	 * @param sensor -- The sensor to check through
	 * @return map -- the map will have a faultIndex if there is a problem.
	 */
	private static Map<String, Value> stuckError(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		// Number of readings that are identical
		int numIdenticalReadings = 0;
		// Trailing data reading
		double lastReading = ( (RealValue) sensor.data.elementAt(0)).get();
		
		for(int index = 1; index < sensor.data.size(); index++) {
			double indexValue = ((RealValue)sensor.data.elementAt(index)).get();
			
			if(indexValue == lastReading) {
				numIdenticalReadings++;
			}
			else {
				numIdenticalReadings=0;			
			}
			
			lastReading = indexValue;
			
			if(index == sensor.data.size()-1 && numIdenticalReadings > MAX_IDENTICAL) {				
				map.put("StuckAt", Value.v(lastReading));
				map.put("faultIndex", Value.v(sensor.data.size() - numIdenticalReadings));
				map.put("faultType", Value.v("Stuck"));
				return map;
			}
		}
		return map;
	}
	
	/**
	 * Looks through the entire data sequence and tallies the times that the signal goes beyond a preset range of the standard deviation
	 * Will wait through a grace period to begin looking at the data stream
	 * @param sensor -- The sensor to check through
	 * @return map -- the map will have a faultIndex if there is a problem.
	 */
	private static Map<String, Value> abruptError(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		int errors = 0;
		
		for(int index = START_CHECK; index < sensor.data.size(); index++) {
			double indexValue = ((RealValue)sensor.data.elementAt(index)).get();
			double standardDeviation = sensor.sdThrough(0, index-1);
			double mean = sensor.meanThrough(0, index-1);
			 
			if(Math.abs(mean - indexValue) > STANDARD_RANGE * standardDeviation) {
				errors++;
			}
				
			
			if(errors > IGNORE) {
				map.put("Offset", Value.v(indexValue-mean));
				map.put("faultIndex", Value.v(index));
				map.put("faultType", Value.v("Offset"));
				return map;
			}
			
			// ST516 hack
			// the ST516 appears to be an anomoly, for this particular check. We now just need to check to see if the value falls outside of its standard bounds
			if(sensor.id.equals("ST516")) {
				if(indexValue > 960 || indexValue < 840) {
					map.put("Offset", Value.v( sensor.meanThrough(index, sensor.data.size()) - 900 ));
					map.put("faultIndex", Value.v(index));
					map.put("faultType", Value.v("Offset"));
					return map;
				}
			}
		}
		return map;
	}
	
	/**
	 * Nasty!
	 * @param sensor
	 * @return
	 */
	private static Map<String, Value> intermittentError(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		Map<String, Value> abruptError = abruptError(sensor);
		
		if(abruptError.size() == 0 || sensor.id.equals("ST516"))
			return map;
		
		int timeOfFault = ((IntegerValue)(abruptError.get("faultIndex"))).get() + INTERMITTENT_OFFSET; 
		
		// calculate min and max values after an abrupt error, instantiate them way out of bounds so that they get pulled back in by our checks
		double maximum = 0;
		double minimum = 999999;
		
		for(int index = timeOfFault; index < sensor.data.size(); index++) {
			double indexValue = ((RealValue)sensor.data.elementAt(index)).get();
			if(indexValue > maximum)
				maximum=indexValue;
			if(indexValue < minimum)
				minimum=indexValue;
		}
		
		// Need quite a bit many more variables to track this next bit
		int countUpper = 0;
		int countLower = 0;
		int switches = 1;
		double meanUpper = 0;
		double meanLower = 0;
		double last = 0;
		double mean = sensor.meanThrough(0, timeOfFault-5);
		
		// calculate upper and lower means
		for(int index = timeOfFault; index < sensor.data.size(); index++) {
			double indexValue = ( (RealValue) sensor.data.elementAt(index)).get();
			if(maximum - indexValue > indexValue - minimum) {
				meanLower += indexValue;
				countLower++;
			} 
			else {
				meanUpper += indexValue;
				countUpper++;
			}
		}
		meanUpper = meanUpper / (double) countUpper;
		meanLower = meanLower / (double) countLower;
		
		double standardDeviation = sensor.sdThrough(0, timeOfFault-6);
		if((meanUpper - meanLower) / standardDeviation > STANDARD_RANGE) {
			if (countUpper / (double) countLower < INTERMITTENT_UPANDLOW && countLower / (double) countUpper < INTERMITTENT_UPANDLOW) {
				// look for "switches"
				for(int index = timeOfFault; index < sensor.data.size(); index++) {
					double indexValue = ( (RealValue) sensor.data.elementAt(index)).get();
					
					if(maximum - indexValue > indexValue - minimum) {
						if(last == meanUpper)
							switches++;
						last = meanLower;
					}
					else {
						if(last == meanLower)
							switches++;
						last = meanUpper;
					}	
				}			
			}
			
			if(switches % 2 == 1)
				switches++;
			switches /= 2;
			
			int sensorFrequency = (int)(1000 / ( (sensor.timestamps.elementAt(sensor.timestamps.size()-1)-sensor.timestamps.elementAt(0)) / sensor.timestamps.size() ));
			
			double meanTimeNominal = 0; 
			double meanTimeError = 0;
			double difference;
			
			if(Math.abs(meanUpper - mean) > Math.abs(meanLower - mean)) {
				meanTimeNominal = countLower / (double)(switches * sensorFrequency);
				meanTimeError = (INTERMITTENT_OFFSET + countUpper) / (double)(switches * sensorFrequency);
				difference = meanUpper - meanLower;
			} 
			else {
				meanTimeNominal = countUpper / (double)(switches*sensorFrequency);
				meanTimeError = (INTERMITTENT_OFFSET+countLower)/(double)(switches*sensorFrequency);
				difference = meanLower-meanUpper;
			}
			
			map.put("faultIndex", Value.v(timeOfFault - INTERMITTENT_OFFSET));
			map.put("MeanOffset", Value.v(difference));
			map.put("MeanFaultDuration", Value.v(meanTimeError));
			map.put("MeanNominalDuration", Value.v(meanTimeNominal));

			map.put("faultType", Value.v("IntermittentOffset"));
			
			return map;
		}
		
		// all that for nothing
		return map;
	}
	
	/**
	 * Nasty!
	 * @param sensor
	 * @return
	 */
	private static Map<String, Value> driftError(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		// check for drift
		int errorPoint = -1;
		// calculate min and max vectors
		double[] maxArray = new double[sensor.data.size()];
		double[] minArray = new double[sensor.data.size()];
		double maxSingle;
		double minSingle;
		double indexValue;
		
		for(int index = 0; index < sensor.data.size(); index++) {
			maxSingle = largeNegativeNumber;
			minSingle = largePositiveNumber;
			for(int jindex = 0; jindex <= index; jindex++) {
				indexValue = ((RealValue)sensor.data.elementAt(jindex)).get(); 
				if(indexValue < minSingle)
					minSingle = indexValue; 
				if(indexValue > maxSingle)
					maxSingle = indexValue;
			}
			minArray[index] = minSingle;
			maxArray[index] = maxSingle;
		}
		
		
		double[] importantArray;
		
		// Snag the last value
		indexValue = ((RealValue)sensor.data.elementAt(sensor.data.size()-1)).get();
		// if the first value is greater than the last value, set the importantArray to minArray
		if(((RealValue)sensor.data.elementAt(0)).get() > indexValue) {
			importantArray = minArray;
		}
		else {
			importantArray = maxArray;
		}
			
		// Need to hold onto your previous value
		double previousValue = importantArray[0];
		// count the amount of repeats
		int repeats = 0;
		// cut off 
		int cutoff = sensor.data.size() / 15;

		for(int index = 1; index < sensor.data.size(); index++) {
			if(previousValue == importantArray[index]) {
				repeats++;
				if(repeats > cutoff) {
					errorPoint = -1;
				}
			} else {
				if(repeats > cutoff)
					errorPoint = index;
				repeats=0;
			}
			previousValue = importantArray[index];
		}
		
		if(errorPoint != -1 && errorPoint < sensor.data.size() * 7/8) {
			// we have a drift error!
			map.put("faultIndex", Value.v(errorPoint));
			map.put("index", Value.v(errorPoint));
			map.put("faultType", Value.v("Drift"));
			
			indexValue = ((RealValue)sensor.data.elementAt(sensor.data.size()-1)).get();
			int sensorFrequency = (int)(1000 / ( (sensor.timestamps.elementAt(sensor.timestamps.size()-1)-sensor.timestamps.elementAt(0)) / sensor.timestamps.size() ));
			
			map.put("Slope", Value.v(sensorFrequency * (indexValue-((RealValue)sensor.data.elementAt(0)).get()) / (sensor.data.size()-errorPoint)));
		}
		return map;
	}
}