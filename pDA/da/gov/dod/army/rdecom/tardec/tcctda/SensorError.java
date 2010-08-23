package gov.dod.army.rdecom.tardec.tcctda;

import org.dxc.api.datatypes.*;

import java.util.Map;
import java.util.HashMap;

public class SensorError {
	
	public static int START_CHECK = 40;			// when to start looking for errors (to allow for a buffer for valid initial stats)
	public static int MAX_IDENTICAL = 80;		// how many identical readings allowed before we assume a stuck error
	public static int STD_RANGE = 8;			// maximum number of standard deviations off a reading can be before we assume an error
	public static int IGNORE = 1;				// minimum number of data errors before we assume the sensor is in error
	public static int INTERMITTENT_OFFSET = 5;	// number of sensor readings after abrupt error before we look for intermittent error
	public static int DRIFT_WINDOW_SIZE = 20;
	
	/**
	 * Takes in an individual sensor and 
	 * @param sensor
	 * @return
	 */
	public static Map<String, Value> findError(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		// Boolean Error
		map = booleanError(sensor);
		if(map.size() > 0)
			return map;			
		
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
		
		if(!(sensor.data.elementAt(0) instanceof RealValue)) {
			boolean firstVal = ((BoolValue)sensor.data.elementAt(0)).get();
			for(int index = 1; index < sensor.data.size()-1; index++) {
				boolean indexValue = ( (BoolValue) sensor.data.elementAt(index)).get();
				if(indexValue != firstVal) {
					map.put("booleanError", Value.v(indexValue));
					map.put("faultIndex", Value.v(index));
					break;
				}
			}
		}	
		return map;
	}
		
	private static Map<String, Value> stuckError(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		int numIdenticalReadings = 0;
		double lastReading = ((RealValue)sensor.data.elementAt(0)).get();
		
		for(int i=1; i<sensor.data.size(); i++) {
			double val = ((RealValue)sensor.data.elementAt(i)).get();
			
			if(val==lastReading)
				numIdenticalReadings++;
			else
				numIdenticalReadings=0;
			
			lastReading = val;
			
			if(i==sensor.data.size()-1 && numIdenticalReadings > MAX_IDENTICAL) {				
				map.put("StuckAt", Value.v(lastReading));
				map.put("faultIndex", Value.v(sensor.data.size() - numIdenticalReadings));
				map.put("faultType", Value.v("Stuck"));
				return map;
			}
		}
		return map;
	}
	
	private static Map<String, Value> abruptError(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		int errors=0;
		for(int i=START_CHECK; i<sensor.data.size(); i++) {
			double val = ((RealValue)sensor.data.elementAt(i)).get();
			double std = sensor.stdThrough(0, i-1);
			double mean = sensor.meanThrough(0, i-1);
			
			if(Math.abs(mean-val) > STD_RANGE*std)
				errors++;
			
			if(errors > IGNORE) {
				map.put("Offset", Value.v(val-mean));
				map.put("faultIndex", Value.v(i));
				map.put("faultType", Value.v("Offset"));
				return map;
			}
			
			// ST516 hack
			if(sensor.id.equals("ST516")) {
				if(val > 960 || val < 840) {
					map.put("Offset", Value.v( sensor.meanThrough(i, sensor.data.size()) - 900 ));
					map.put("faultIndex", Value.v(i));
					map.put("faultType", Value.v("Offset"));
					return map;
				}
			}
		}
		return map;
	}
	
	private static Map<String, Value> intermittentError(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		Map<String, Value> abrupt = abruptError(sensor);
		
		if(abrupt.size()==0 || sensor.id.equals("ST516"))
			return map;
		
		int t = ((IntegerValue)(abrupt.get("faultIndex"))).get() + INTERMITTENT_OFFSET; 
		
		// calculate min and max values after error
		double max=0;
		double min=999999;
		for(int i=t; i<sensor.data.size(); i++) {
			double val = ((RealValue)sensor.data.elementAt(i)).get();
			if(val > max)
				max=val;
			if(val < min)
				min=val;
		}
		
		int countU=0, countL=0, switches=1;
		double meanU = 0, meanL = 0, last=0, mean = sensor.meanThrough(0, t-5);
		
		// calculate upper and lower means
		for(int i=t; i<sensor.data.size(); i++) {
			double val = ((RealValue)sensor.data.elementAt(i)).get();
			if(max-val > val-min) {
				meanL+=val;
				countL++;
			} else {
				meanU+=val;
				countU++;
			}
		}
		meanU = meanU / (double)countU;
		meanL = meanL / (double)countL;
		
		double oldStd = sensor.stdThrough(0, t-6);
		if((meanU-meanL)/oldStd > STD_RANGE  && countU/(double)countL < 50 && countL/(double)countU < 50) {
			// look for "switches"
			for(int i=t; i<sensor.data.size(); i++) {
				double val = ((RealValue)sensor.data.elementAt(i)).get();
				if(max-val > val-min) {
					if(last==meanU)
						switches++;
					last = meanL;
				} else {
					if(last==meanL)
						switches++;
					last = meanU;
				}
			}
			
			if(switches%2==1)
				switches++;
			switches /= 2;
			int sensorFrequency = (int)(1000 / ( (sensor.timestamps.elementAt(sensor.timestamps.size()-1)-sensor.timestamps.elementAt(0)) / sensor.timestamps.size() ));
			
			double meanTimeNominal=0, meanTimeError=0, diff;
			if(Math.abs(meanU-mean) > Math.abs(meanL-mean)) {
				meanTimeNominal = countL/(double)(switches*sensorFrequency);
				meanTimeError = (INTERMITTENT_OFFSET+countU)/(double)(switches*sensorFrequency);
				diff = meanU-meanL;
			} else {
				meanTimeNominal = countU/(double)(switches*sensorFrequency);
				meanTimeError = (INTERMITTENT_OFFSET+countL)/(double)(switches*sensorFrequency);
				diff = meanL-meanU;
			}
			
			map.put("MeanOffset", Value.v(diff));
			map.put("MeanFaultDuration", Value.v(meanTimeError));
			map.put("MeanNominalDuration", Value.v(meanTimeNominal));
			map.put("faultIndex", Value.v(t - INTERMITTENT_OFFSET));
			map.put("faultType", Value.v("IntermittentOffset"));
			
			return map;
		}

		return map;
	}
	
	private static Map<String, Value> driftError(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		// check for drift
		int errorPoint = -1;
		// calculate min and max vectors
		double[] maxs = new double[sensor.data.size()];
		double[] mins = new double[sensor.data.size()];
		double theMax=-999999, theMin=99999999, val;
		for(int i=0; i<sensor.data.size(); i++) {
			theMax=-999999;
			theMin=9999999;
			for(int j=0; j<=i; j++) {
				val = ((RealValue)sensor.data.elementAt(j)).get(); 
				if(val < theMin)
					theMin = val; 
				if(val > theMax)
					theMax = val;
			}
			mins[i]=theMin;
			maxs[i]=theMax;
		}
		double[] important;
		val = ((RealValue)sensor.data.elementAt(sensor.data.size()-1)).get();
		important=maxs;
		if(((RealValue)sensor.data.elementAt(0)).get() > val) {
			important = mins;
		}
			
		
		double lastVal = important[0];
		int repeats=0;
		int cutoff = sensor.data.size()/15;

		for(int i=1; i<sensor.data.size(); i++) {
			if(lastVal == important[i]) {
				repeats++;
				if(repeats > cutoff) {
					errorPoint = -1;
				}
			} else {
				if(repeats > cutoff)
					errorPoint = i;
				repeats=0;
			}
			lastVal = important[i];
		}
		
		if(errorPoint != -1 && errorPoint < sensor.data.size()*7/8) {
			// we have a drift error!
			map.put("faultIndex", Value.v(errorPoint));
			map.put("index", Value.v(errorPoint));
			map.put("faultType", Value.v("Drift"));
			val = ((RealValue)sensor.data.elementAt(sensor.data.size()-1)).get();
			int sensorFrequency = (int)(1000 / ( (sensor.timestamps.elementAt(sensor.timestamps.size()-1)-sensor.timestamps.elementAt(0)) / sensor.timestamps.size() ));
			
			map.put("Slope", Value.v(sensorFrequency * (val-((RealValue)sensor.data.elementAt(0)).get()) / (sensor.data.size()-errorPoint)));
		}
		return map;
	}
	
}