import org.dxc.api.datatypes.*;

import java.util.Map;
import java.util.HashMap;

public class ErrorFinder {
	
	public static int START_CHECK = 40;			// when to start looking for errors (to allow for a buffer for valid initial stats)
	public static int MAX_IDENTICAL = 80;		// how many identical readings allowed before we assume a stuck error
	public static int STD_RANGE = 8;			// maximum number of standard deviations off a reading can be before we assume an error
	public static int IGNORE = 1;				// minimum number of data errors before we assume the sensor is in error
	public static int INTERMITTENT_OFFSET = 5;	// number of sensor readings after abrupt error before we look for intermittent error
	public static int DRIFT_SMOOTH = 30;
	public static int DRIFT_STEP = 5;
		
	//---------------------------------------------------------------------------------------------
	
	public static Map<String, Value> errorParams(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		// if this is a boolean sensor ... do nothing for now
		if(!(sensor.data.elementAt(0) instanceof RealValue)) {
			boolean firstVal = ((BoolValue)sensor.data.elementAt(0)).get();
			boolean allSame = true;
			for(int i=1; i<sensor.data.size()-1; i++) {
				boolean val = ((BoolValue)sensor.data.elementAt(i)).get();
				if(val != firstVal) {
					allSame = false;
					break;
				}
			}
			if(allSame) {
				boolean val = ((BoolValue)sensor.data.elementAt(sensor.data.size()-1)).get();
				if(val!=firstVal) {
					map.put("booleanError", Value.v(val));
				}
			}
			return map;
		}			
		
		// overriding case: stuck error
		map = stuckErrorParams(sensor);
		if(map.size()>0)
			return map;
		
		// next look for intermittent
		map = intermittentErrorParams(sensor);
		if(map.size()>0)
			return map;
				
		// then abrupt
		map = abruptErrorParams(sensor);
		if(map.size()>0)
			return map;
		
		// finally drift
		map = driftErrorParams(sensor);
		if(map.size()>0)
			return map;
		return map;
	}
	
	//---------------------------------------------------------------------------------------------
	
	private static Map<String, Value> stuckErrorParams(Sensor sensor) {
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
	
	//---------------------------------------------------------------------------------------------
	
	private static Map<String, Value> abruptErrorParams(Sensor sensor) {
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
		}
		return map;
	}
	
//---------------------------------------------------------------------------------------------
	
	private static Map<String, Value> intermittentErrorParams(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		Map<String, Value> abrupt = abruptErrorParams(sensor);
		
		if(abrupt.size()==0)
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
			
			System.out.println("switches: " + switches);
			
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
	
//---------------------------------------------------------------------------------------------
	
	private static Map<String, Value> driftErrorParams(Sensor sensor) {
		Map<String, Value> map = new HashMap<String, Value>();
		
		// smooth data (to hopefully remove noise)
		int smoothCount = DRIFT_SMOOTH / 2;
		double sum=0, val;
		double[] smoothedData = new double[sensor.data.size()];
		// calculate initial sum
		for(int i=0; i<DRIFT_SMOOTH/2; i++) {
			val = ((RealValue)sensor.data.elementAt(i)).get();
			sum += val;
		}
		smoothedData[0] = sum / (double)smoothCount;
		
		for(int i=1; i<sensor.data.size(); i++) {
			if(i+DRIFT_SMOOTH/2-1 < sensor.data.size()) {
				val = ((RealValue)sensor.data.elementAt(i+DRIFT_SMOOTH/2-1)).get();
				sum += val;
				smoothCount++;
			}
			if(i-DRIFT_SMOOTH/2 >= 0) {
				val = ((RealValue)sensor.data.elementAt(i-DRIFT_SMOOTH/2)).get();
				sum -= val;
				smoothCount--;
			}
			smoothedData[i] = sum / (double)smoothCount;
		}
			
		// now check for monotonic
		double direction=0, lastVal;
		boolean monotonic = false;
		int i;
		for(i=START_CHECK; i<sensor.data.size()-START_CHECK; i++) {
			if(smoothedData[i] > smoothedData[i+DRIFT_STEP])
				direction = -1;
			else
				direction = 1;
			lastVal = smoothedData[i];
			monotonic = true;
			for(int j=i+DRIFT_STEP; j<sensor.data.size()-DRIFT_STEP; j+=DRIFT_STEP) {
				if(lastVal + (smoothedData[j]-lastVal)*direction <= lastVal) {
					monotonic=false;
					break;
				}
				lastVal = smoothedData[j];
			}
			if(monotonic)
				break;
		}
		
		if(monotonic) {
			// we have a drift error!
			map.put("faultIndex", Value.v(i));
			map.put("faultType", Value.v("Drift"));
		}
		return map;
	}
	
}