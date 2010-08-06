import org.dxc.api.connection.ConnectorFactory;
import org.dxc.api.connection.DxcConnector;
import org.dxc.api.datatypes.*;

import java.util.Iterator;
import java.util.Vector;
import java.util.Map;
import java.util.HashMap;

public class ErrorFinder {
	
	public static int START_CHECK = 40;			// when to start looking for errors (to allow for a buffer for valid initial stats)
	public static int MAX_IDENTICAL = 80;		// how many identical readings allowed before we assume a stuck error
	public static int STD_RANGE = 8;			// maximum number of standard deviations off a reading can be before we assume an error
	public static int IGNORE = 1;				// minimum number of data errors before we assume the sensor is in error
	public static int INTERMITTENT_OFFSET = 5;	// number of sensor readings after abrupt error before we look for intermittent error
	public static int DRIFT_SMOOTH = 20;
	public static int DRIFT_STEP = 5;
	public static int DRIFT_WINDOW_SIZE = 20;
		
	//---------------------------------------------------------------------------------------------
	
	public static Map<String, Value> errorParams(Sensor sensor) {
		Map<String, Value> map = new HashMap();
		
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
		Map<String, Value> map = new HashMap();
		
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
		Map<String, Value> map = new HashMap();
		
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
		Map<String, Value> map = new HashMap();
		
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
		Map<String, Value> map = new HashMap();
		
		// check for drift
		int errorPoint = -1;
		int start = sensor.data.size()/10;
		int end = sensor.data.size()/4;
		int direction = 0;
		
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
		if(((RealValue)sensor.data.elementAt(0)).get() > val)
			important = mins;
		
		double lastVal = important[0];
		int repeats=0;
		int cutoff = sensor.data.size()/15;
		
		// take out
		/*if(sensor.id.equals("IT281")) {
			for(int i=0; i<sensor.data.size(); i++)
				System.out.println(sensor.data.elementAt(i));
		}*/
		
		for(int i=1; i<sensor.data.size(); i++) {
			if(lastVal == important[i]) {
				repeats++;
				if(repeats > cutoff)
					errorPoint = -1;
			} else {
				if(repeats > cutoff)
					errorPoint = i;
				repeats=0;
			}
			lastVal = important[i];
		}
		
		
		if(errorPoint != -1 && errorPoint < sensor.data.size()*4/5) {
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