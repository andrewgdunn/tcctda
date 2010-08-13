import org.dxc.api.connection.ConnectorFactory;
import org.dxc.api.connection.DxcConnector;
import org.dxc.api.datatypes.*;
import java.util.Iterator;
import java.util.Vector;

public class Sensor {
	public String id;
	public Vector<Value> data;
	public Vector<Long> timestamps;
	public boolean hitsZero;
	
	//---------------------------------------------------------------------------------------------
	
	public Sensor(String sensorID) {
		id = sensorID;
		data = new Vector();
		timestamps = new Vector();
		hitsZero = false;
	}
	
	//---------------------------------------------------------------------------------------------
	
	public void addData(Value value, long timestamp) {
		// make all Integers into Reals
		if(value instanceof IntegerValue)
			value = Value.v(new Double(((IntegerValue)value).get()));
		data.add(value);
		timestamps.add(timestamp);
		
		if( (value instanceof IntegerValue || value instanceof RealValue) && ((RealValue)value).get()<0.3 )
			hitsZero = true;
	}
	
	//---------------------------------------------------------------------------------------------
	
	public double meanThrough(int start, int numSamples) {
		double m = 0;
		for(int i=start; i<numSamples; i++)
			m += ((RealValue)data.elementAt(i)).get();
		m /= (numSamples-start);
		return m;
	}
	
	//---------------------------------------------------------------------------------------------
	
	public static double meanThrough(double[] data, int start, int numSamples) {
		double m = 0;
		for(int i=start; i<numSamples; i++)
			m += data[i];
		m /= (numSamples-start);
		return m;
	}
	
	//---------------------------------------------------------------------------------------------
	
	public double stdThrough(int start, int numSamples) {
		double m = meanThrough(0, numSamples);
		double std = 0;
		for(int i=start; i<numSamples; i++)
			std += Math.pow(Math.abs(((RealValue)data.elementAt(i)).get() - m), 2);
		std = Math.sqrt(std/((double)(numSamples-start)-1));
		return std;
	}
	
	//---------------------------------------------------------------------------------------------
	
	public static double stdThrough(double[] data, int start, int numSamples) {
		double m = meanThrough(data, 0, numSamples);
		double std = 0;
		for(int i=start; i<numSamples; i++)
			std += Math.pow(Math.abs(data[i] - m), 2);
		std = Math.sqrt(std/((double)(numSamples-start)-1));
		return std;
	}
	
	//---------------------------------------------------------------------------------------------
	
	public double minThrough(int numSamples) {
		double min = 99999999, val;
		for(int i=0; i<numSamples; i++) {
			val = ((RealValue)data.elementAt(i)).get();
			if(val < min)
				min = val;
		}
			
		return min;
	}
	
	//---------------------------------------------------------------------------------------------
	
	public double maxThrough(int numSamples) {
		double min = -99999999, val;
		for(int i=0; i<numSamples; i++) {
			val = ((RealValue)data.elementAt(i)).get();
			if(val > min)
				min = val;
		}
			
		return min;
	}
	
	//---------------------------------------------------------------------------------------------
	
	public void removeOutliers() {
		// removes the maximum and minimum values from the data (necessary for one scenario)
		if(data.elementAt(0) instanceof RealValue) {
			double min = 99999999, max=-99999999;
			int minI=-1, maxI=-1;
			for(int i=0; i<data.size(); i++) {
				double val = ((RealValue)data.elementAt(i)).get();
				if(val >= max) {
					max = val;
					maxI = i;
				}
				if(val <= min) {
					min=val;
					minI=i;
				}
			}
			if(minI > -1)
				data.removeElementAt( minI );
			// index for maxI will have decreased from the removal
			if(maxI > 0)
				data.removeElementAt( maxI-1 );
		}
	}
	
	//---------------------------------------------------------------------------------------------
}