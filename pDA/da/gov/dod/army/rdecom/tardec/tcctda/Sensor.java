package gov.dod.army.rdecom.tardec.tcctda;

import org.dxc.api.datatypes.*;

import java.util.Vector;

public class Sensor {
	public String id;
	public Vector<Value> data;
	public Vector<Long> timestamps;
	public boolean hitsZero;
	
	private double noiseFloor = 0.3;
	private int largePositiveNumber = 99999999;
	private int largeNegativeNumber = -99999999;
	
	public Sensor(String sensorID) {
		id = sensorID;
		data = new Vector<Value>();
		timestamps = new Vector<Long>();
		hitsZero = false;
	}
	
	public void addData(Value value, long timestamp) {
		// make all Integers into Reals
		if(value instanceof IntegerValue)
			value = Value.v(new Double(((IntegerValue)value).get()));
		data.add(value);
		timestamps.add(timestamp);
		
		// In some scenarios the noise will make it so that zero is never truely achieved, noiseFloor is our allowable range
		if( (value instanceof IntegerValue || value instanceof RealValue) && ((RealValue)value).get() < noiseFloor ) {
			hitsZero = true;
		}
			
	}
	
	public double meanThrough(int start, int numSamples) {
		double mean = 0;
		for(int index = start; index < numSamples; index++) {
			mean += ((RealValue)data.elementAt(index)).get();
		}			
		mean /= (numSamples-start);
		
		return mean;
	}
	
	public double stdThrough(int start, int numSamples) {
		double mean = meanThrough(0, numSamples);
		double std = 0;
		for(int index = start; index < numSamples; index++) {
			std += Math.pow(Math.abs(((RealValue)data.elementAt(index)).get() - mean), 2);
		}			
		std = Math.sqrt(std/((double)(numSamples-start)-1));
		
		return std;
	}

	public double minThrough(int numSamples) {
		double minimum = largePositiveNumber;
		double indexValue;
		for(int index = 0; index < numSamples; index++) {
			indexValue = ((RealValue)data.elementAt(index)).get();
			if(indexValue < minimum)
				minimum = indexValue;
		}			
		return minimum;
	}
	
	public double maxThrough(int numSamples) {
		double maximum = largeNegativeNumber;
		double indexValue;
		for(int index = 0; index < numSamples; index++) {
			indexValue = ((RealValue)data.elementAt(index)).get();
			if(indexValue > maximum)
				maximum = indexValue;
		}			
		return maximum;
	}
	
	public void removeOutliers() {
		// removes the maximum and minimum values from the data (necessary for one scenario)
		if(data.elementAt(0) instanceof RealValue) {
			double min = largePositiveNumber;
			double max= largeNegativeNumber;
			int minIndex=-1; 
			int maxIndex=-1;
			for(int index = 0; index < data.size(); index++) {
				double valueIndex = ((RealValue)data.elementAt(index)).get();
				if(valueIndex >= max) {
					max = valueIndex;
					maxIndex = index;
				}
				if(valueIndex <= min) {
					min=valueIndex;
					minIndex=index;
				}
			}
			if(minIndex > -1)
				data.removeElementAt(minIndex);
				timestamps.removeElementAt(minIndex);
			// index for maxI will have decreased from the removal
			if(maxIndex > 0)
				data.removeElementAt(maxIndex-1);
				timestamps.removeElementAt(maxIndex-1);
		}
	}
}