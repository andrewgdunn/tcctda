import org.dxc.api.connection.ConnectorFactory;
import org.dxc.api.connection.DxcConnector;
import org.dxc.api.datatypes.*;
import java.util.Iterator;
import java.util.Vector;

public class Sensor {
	public String id;
	public Vector<Value> data;
	public Vector<Long> timestamps;
	
	//---------------------------------------------------------------------------------------------
	
	public Sensor(String sensorID) {
		id = sensorID;
		data = new Vector();
		timestamps = new Vector();
	}
	
	//---------------------------------------------------------------------------------------------
	
	public void addData(Value value, long timestamp) {
		// make all Integers into Reals
		if(value instanceof IntegerValue)
			value = Value.v(new Double(((IntegerValue)value).get()));
		data.add(value);
		timestamps.add(timestamp);
	}
	
	//---------------------------------------------------------------------------------------------
	
	public double meanThrough(int numSamples) {
		double m = 0;
		for(int i=0; i<numSamples; i++)
			m += ((RealValue)data.elementAt(i)).get();
		m /= numSamples;
		return m;
	}
	
	//---------------------------------------------------------------------------------------------
	
	public double stdThrough(int numSamples) {
		double m = meanThrough(numSamples);
		double std = 0;
		for(int i=0; i<numSamples; i++)
			std += Math.pow(Math.abs(((RealValue)data.elementAt(i)).get() - m), 2);
		std = Math.sqrt(std/((double)numSamples-1));
		return std;
	}
	
	//---------------------------------------------------------------------------------------------
}